"""Ridge-regression linear probe and regression metrics for torchgeo-bench.

This module adds a ``"regression"`` evaluation path that mirrors the frozen-
backbone + linear-probe methodology used for classification, but for a
*continuous* target (e.g. per-building structure value). It is deliberately
torch-free (NumPy + a closed-form ridge) so it can be unit-tested without a GPU
or the heavy model stack, and so the probe is fast enough to sweep on CPU.

Why these metrics
-----------------
Building value is heavy-tailed and log-distributed, so plain RMSE is dominated
by a few high-value structures. The primary metric is therefore **RMSLE**
(root-mean-squared log error). We also report:

* ``mae`` / ``rmse`` — raw-scale errors for interpretability.
* ``mape`` / ``median_ape`` — relative error (median is outlier-robust).
* ``within_factor_1.25`` / ``within_factor_2`` — fraction of predictions within
  a multiplicative factor of truth (a "good enough for exposure" view).
* ``mean_log_ratio`` — calibration/bias term; >0 means systematic
  over-valuation, <0 under-valuation.
* ``exposure_weighted_mae`` — MAE weighted by true value, because for disaster
  risk the dollars at stake matter more than the count of buildings.
* ``r2`` — variance explained.

Slice reporting (overall + informal/formal + value deciles) is applied by the
runner using group columns declared on the dataset.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ridge-regression probe (closed form, log-space target)
# ---------------------------------------------------------------------------
class RidgeRegression:
    """Closed-form ridge regression with a standardized feature space.

    Fits ``w`` minimizing ``||Xw - y||^2 + alpha * ||w||^2`` (bias unpenalized).
    Features are z-scored using train statistics for numerical stability, which
    also makes a single ``alpha`` behave comparably across backbones with very
    different feature scales.

    When ``log_target=True`` the model is trained on ``log(y)`` and predictions
    are exponentiated back, which is the right space for multiplicative,
    heavy-tailed value targets.
    """

    def __init__(self, alpha: float = 1.0, log_target: bool = True) -> None:
        if alpha < 0:
            raise ValueError("alpha must be >= 0")
        self.alpha = float(alpha)
        self.log_target = bool(log_target)
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeRegression":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D; got {X.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y length mismatch")

        self._mu = X.mean(axis=0)
        self._sigma = X.std(axis=0) + 1e-8
        Xs = (X - self._mu) / self._sigma

        if self.log_target:
            if np.any(y <= 0):
                # guard: shift is not appropriate for values; drop/clip below.
                y = np.clip(y, 1e-6, None)
            target = np.log(y)
        else:
            target = y

        n, d = Xs.shape
        Xb = np.hstack([Xs, np.ones((n, 1))])
        reg = self.alpha * np.eye(d + 1)
        reg[-1, -1] = 0.0  # do not penalize bias
        A = Xb.T @ Xb + reg
        self._w = np.linalg.solve(A, Xb.T @ target)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("RidgeRegression not fitted")
        X = np.asarray(X, dtype=np.float64)
        Xs = (X - self._mu) / self._sigma
        Xb = np.hstack([Xs, np.ones((Xs.shape[0], 1))])
        pred = Xb @ self._w
        if self.log_target:
            # A near-constant train feature (sigma ~ 1e-8) can be amplified into
            # an extreme z-score by an unseen test value, sending the log-space
            # prediction far enough that np.exp overflows to +inf. Those infs are
            # silently dropped downstream (_clean), biasing raw-scale metrics, so
            # clamp to a generous ceiling (exp(40) ~ 2.4e17 >> any real value).
            pred = np.clip(pred, None, 40.0)
            return np.exp(pred)
        return pred


def select_alpha(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    alphas: list[float],
    log_target: bool = True,
) -> tuple[float, RidgeRegression]:
    """Pick alpha by validation RMSLE (or RMSE when log_target=False)."""
    best_alpha = alphas[0]
    best_score = np.inf
    for a in alphas:
        model = RidgeRegression(alpha=a, log_target=log_target).fit(x_train, y_train)
        pred = model.predict(x_val)
        score = _rmsle(y_val, pred) if log_target else _rmse(y_val, pred)
        if score < best_score:
            best_score, best_alpha = score, a
    final = RidgeRegression(alpha=best_alpha, log_target=log_target).fit(x_train, y_train)
    return best_alpha, final


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _clean(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[mask], y_pred[mask]


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt, yp = _clean(y_true, y_pred)
    return float(np.sqrt(np.mean((yp - yt) ** 2))) if yt.size else float("nan")


def _rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt, yp = _clean(y_true, y_pred)
    pos = (yt > 0) & (yp > 0)
    if not pos.any():
        return float("nan")
    return float(np.sqrt(np.mean((np.log(yp[pos]) - np.log(yt[pos])) ** 2)))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Full metric dict for a set of continuous predictions."""
    yt, yp = _clean(y_true, y_pred)
    n = yt.size
    if n == 0:
        return {"n": 0}

    err = yp - yt
    abs_err = np.abs(err)
    pos = (yt > 0) & (yp > 0)
    if pos.any():
        lr = np.log(yp[pos]) - np.log(yt[pos])
        rmsle = float(np.sqrt(np.mean(lr**2)))
        mean_log_ratio = float(np.mean(lr))
        within_1_25 = float(np.mean(np.abs(lr) < np.log(1.25)))
        within_2 = float(np.mean(np.abs(lr) < np.log(2.0)))
    else:
        rmsle = mean_log_ratio = within_1_25 = within_2 = float("nan")

    nz = yt != 0
    ape = abs_err[nz] / np.abs(yt[nz])
    w = yt.clip(min=0)
    exp_wmae = float(np.sum(w * abs_err) / np.sum(w)) if np.sum(w) > 0 else float("nan")

    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {
        "n": int(n),
        "mae": float(np.mean(abs_err)),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "rmsle": rmsle,
        "mape": float(np.mean(ape)) if ape.size else float("nan"),
        "median_ape": float(np.median(ape)) if ape.size else float("nan"),
        "exposure_weighted_mae": exp_wmae,
        "mean_log_ratio": mean_log_ratio,
        "within_factor_1.25": within_1_25,
        "within_factor_2": within_2,
        "r2": r2,
    }


def value_decile_slices(y_true: np.ndarray) -> dict[str, np.ndarray]:
    """Boolean masks for bottom-10% / mid / top-10% of true value."""
    yt = np.asarray(y_true, dtype=float)
    finite = np.isfinite(yt)
    out: dict[str, np.ndarray] = {}
    if finite.sum() < 10:
        return out
    q = np.nanquantile(yt, [0.1, 0.9])
    out["value_bottom_10pct"] = finite & (yt <= q[0])
    out["value_mid"] = finite & (yt > q[0]) & (yt < q[1])
    out["value_top_10pct"] = finite & (yt >= q[1])
    return out


def bootstrap_rmsle(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_boot: int = 200,
    ci: float = 95.0,
    seed: int | None = None,
) -> tuple[float, float, float]:
    """Bootstrap RMSLE point estimate + CI. Returns (mean, lo, hi)."""
    yt, yp = _clean(y_true, y_pred)
    point = _rmsle(yt, yp)
    n = yt.size
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals.append(_rmsle(yt[idx], yp[idx]))
    vals = np.array([v for v in vals if np.isfinite(v)], dtype=float)
    if vals.size == 0:
        return point, point, point
    lo = float(np.percentile(vals, (100 - ci) / 2))
    hi = float(np.percentile(vals, 100 - (100 - ci) / 2))
    return point, lo, hi
