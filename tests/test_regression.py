"""Torch-free tests for the regression probe and metrics.

These exercise :mod:`torchgeo_bench.regression` directly (NumPy only), so they
run without the model stack. The end-to-end runner wiring is covered in
``test_building_value_regression.py`` (gated on torch).
"""

import numpy as np

from torchgeo_bench.regression import (
    RidgeRegression,
    bootstrap_rmsle,
    regression_metrics,
    select_alpha,
    value_decile_slices,
)


def _make_data(seed=0, n=800, d=16, noise=0.3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    w = rng.normal(size=d)
    y = np.exp(3.0 + X @ w * 0.5 + rng.normal(scale=noise, size=n))
    return X, y


def test_ridge_learns_signal():
    X, y = _make_data()
    tr, va, te = slice(0, 500), slice(500, 650), slice(650, 800)
    best_alpha, model = select_alpha(
        X[tr], y[tr], X[va], y[va], [0.01, 0.1, 1, 10, 100]
    )
    m = regression_metrics(y[te], model.predict(X[te]))
    assert best_alpha in {0.01, 0.1, 1, 10, 100}
    assert m["rmsle"] < 0.6          # clearly better than a constant predictor
    assert m["r2"] > 0.8
    assert m["within_factor_2"] > 0.8


def test_perfect_prediction():
    _, y = _make_data()
    m = regression_metrics(y, y.copy())
    assert m["rmsle"] == 0.0
    assert m["within_factor_1.25"] == 1.0
    assert abs(m["r2"] - 1.0) < 1e-9


def test_log_bias_sign():
    y = np.array([100.0, 100.0, 100.0])
    assert regression_metrics(y, y * 2)["mean_log_ratio"] > 0   # over-valuation
    assert regression_metrics(y, y / 2)["mean_log_ratio"] < 0   # under-valuation


def test_metrics_ignore_nan():
    yt = np.array([100.0, np.nan, 300.0])
    yp = np.array([110.0, 200.0, np.nan])
    assert regression_metrics(yt, yp)["n"] == 1


def test_exposure_weighting_favours_high_value():
    # Same absolute error everywhere; exposure weighting must exceed plain MAE
    # because the weight concentrates on the high-value item.
    yt = np.array([10.0, 1000.0])
    yp = np.array([20.0, 1010.0])
    m = regression_metrics(yt, yp)
    assert m["mae"] == 10.0
    assert m["exposure_weighted_mae"] == 10.0  # equal errors -> equals MAE
    # now make the big one wrong: exposure MAE should jump above plain MAE
    yp2 = np.array([20.0, 1100.0])
    m2 = regression_metrics(yt, yp2)
    assert m2["exposure_weighted_mae"] > m2["mae"]


def test_value_decile_slices():
    _, y = _make_data()
    sl = value_decile_slices(y)
    assert set(sl) == {"value_bottom_10pct", "value_mid", "value_top_10pct"}
    # masks are disjoint and cover all finite entries
    total = sum(int(m.sum()) for m in sl.values())
    assert total == np.isfinite(y).sum()


def test_bootstrap_ci_brackets_point():
    X, y = _make_data()
    tr, te = slice(0, 600), slice(600, 800)
    model = RidgeRegression(alpha=1.0).fit(X[tr], y[tr])
    pt, lo, hi = bootstrap_rmsle(y[te], model.predict(X[te]), n_boot=100, seed=1)
    assert lo <= pt <= hi


def test_ridge_rejects_bad_input():
    import pytest

    with pytest.raises(ValueError):
        RidgeRegression(alpha=-1.0)
    with pytest.raises(ValueError):
        RidgeRegression().fit(np.zeros((3, 2)), np.zeros(4))
