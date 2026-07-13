"""End-to-end-ish tests for the building-value regression path.

Gated on torch (the dataset yields tensors and the runner helper lives in
``main`` which imports the model stack). Covers:

* the dataset registers, loads, and honours the label convention
  ``[structure_value, is_informal]``;
* ``evaluate_regression`` turns frozen features into per-metric result rows,
  including slice rows, and a learnable synthetic signal is actually learned.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def test_dataset_registered_and_regression_task():
    from torchgeo_bench.datasets import get_bench_dataset_class, list_datasets

    assert "building-value" in list_datasets()
    assert "building-value-transfer" in list_datasets()
    cls = get_bench_dataset_class("building-value")
    assert cls.task == "regression"
    assert cls.num_classes == 1
    assert cls.regression_group_names == ["is_informal"]


def test_sample_shape_and_label_convention():
    from torchgeo_bench.datasets import get_bench_dataset_class

    bench = get_bench_dataset_class("building-value")()
    ds = bench.get_dataset("test", bands=tuple(bench.rgb_bands))
    sample = ds[0]
    img, label = sample["image"], sample["label"]
    assert img.shape[0] == 3            # rgb
    assert img.ndim == 3
    assert label.shape == (2,)          # [value, is_informal]
    assert label[0] > 0                 # positive structure value
    assert label[1] in (0.0, 1.0)


def test_transfer_variant_is_a_different_city():
    from torchgeo_bench.datasets import get_bench_dataset_class

    a = get_bench_dataset_class("building-value")()
    b = get_bench_dataset_class("building-value-transfer")()
    assert a.city != b.city
    # Different value distributions -> different median (metro_b is cheaper).
    va = np.array([a.get_dataset("test")[i]["label"][0].item() for i in range(300)])
    vb = np.array([b.get_dataset("test")[i]["label"][0].item() for i in range(300)])
    assert np.median(vb) < np.median(va)


def test_evaluate_regression_produces_metric_rows():
    from torchgeo_bench.main import _split_regression_labels, evaluate_regression

    # Synthetic frozen "features" linearly related to log-value, plus an
    # informal group column so slice rows are produced. The weight vector is
    # fixed across splits so the mapping is consistent and thus learnable.
    d = 12
    w = np.random.default_rng(0).normal(size=d)

    def make(n, seed):
        r = np.random.default_rng(seed)
        X = r.normal(size=(n, d))
        val = np.exp(3.0 + X @ w * 0.4 + r.normal(scale=0.3, size=n))
        informal = (r.random(n) < 0.3).astype(float)
        y = np.stack([val, informal], axis=1)
        return X.astype(np.float32), y.astype(np.float32)

    xtr, ytr = make(500, 1)
    xva, yva = make(150, 2)
    xte, yte = make(150, 3)

    common_meta = dict(
        dataset="building-value", seed=0, model="test.Model", name="test",
        normalization="bandspec_zscore", image_size=32, interpolation="bilinear",
        partition="default", bands="rgb", c_range_start=-6, c_range_stop=4,
        c_range_num=10, merge_val=True, bootstrap=50,
    )
    alphas = (10.0 ** np.linspace(-6, 4, 10)).tolist()

    rows = evaluate_regression(
        xtr, ytr, xva, yva, xte, yte,
        alphas=alphas, group_names=["is_informal"],
        seed=0, n_bootstrap=50, merge_val=True,
        common_meta=common_meta, feature_dim=d,
    )
    names = {r["metric_name"] for r in rows}
    assert "rmsle" in names
    assert "within_factor_2" in names
    # slice rows present for informal and its complement + value deciles
    assert any(n.startswith("rmsle@is_informal") for n in names)
    assert any(n.startswith("rmsle@not_is_informal") for n in names)
    assert any(n.startswith("rmsle@value_top_10pct") for n in names)

    # all rows are ridge rows with a bootstrap CI on the headline metric
    assert all(r["method"] == "ridge" for r in rows)
    rmsle_row = next(r for r in rows if r["metric_name"] == "rmsle")
    assert rmsle_row["ci_lower"] <= rmsle_row["metric_value"] <= rmsle_row["ci_upper"]
    assert rmsle_row["metric_value"] < 0.6   # signal is learned

    # label splitter round-trips
    target, groups = _split_regression_labels(yte, ["is_informal"])
    assert target.shape == (150,)
    assert "is_informal" in groups
