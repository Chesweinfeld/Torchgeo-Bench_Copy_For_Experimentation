"""Tests for the RioFavela dataset, against a synthetic HDF5 fixture.

No network access or real download required: builds a tiny ``rio_favela.h5``
matching the layout written by ``experiments/scripts/build_rio_favela.py``.
"""

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from torchgeo_bench.datasets.loading import _make_resize_transform
from torchgeo_bench.datasets.rio_favela import RioFavela

N_BANDS = 10
CHIP_PX = 16  # smaller than the real 64px -- shape correctness doesn't need real size


@pytest.fixture
def rio_favela_h5(tmp_path: Path) -> Path:
    h5_path = tmp_path / "rio_favela.h5"
    rng = np.random.default_rng(0)
    with h5py.File(h5_path, "w") as f:
        f.attrs["band_order"] = [
            "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12",
        ]
        sizes = {"train": 4, "val": 2, "test": 2}
        for split, n in sizes.items():
            grp = f.create_group(split)
            grp.create_dataset(
                "images", data=rng.random((n, N_BANDS, CHIP_PX, CHIP_PX)).astype(np.float32)
            )
            grp.create_dataset("labels", data=rng.integers(0, 2, size=n).astype(np.int64))
            grp.create_dataset("lon", data=rng.uniform(-43.3, -43.1, size=n).astype(np.float32))
            grp.create_dataset("lat", data=rng.uniform(-23.0, -22.8, size=n).astype(np.float32))
    return h5_path


@pytest.fixture
def rio_favela_dataset(rio_favela_h5: Path, monkeypatch: pytest.MonkeyPatch) -> RioFavela:
    ds = RioFavela()
    monkeypatch.setattr(RioFavela, "data_root", classmethod(lambda cls: rio_favela_h5.parent))
    return ds


def test_all_bands_shape(rio_favela_dataset: RioFavela) -> None:
    inner = rio_favela_dataset.get_dataset("train", bands=None)
    assert len(inner) == 4
    sample = inner[0]
    assert sample["image"].shape == (N_BANDS, CHIP_PX, CHIP_PX)
    assert sample["label"].dtype == torch.long
    assert sample["label"].item() in (0, 1)


def test_rgb_band_subsetting(rio_favela_dataset: RioFavela) -> None:
    inner = rio_favela_dataset.get_dataset("train", bands=tuple(rio_favela_dataset.rgb_bands))
    sample = inner[0]
    assert sample["image"].shape == (3, CHIP_PX, CHIP_PX)


def test_split_sizes(rio_favela_dataset: RioFavela) -> None:
    assert len(rio_favela_dataset.get_dataset("train")) == 4
    assert len(rio_favela_dataset.get_dataset("val")) == 2
    assert len(rio_favela_dataset.get_dataset("test")) == 2


def test_geo_fields_opt_in(rio_favela_dataset: RioFavela) -> None:
    inner = rio_favela_dataset.get_dataset("train")
    sample = inner[0]
    assert "lon" not in sample
    assert "lat" not in sample

    rio_favela_dataset.geo_fields = ("lon", "lat")
    inner_geo = rio_favela_dataset.get_dataset("train")
    sample_geo = inner_geo[0]
    assert sample_geo["lon"].dtype == torch.float32
    assert sample_geo["lat"].dtype == torch.float32


def test_resize_transform_compatibility(rio_favela_dataset: RioFavela) -> None:
    transform = _make_resize_transform(8, "bilinear")
    inner = rio_favela_dataset.get_dataset("train", transform=transform)
    sample = inner[0]
    assert sample["image"].shape == (N_BANDS, 8, 8)
