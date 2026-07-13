"""Regression tests for silent-failure audit fixes."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from torchgeo_bench.datasets.base import BandSpec
from torchgeo_bench.datasets.caffe import CaFFe
from torchgeo_bench.datasets.loading import _make_resize_transform
from torchgeo_bench.models import InputUnit
from torchgeo_bench.models._input_units import detect_input_unit
from torchgeo_bench.utils import extract_features


class _HeadPoolModel(torch.nn.Module):
    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        batch = images.shape[0]
        return {
            "head.global_pool": torch.arange(batch * 4, dtype=torch.float32).reshape(batch, 1, 4)
        }


def test_extract_features_preserves_batch_dimension_for_head_global_pool() -> None:
    dataset = [
        {"image": torch.zeros(3, 2, 2), "label": torch.tensor(0)},
        {"image": torch.zeros(3, 2, 2), "label": torch.tensor(1)},
    ]
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    features, labels = extract_features(_HeadPoolModel(), loader, "cpu", verbose=False)
    assert features.shape == (2, 4)
    np.testing.assert_array_equal(labels, np.array([0, 1]))


def test_invalid_interpolation_raises_instead_of_falling_back() -> None:
    with pytest.raises(ValueError, match="interpolation must be one of"):
        _make_resize_transform(224, "bilnear")


def test_mixed_scale_unit_detection_raises() -> None:
    bands = [
        BandSpec("s2", "red", "red", mean=0.1, std=0.1, min=0.0, max=1.0, wavelength_um=0.665),
        BandSpec("sar", "vv", "VV", mean=20.0, std=4.0, min=0.0, max=255.0),
    ]
    with pytest.raises(ValueError, match="mixed-scale bands"):
        detect_input_unit(bands)


def test_unit_detection_keeps_low_magnitude_bands_in_raw_sensor_stack() -> None:
    bands = [
        BandSpec("s2", "red", "red", mean=950.0, std=500.0, min=0.0, max=28000.0),
        BandSpec("s2", "cirrus", "B10", mean=12.0, std=5.0, min=0.0, max=90.0),
    ]
    assert detect_input_unit(bands) == InputUnit.S2_DN


def test_caffe_rgb_mode_uses_single_declared_gray_channel() -> None:
    assert CaFFe.rgb_bands == ["gray"]


def test_resize_transform_handles_standard_3d_image_and_2d_mask() -> None:
    """Backward-compat: the common (C,H,W) image / (H,W) mask case is unchanged."""
    transform = _make_resize_transform(8, "bilinear")
    sample = {"image": torch.rand(3, 16, 16), "mask": torch.randint(0, 5, (16, 16))}
    out = transform(sample)
    assert out["image"].shape == (3, 8, 8)
    assert out["mask"].shape == (8, 8)
    assert out["mask"].dtype == torch.long


def test_resize_transform_handles_extra_leading_image_dim() -> None:
    """dynamic_earthnet's Planet stream is (T, C, H, W), not (C, H, W) -- the
    resize must preserve every leading dim, only touching the spatial ones.
    Previously crashed: F.interpolate saw a 4D tensor after an extra
    unsqueeze and expected 3D spatial input instead of 2D.
    """
    transform = _make_resize_transform(8, "bilinear")
    sample = {"image": torch.rand(1, 4, 16, 16)}
    out = transform(sample)
    assert out["image"].shape == (1, 4, 8, 8)


def test_resize_transform_handles_extra_leading_mask_dim() -> None:
    """dynamic_earthnet's mask still carries rasterio's leading band dim
    (1, H, W) at transform time, not yet squeezed to plain (H, W). Previously
    crashed the same way as the image case above.
    """
    transform = _make_resize_transform(8, "bilinear")
    sample = {"mask": torch.randint(0, 5, (1, 16, 16))}
    out = transform(sample)
    assert out["mask"].shape == (1, 8, 8)


def test_resize_transform_can_be_called_with_mask_only() -> None:
    """Required for by_sensor datasets: the per-modality resize loop in
    geobench_v2.py's ``chained`` wrapper calls the transform once per
    image_* key and once more for "mask" alone -- "image" must be optional.
    """
    transform = _make_resize_transform(8, "bilinear")
    out = transform({"mask": torch.randint(0, 5, (16, 16))})
    assert "image" not in out
    assert out["mask"].shape == (8, 8)
