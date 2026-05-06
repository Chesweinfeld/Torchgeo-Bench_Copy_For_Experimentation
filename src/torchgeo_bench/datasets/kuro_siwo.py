"""Kuro Siwo (GeoBench V2) benchmark dataset."""

from .base import BandSpec
from .geobench_v2 import _V2Dataset


class KuroSiwo(_V2Dataset):
    """SAR flood mapping segmentation (4 classes).

    Upstream returns multi-temporal SAR (pre / pre / post) plus a static
    DEM, so :meth:`canonicalize_sample` collapses the time axis by keeping
    the post-flood acquisition.
    """

    band_order_strategy = "by_sensor"

    name = "kuro_siwo"
    task = "segmentation"
    num_classes = 4
    multilabel = False
    rgb_bands = ["vv", "vh"]
    split_sizes = {"train": 4000, "val": 1000, "test": 2000}

    bands = [
        BandSpec("sar", "vv", "vv", mean=0.0953, std=0.0427),
        BandSpec("sar", "vh", "vh", mean=0.0264, std=0.0215),
        BandSpec("dem", "dem", "dem", mean=93.4313, std=1410.8382),
    ]

    def canonicalize_sample(self, sample: dict) -> dict:
        """Squeeze the (C, T, H, W) tensor to (C, H, W) by keeping post-flood."""
        img = sample.get("image")
        if img is not None and img.dim() == 4:
            sample["image"] = img[:, -1, ...]
        return sample
