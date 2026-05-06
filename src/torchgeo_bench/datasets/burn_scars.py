"""Burn Scars (GeoBench V2) benchmark dataset."""

from .base import BandSpec
from .geobench_v2 import _V2Dataset


class BurnScars(_V2Dataset):
    """Sentinel-2 burn scar segmentation (3 classes).

    Classes: background, burn, cloud.
    """

    name = "burn_scars"
    task = "segmentation"
    num_classes = 3
    multilabel = False
    rgb_bands = ["b04", "b03", "b02"]
    split_sizes = {"train": 524, "val": 160, "test": 120}

    bands = [
        BandSpec("s2", "b02", "B02", wavelength_um=0.49, mean=0.0333, std=0.0227),
        BandSpec("s2", "b03", "B03", wavelength_um=0.56, mean=0.057, std=0.0268),
        BandSpec("s2", "b04", "B04", wavelength_um=0.665, mean=0.0589, std=0.04),
        BandSpec("s2", "b8a", "B8A", wavelength_um=0.865, mean=0.2323, std=0.0779),
        BandSpec("s2", "b11", "B11", wavelength_um=1.61, mean=0.1973, std=0.0871),
        BandSpec("s2", "b12", "B12", wavelength_um=2.19, mean=0.1194, std=0.0724),
    ]
