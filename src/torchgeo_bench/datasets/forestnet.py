"""Forestnet (GeoBench V2) benchmark dataset."""

from .base import BandSpec
from .geobench_v2 import _V2Dataset


class Forestnet(_V2Dataset):
    """Sentinel-2 forest-change classification (12 classes).

    GeoBench V2 version with 6 Sentinel-2 spectral bands.
    """

    name = "forestnet"
    task = "classification"
    num_classes = 12
    multilabel = False
    rgb_bands = ["b04", "b03", "b02"]
    split_sizes = {"train": 6464, "val": 989, "test": 993}

    bands = [
        BandSpec("s2", "b02", "B02", wavelength_um=0.49, mean=72.3759, std=16.2839),
        BandSpec("s2", "b03", "B03", wavelength_um=0.56, mean=83.1816, std=15.3587),
        BandSpec("s2", "b04", "B04", wavelength_um=0.665, mean=77.0861, std=16.6665),
        BandSpec("s2", "b8a", "B8A", wavelength_um=0.865, mean=123.5425, std=16.9485),
        BandSpec("s2", "b11", "B11", wavelength_um=1.61, mean=91.0483, std=14.2801),
        BandSpec("s2", "b12", "B12", wavelength_um=2.19, mean=74.3097, std=13.2854),
    ]
