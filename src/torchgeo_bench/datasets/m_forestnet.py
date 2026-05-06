"""MForestnet (GeoBench V1) benchmark dataset."""

from .base import BandSpec
from .geobench_v1 import _V1Dataset


class MForestnet(_V1Dataset):
    """Landsat forest-change classification (12 classes).

    Based on the ForestNet dataset with 6 Landsat spectral bands.
    """

    name = "m-forestnet"
    task = "classification"
    num_classes = 12
    multilabel = False
    rgb_bands = ["red", "green", "blue"]
    split_sizes = {"train": 6464, "val": 989, "test": 993}

    bands = [
        BandSpec("landsat", "blue", "02 - Blue", wavelength_um=0.49, mean=72.85, std=15.84),
        BandSpec("landsat", "green", "03 - Green", wavelength_um=0.56, mean=83.68, std=14.79),
        BandSpec("landsat", "red", "04 - Red", wavelength_um=0.665, mean=77.58, std=16.1),
        BandSpec("landsat", "nir", "05 - NIR", mean=123.99, std=16.35),
        BandSpec("landsat", "swir_1", "06 - SWIR1", mean=91.54, std=13.79),
        BandSpec("landsat", "swir_2", "07 - SWIR2", mean=74.72, std=12.69),
    ]
