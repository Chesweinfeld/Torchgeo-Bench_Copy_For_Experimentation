"""FLAIR2 (GeoBench V2) benchmark dataset."""

from .base import BandSpec
from .geobench_v2 import _V2Dataset


class FLAIR2(_V2Dataset):
    """Aerial land-cover segmentation (13 classes).

    French aerial imagery with RGB, NIR, and elevation bands.  The upstream
    ``GeoBenchFLAIR2`` accepts a flat ``band_order`` list and returns a
    single stacked ``image`` tensor, so this wrapper does **not** use the
    multi-modality dict shape.
    """

    name = "flair2"
    task = "segmentation"
    num_classes = 13
    multilabel = False
    rgb_bands = ["red", "green", "blue"]
    split_sizes = {"train": 4049, "val": 1022, "test": 3022}

    bands = [
        BandSpec("aerial", "red", "red", mean=110.305, std=50.71),
        BandSpec("aerial", "green", "green", mean=114.7908, std=44.3165),
        BandSpec("aerial", "blue", "blue", mean=105.6127, std=43.2948),
        BandSpec("aerial", "nir", "nir", mean=104.3409, std=39.0496),
        BandSpec("elevation", "elevation", "elevation", mean=17.6965, std=29.9427),
    ]
