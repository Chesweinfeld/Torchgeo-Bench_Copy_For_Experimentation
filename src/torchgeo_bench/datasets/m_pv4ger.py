"""MPv4ger (GeoBench V1) benchmark dataset."""

from .base import BandSpec
from .geobench_v1 import _V1Dataset


class MPv4ger(_V1Dataset):
    """Aerial solar panel detection (2 classes).

    Based on the PV4GER dataset with 3 aerial RGB bands.
    """

    name = "m-pv4ger"
    task = "classification"
    num_classes = 2
    multilabel = False
    rgb_bands = ["red", "green", "blue"]
    split_sizes = {"train": 11814, "val": 999, "test": 999}

    bands = [
        BandSpec("aerial", "blue", "Blue", mean=116.63, std=44.67),
        BandSpec("aerial", "green", "Green", mean=119.66, std=48.28),
        BandSpec("aerial", "red", "Red", mean=113.39, std=54.2),
    ]
