"""SpaceNet7 (GeoBench V2) benchmark dataset."""

from .base import BandSpec
from .geobench_v2 import _V2Dataset


class SpaceNet7(_V2Dataset):
    """Planet building change segmentation (3 classes).

    RGB imagery from Planet satellites.
    """

    name = "spacenet7"
    task = "segmentation"
    num_classes = 3
    multilabel = False
    rgb_bands = ["red", "green", "blue"]
    split_sizes = {"train": 3500, "val": 652, "test": 1152}

    bands = [
        BandSpec("planet", "red", "red", mean=116.9447, std=61.6558),
        BandSpec("planet", "green", "green", mean=103.5589, std=49.649),
        BandSpec("planet", "blue", "blue", mean=76.7743, std=45.8807),
    ]
