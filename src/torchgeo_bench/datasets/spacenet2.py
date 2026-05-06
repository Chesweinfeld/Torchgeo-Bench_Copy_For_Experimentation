"""SpaceNet2 (GeoBench V2) benchmark dataset."""

from .base import BandSpec
from .geobench_v2 import _V2Dataset


class SpaceNet2(_V2Dataset):
    """WorldView building footprint segmentation (3 classes).

    8 multispectral + 1 panchromatic band from WorldView satellite.
    """

    band_order_strategy = "by_sensor"

    name = "spacenet2"
    task = "segmentation"
    num_classes = 3
    multilabel = False
    rgb_bands = ["red", "green", "blue"]
    split_sizes = {"train": 5186, "val": 1461, "test": 2961}

    bands = [
        BandSpec("worldview", "coastal", "coastal", mean=298.7281, std=106.9792),
        BandSpec("worldview", "blue", "blue", mean=358.0099, std=148.1868),
        BandSpec("worldview", "green", "green", mean=464.5104, std=224.4095),
        BandSpec("worldview", "yellow", "yellow", mean=419.9473, std=225.7901),
        BandSpec("worldview", "red", "red", mean=333.6004, std=194.0233),
        BandSpec("worldview", "red_edge", "red_edge", mean=408.6689, std=208.4557),
        BandSpec("worldview", "nir1", "nir1", mean=475.0842, std=234.7585),
        BandSpec("worldview", "nir2", "nir2", mean=362.3487, std=193.2321),
        BandSpec("pan", "pan", "pan", mean=468.574, std=260.8954),
    ]
