"""Fields of the World (GeoBench V2) benchmark dataset."""

from .base import BandSpec
from .geobench_v2 import _V2Dataset


class FieldsOfTheWorld(_V2Dataset):
    """Sentinel-2 field boundary segmentation (4 classes).

    Classes: background, field, boundary, other. Upstream returns
    ``image_a`` / ``image_b`` change-detection pairs;
    :meth:`canonicalize_sample` keeps the later acquisition (``image_b``).
    """

    name = "fotw"
    task = "segmentation"
    num_classes = 4
    multilabel = False
    rgb_bands = ["red", "green", "blue"]
    split_sizes = {"train": 4000, "val": 1000, "test": 2000}

    bands = [
        BandSpec("s2", "red", "red", mean=862.084, std=681.1667),
        BandSpec("s2", "green", "green", mean=853.3895, std=508.6401),
        BandSpec("s2", "blue", "blue", mean=592.008, std=454.0239),
        BandSpec("s2", "nir", "nir", mean=2984.3018, std=1043.6527),
    ]

    def canonicalize_sample(self, sample: dict) -> dict:
        """Pick the later acquisition (``image_b``) and surface it as ``image``."""
        if "image" not in sample and "image_b" in sample:
            sample["image"] = sample.pop("image_b")
            sample.pop("image_a", None)
        return sample
