"""So2Sat (GeoBench V2) benchmark dataset."""

from .base import BandSpec
from .geobench_v2 import _V2Dataset


class So2Sat(_V2Dataset):
    """Sentinel-2 + SAR local climate zone classification (17 classes).

    GeoBench V2 version with 10 Sentinel-2 and 2 SAR bands.
    """

    band_order_strategy = "by_sensor"

    name = "so2sat"
    task = "classification"
    num_classes = 17
    multilabel = False
    rgb_bands = ["b04", "b03", "b02"]
    split_sizes = {"train": 19992, "val": 986, "test": 986}

    bands = [
        BandSpec("s2", "b02", "B02", wavelength_um=0.49, mean=0.1295, std=0.0414),
        BandSpec("s2", "b03", "B03", wavelength_um=0.56, mean=0.1172, std=0.052),
        BandSpec("s2", "b04", "B04", wavelength_um=0.665, mean=0.1138, std=0.0733),
        BandSpec("s2", "b05", "B05", wavelength_um=0.705, mean=0.1272, std=0.0694),
        BandSpec("s2", "b06", "B06", wavelength_um=0.74, mean=0.1707, std=0.0751),
        BandSpec("s2", "b07", "B07", wavelength_um=0.783, mean=0.1928, std=0.0856),
        BandSpec("s2", "b08", "B08", wavelength_um=0.842, mean=0.1855, std=0.0865),
        BandSpec("s2", "b8a", "B8A", wavelength_um=0.865, mean=0.2073, std=0.094),
        BandSpec("s2", "b11", "B11", wavelength_um=1.61, mean=0.1768, std=0.1024),
        BandSpec("s2", "b12", "B12", wavelength_um=2.19, mean=0.1285, std=0.0923),
        BandSpec("s1", "vv", "VV", mean=-0.0, std=0.5443),
        BandSpec("s1", "vh", "VH", mean=-0.0, std=0.2156),
    ]
