"""TreeSatAI (GeoBench V2) benchmark dataset."""

from .base import BandSpec
from .geobench_v2 import _V2Dataset


class TreeSatAI(_V2Dataset):
    """Aerial + Sentinel-2 + SAR tree species classification (13 classes).

    Multi-sensor dataset with aerial RGB+NIR, 12 Sentinel-2 bands, and 3 SAR bands.
    """

    band_order_strategy = "by_sensor"

    name = "treesatai"
    task = "classification"
    num_classes = 13
    multilabel = False
    rgb_bands = ["red", "green", "blue"]
    split_sizes = {"train": 4000, "val": 1000, "test": 2000}

    bands = [
        BandSpec("aerial", "red", "red", mean=79.3079, std=33.3639),
        BandSpec("aerial", "green", "green", mean=92.1351, std=33.5291),
        BandSpec("aerial", "blue", "blue", mean=85.3632, std=27.9319),
        BandSpec("aerial", "nir", "nir", mean=154.2898, std=49.0291),
        BandSpec("s2", "b02", "B02", wavelength_um=0.49, mean=245.3107, std=117.7349),
        BandSpec("s2", "b03", "B03", wavelength_um=0.56, mean=387.6357, std=130.0996),
        BandSpec("s2", "b04", "B04", wavelength_um=0.665, mean=248.4667, std=129.6638),
        BandSpec("s2", "b08", "B08", wavelength_um=0.842, mean=2825.936, std=756.8176),
        BandSpec("s2", "b05", "B05", wavelength_um=0.705, mean=625.9301, std=191.3524),
        BandSpec("s2", "b06", "B06", wavelength_um=0.74, mean=2118.8374, std=517.2822),
        BandSpec("s2", "b07", "B07", wavelength_um=0.783, mean=2709.3789, std=691.1488),
        BandSpec("s2", "b8a", "B8A", wavelength_um=0.865, mean=2982.2087, std=754.942),
        BandSpec("s2", "b11", "B11", wavelength_um=1.61, mean=1316.7186, std=411.3391),
        BandSpec("s2", "b12", "B12", wavelength_um=2.19, mean=594.2034, std=234.4886),
        BandSpec("s2", "b01", "B01", wavelength_um=0.443, mean=265.807, std=125.9928),
        BandSpec("s2", "b09", "B09", wavelength_um=0.945, mean=2962.1824, std=674.1692),
        BandSpec("s1", "vv", "vv", mean=-6.3649, std=3.5287),
        BandSpec("s1", "vh", "vh", mean=-12.5086, std=3.2121),
        BandSpec("s1", "vv_vh", "vv/vh", mean=0.4892, std=0.2583),
    ]
