"""MSo2Sat (GeoBench V1) benchmark dataset."""

from .base import BandSpec
from .geobench_v1 import _V1Dataset


class MSo2Sat(_V1Dataset):
    """Sentinel-2 + SAR local climate zone classification (17 classes).

    Based on the So2Sat dataset with 10 Sentinel-2 and 8 SAR bands.
    """

    name = "m-so2sat"
    task = "classification"
    num_classes = 17
    multilabel = False
    rgb_bands = ["red", "green", "blue"]
    split_sizes = {"train": 19992, "val": 986, "test": 986}

    bands = [
        BandSpec("sar", "vh_real", "01 - VH.Real", mean=0.0, std=0.21),
        BandSpec("s2", "blue", "02 - Blue", wavelength_um=0.49, mean=0.13, std=0.04),
        BandSpec("sar", "vh_imag", "02 - VH.Imaginary", mean=-0.0, std=0.2),
        BandSpec("s2", "green", "03 - Green", wavelength_um=0.56, mean=0.12, std=0.05),
        BandSpec("sar", "vv_real", "03 - VV.Real", mean=0.0, std=0.52),
        BandSpec("s2", "red", "04 - Red", wavelength_um=0.665, mean=0.11, std=0.07),
        BandSpec("sar", "vv_imag", "04 - VV.Imaginary", mean=-0.0, std=0.52),
        BandSpec("sar", "vh_lee", "05 - VH.LEE Filtered", mean=0.06, std=1.81),
        BandSpec(
            "s2", "red_edge_1", "05 - Vegetation Red Edge", wavelength_um=0.705, mean=0.13, std=0.07
        ),
        BandSpec("sar", "vv_lee", "06 - VV.LEE Filtered", mean=0.34, std=4.89),
        BandSpec(
            "s2", "red_edge_2", "06 - Vegetation Red Edge", wavelength_um=0.74, mean=0.17, std=0.07
        ),
        BandSpec("sar", "vh_lee_real", "07 - VH.LEE Filtered.Real", mean=-0.0, std=0.89),
        BandSpec(
            "s2", "red_edge_3", "07 - Vegetation Red Edge", wavelength_um=0.783, mean=0.19, std=0.08
        ),
        BandSpec("s2", "nir", "08 - NIR", wavelength_um=0.842, mean=0.18, std=0.09),
        BandSpec("sar", "vv_lee_imag", "08 - VV.LEE Filtered.Imaginary", mean=-0.0, std=1.28),
        BandSpec(
            "s2",
            "red_edge_4",
            "08A - Vegetation Red Edge",
            wavelength_um=0.865,
            mean=0.21,
            std=0.09,
        ),
        BandSpec("s2", "swir_1", "11 - SWIR", wavelength_um=1.61, mean=0.18, std=0.1),
        BandSpec("s2", "swir_2", "12 - SWIR", wavelength_um=2.19, mean=0.13, std=0.09),
    ]
