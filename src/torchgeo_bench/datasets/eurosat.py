"""EuroSAT (torchgeo) benchmark dataset template.

Demonstrates how to wrap a non-GeoBench :class:`~torch.utils.data.Dataset`
in a :class:`~torchgeo_bench.datasets.base.BenchDataset`.  The data and
splits come from :class:`torchgeo.datasets.EuroSAT`; metadata and the
``BenchDataset`` interface live here.
"""

from collections.abc import Callable
from pathlib import Path

from torch.utils.data import Dataset
from torchgeo.datasets import EuroSAT as TGEuroSAT

from .base import BandSpec, BenchDataset


class EuroSAT(BenchDataset):
    """Sentinel-2 land-use classification (10 classes), via torchgeo.

    13 Sentinel-2 spectral bands. Identical task and class set as
    :class:`~torchgeo_bench.datasets.MEurosat` (GeoBench V1) but loads
    data through :class:`torchgeo.datasets.EuroSAT`, so file layout and
    download behaviour are managed by torchgeo.
    """

    name = "eurosat"
    task = "classification"
    num_classes = 10
    multilabel = False
    rgb_bands = ["red", "green", "blue"]
    split_sizes = {"train": 16200, "val": 5400, "test": 5400}
    supports_partitions = False

    # Band statistics mirror m-eurosat (computed from the same EuroSAT data).
    bands = [
        BandSpec(
            "s2",
            "coastal_aerosol",
            "B01",
            wavelength_um=0.443,
            mean=1356.78,
            std=246.34,
        ),
        BandSpec("s2", "blue", "B02", wavelength_um=0.49, mean=1123.14, std=334.28),
        BandSpec("s2", "green", "B03", wavelength_um=0.56, mean=1057.28, std=392.23),
        BandSpec("s2", "red", "B04", wavelength_um=0.665, mean=959.18, std=590.94),
        BandSpec("s2", "red_edge_1", "B05", wavelength_um=0.705, mean=1227.44, std=548.57),
        BandSpec("s2", "red_edge_2", "B06", wavelength_um=0.74, mean=2076.63, std=843.08),
        BandSpec("s2", "red_edge_3", "B07", wavelength_um=0.783, mean=2463.43, std=1071.73),
        BandSpec("s2", "nir", "B08", wavelength_um=0.842, mean=2390.32, std=1106.87),
        BandSpec("s2", "water_vapour", "B09", wavelength_um=0.945, mean=12.33, std=5.08),
        BandSpec("s2", "swir_cirrus", "B10", wavelength_um=1.375, mean=1861.29, std=963.03),
        BandSpec("s2", "swir_1", "B11", wavelength_um=1.61, mean=1138.94, std=742.79),
        BandSpec("s2", "swir_2", "B12", wavelength_um=2.19, mean=2699.78, std=1215.04),
        BandSpec("s2", "red_edge_4", "B8A", wavelength_um=0.865, mean=761.55, std=404.41),
    ]

    @classmethod
    def data_root(cls) -> Path:
        """Return ``Path("data/eurosat")`` (torchgeo manages its own layout below)."""
        return Path("data/eurosat")

    def get_dataset(
        self,
        split: str,
        *,
        partition: str = "default",
        bands: tuple[str, ...] | None = None,
        transform: Callable | None = None,
        normalize: str = "mean_stdev",
    ) -> Dataset:
        """Return a :class:`torchgeo.datasets.EuroSAT` for the given split."""
        del partition, normalize
        band_codes = tuple(spec.source_name for spec in self._select_band_specs(bands))
        return TGEuroSAT(
            root=str(self.data_root()),
            split=split,
            bands=band_codes,
            transforms=transform,
        )
