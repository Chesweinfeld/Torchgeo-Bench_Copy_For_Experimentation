"""Rio de Janeiro favela vs. non-favela binary classification.

Sentinel-2 L2A chips (64x64 px @ 10m) tiled over a buffered footprint of Rio's
official 2022 favela boundaries (IPP/PCRJ), labeled by vector-coverage overlap
(>=15% chip area) and split via a spatial checkerboard to avoid leakage
between adjacent chips. Built by :file:`experiments/scripts/build_rio_favela.py`
(a one-time, non-package extraction script) and packaged into a single HDF5
file, auto-downloaded from HuggingFace Hub the same way as every other
auto-downloading dataset/model in this package.
"""

from collections.abc import Callable
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .base import BandSpec, BenchDataset

_HF_REPO_ID = "Chesapeakeiw/rio-favela-s2-classification"
_HF_FILENAME = "rio_favela.h5"


class _RioFavelaDataset(Dataset):
    """Reads one split's chips from ``rio_favela.h5``.

    Opens the file per ``__getitem__`` call (not held open across calls) so
    instances are safe to share across DataLoader worker processes -- same
    pattern as :class:`~torchgeo_bench.datasets.geobench_v1.GeoBenchv1`.
    """

    def __init__(
        self,
        h5_path: Path,
        split: str,
        band_indices: list[int],
        transform: Callable | None = None,
        geo_fields: tuple[str, ...] = (),
    ) -> None:
        self.h5_path = h5_path
        self.split = split
        self.band_indices = band_indices
        self.transform = transform
        self.geo_fields = geo_fields
        with h5py.File(h5_path, "r") as f:
            self._len = f[split]["labels"].shape[0]

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> dict:
        with h5py.File(self.h5_path, "r") as f:
            grp = f[self.split]
            # h5py fancy-indexing requires increasing order (e.g. rgb_bands is
            # [2, 1, 0]) -- read all bands, then reorder/subset in numpy.
            image = grp["images"][idx][self.band_indices].astype(np.float32)
            label = int(grp["labels"][idx])
            lon = float(grp["lon"][idx])
            lat = float(grp["lat"][idx])

        sample: dict = {
            "image": torch.from_numpy(image),
            "label": torch.tensor(label, dtype=torch.long),
        }
        if "lon" in self.geo_fields:
            sample["lon"] = torch.tensor(lon, dtype=torch.float32)
        if "lat" in self.geo_fields:
            sample["lat"] = torch.tensor(lat, dtype=torch.float32)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


class RioFavela(BenchDataset):
    """Binary favela / non-favela classification over Rio de Janeiro.

    10 Sentinel-2 L2A bands, 64x64 px @ 10m chips. Positive class (favela) is
    assigned when a chip overlaps >=15% with the official 2022 favela
    boundary polygons; split is a spatial checkerboard (not random) to
    prevent adjacent chips leaking across train/val/test.
    """

    name = "rio_favela"
    task = "classification"
    num_classes = 2
    multilabel = False
    rgb_bands = ["red", "green", "blue"]
    supports_partitions = False
    # 2,993 chips total from build_rio_favela.py's dry-season 2022 composite
    # (8.8% / 9.2% / 7.1% favela-positive in train/val/test respectively).
    split_sizes = {"train": 1858, "val": 587, "test": 548}

    # Real train-split statistics logged by build_rio_favela.py.
    # fmt: off
    bands = [
        BandSpec("s2", "blue", "B02", mean=1508.05, std=416.15, min=957.50, max=12832.00, wavelength_um=0.49),
        BandSpec("s2", "green", "B03", mean=1665.59, std=444.57, min=996.00, max=13960.00, wavelength_um=0.56),
        BandSpec("s2", "red", "B04", mean=1653.56, std=539.49, min=990.00, max=14536.00, wavelength_um=0.665),
        BandSpec("s2", "red_edge_1", "B05", mean=1949.81, std=493.62, min=1012.00, max=9629.00, wavelength_um=0.705),
        BandSpec("s2", "red_edge_2", "B06", mean=2668.78, std=718.49, min=972.00, max=10162.00, wavelength_um=0.74),
        BandSpec("s2", "red_edge_3", "B07", mean=2948.71, std=892.32, min=961.00, max=10967.00, wavelength_um=0.783),
        BandSpec("s2", "nir", "B08", mean=3010.28, std=988.64, min=976.50, max=14488.00, wavelength_um=0.842),
        BandSpec("s2", "red_edge_4", "B8A", mean=3114.44, std=977.08, min=987.00, max=12218.00, wavelength_um=0.865),
        BandSpec("s2", "swir_1", "B11", mean=2721.24, std=735.10, min=1015.00, max=15274.00, wavelength_um=1.61),
        BandSpec("s2", "swir_2", "B12", mean=2247.71, std=740.16, min=1003.00, max=16135.50, wavelength_um=2.19),
    ]
    # fmt: on

    @classmethod
    def data_root(cls) -> Path:
        """Return ``Path("data/rio_favela")``."""
        return Path("data/rio_favela")

    def get_dataset(
        self,
        split: str,
        *,
        partition: str = "default",
        bands: tuple[str, ...] | None = None,
        transform: Callable | None = None,
    ) -> Dataset:
        """Return a :class:`_RioFavelaDataset` for the given split.

        Uses a local ``data/rio_favela/rio_favela.h5`` if present, otherwise
        auto-downloads it from the HuggingFace Hub mirror.
        """
        del partition
        local_path = self.data_root() / "rio_favela.h5"
        if local_path.exists():
            h5_path = local_path
        else:
            from huggingface_hub import hf_hub_download

            h5_path = Path(
                hf_hub_download(repo_id=_HF_REPO_ID, filename=_HF_FILENAME, repo_type="dataset")
            )

        specs = self.select_band_specs(bands)
        all_source_names = [b.source_name for b in self.bands]
        band_indices = [all_source_names.index(spec.source_name) for spec in specs]

        return _RioFavelaDataset(
            h5_path,
            split,
            band_indices,
            transform=transform,
            geo_fields=self.geo_fields,
        )
