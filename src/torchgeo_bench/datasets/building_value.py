"""Per-building value estimation — a regression benchmark dataset.

Motivation
----------
Disaster-risk analytics need an *exposure* layer (risk = exposure x hazard x
vulnerability). The scarce, high-uncertainty component is building value in
regions where valuation registries are weak. This dataset frames "predict the
structure-only replacement value of a building from an image patch" as a
**continuous regression** task, so frozen geospatial foundation models can be
benchmarked on it with the same linear-probe methodology used for GeoBench
classification — the resulting features -> value probe lands in the standard
results CSV alongside every other backbone.

Design choices tied to the pilot brief
---------------------------------------
* Target is **structure value**, with land value removed. Real assessor sources
  report land and improvement value separately; the value join subtracts land.
* Value is **continuous**, scored with RMSLE / log-bias / within-factor (see
  :mod:`torchgeo_bench.regression`), not binned into classes.
* The ``label`` carries a second column, an **informal-settlement flag**, so the
  evaluator can report a robustness slice for informal vs formal structures.
* A ``building-value-transfer`` variant loads a *different* city so the runner
  can measure the generalization gap (train city A, test held-out A vs city B).

Data status
-----------
The loader ships a **synthetic, offline generator** so the benchmark runs in CI
and on a laptop with no downloads. The generator produces small image patches
whose pixel statistics are correlated with the value target, so a real backbone
can learn a non-trivial probe. Replace :meth:`_build_split` with real data by
wiring:

* footprints — Overture Maps (https://overturemaps.org/) or Microsoft Global ML
  Building Footprints (used to center image patches and guarantee "a building is
  a building");
* imagery — your chosen high-resolution tiles (this is the main $ cost at
  scale, tracked as a cost metric in the wider eval);
* value ground truth — a benchmark jurisdiction: US county assessor records
  (separate land/improvement value) or the GEM Global Exposure Model
  replacement-cost layer for Asian coverage.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .base import BandSpec, BenchDataset


# Per-city generative parameters. Two contrasting cities so the transfer
# variant is meaningful: different price level, informal share, and height.
_CITY_PARAMS = {
    "metro_a": dict(value_scale=1.0, informal_share=0.15, height_mean=12.0,
                    land_frac=0.45, brightness=0.55),
    "metro_b": dict(value_scale=0.55, informal_share=0.40, height_mean=6.0,
                    land_frac=0.60, brightness=0.40),
}


class _RealBuildingPatches(Dataset):
    """Patches prepared on disk by ``scripts/prepare_building_value.py``.

    Reads ``{split}/patches.npy`` (N,C,H,W float32 DN) and ``{split}/labels.npy``
    (N,2 -> [structure_value_usd, is_informal]). Emits the exact same item shape
    as :class:`_SyntheticBuildingPatches`, so the evaluator, metrics, and slices
    are unchanged whether the data is synthetic or real.
    """

    def __init__(self, root: Path, split: str, channels: int,
                 transform: Callable | None):
        d = Path(root) / split
        patches = np.load(d / "patches.npy")          # (N, C, H, W)
        labels = np.load(d / "labels.npy")            # (N, 2)
        if patches.shape[1] < channels:
            raise ValueError(
                f"{d}/patches.npy has {patches.shape[1]} bands, need {channels}. "
                "Re-run prep with the requested bands."
            )
        self._patches = patches[:, :channels].astype(np.float32)
        self._labels = labels.astype(np.float32)
        self.transform = transform

    def __len__(self) -> int:
        return len(self._patches)

    def __getitem__(self, i: int) -> dict:
        sample = {
            "image": torch.from_numpy(self._patches[i]),
            "label": torch.from_numpy(self._labels[i]),  # [value, is_informal]
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


class _SyntheticBuildingPatches(Dataset):
    """In-memory synthetic patches: image correlated with structure value.

    Each item is ``{"image": (C,H,W) float32, "label": (1+G,) float32}`` where
    ``label[0]`` is the structure value (land removed) and ``label[1]`` is the
    informal flag. Pixel intensity encodes log-area/height/type so a frozen
    backbone's pooled features can regress the value; noise keeps it non-trivial.
    """

    def __init__(self, city: str, n: int, patch: int, channels: int,
                 seed: int, transform: Callable | None):
        p = _CITY_PARAMS[city]
        rng = np.random.default_rng(seed)
        self.transform = transform
        self.patch = patch
        self.channels = channels

        area = rng.lognormal(mean=4.4, sigma=0.6, size=n)
        height = np.clip(rng.normal(p["height_mean"], 4.0, n), 2.0, None)
        is_informal = rng.random(n) < p["informal_share"]
        type_mult = np.where(is_informal, 0.25, rng.choice([1.0, 1.8, 1.3, 1.5, 1.4], size=n,
                                                           p=[0.6, 0.15, 0.1, 0.05, 0.10]))
        noise = rng.lognormal(mean=0.0, sigma=0.35, size=n)
        structure = (p["value_scale"] * type_mult * (area ** 0.9)
                     * (1.0 + 0.05 * height) * 1000.0 * noise)

        # Per-channel patch intensity is a *fixed* deterministic function of the
        # value drivers (standardized by constants, not per-split statistics).
        # This is critical: the intensity->value mapping must be identical
        # across train/val/test (and across cities up to their params) so a
        # probe fit on one split transfers to another. Only the sampled
        # buildings differ per split. Each channel emphasises a different driver
        # (red~area, green~height, blue~type, nir~mix) so a frozen backbone that
        # pools the patch gets a genuinely multi-dimensional, learnable signal.
        z_area = (np.log(area) - 4.4) / 0.6
        z_height = (height - p["height_mean"]) / 4.0            # city-mean centered
        z_type = (type_mult - 1.0) / 0.5
        drivers = np.stack([z_area, z_height, z_type], axis=1)  # (n, 3)
        # fixed channel x driver mixing matrix (constant, shared across splits)
        chan_mix = np.array(
            [[0.20, 0.04, 0.06],   # red   ~ area
             [0.05, 0.18, 0.05],   # green ~ height
             [0.05, 0.04, 0.20],   # blue  ~ type
             [0.12, 0.10, 0.12]],  # nir   ~ mix
            dtype=np.float64,
        )[:channels]
        self._intensity = p["brightness"] + drivers @ chan_mix.T   # (n, channels)
        self._informal = is_informal
        self._structure = structure
        self._seed = seed

    def __len__(self) -> int:
        return len(self._structure)

    def __getitem__(self, i: int) -> dict:
        # Deterministic per-item texture around the value-driven per-channel mean.
        rng = np.random.default_rng(self._seed * 1_000_003 + i)
        base = self._intensity[i].reshape(self.channels, 1, 1)  # (C,1,1)
        img = np.clip(
            base + 0.05 * rng.standard_normal((self.channels, self.patch, self.patch)),
            0.0, 1.0,
        ).astype(np.float32)
        # scale to a sensor-like DN range so model normalizers behave
        img = img * 3000.0
        sample = {
            "image": torch.from_numpy(img),
            "label": torch.tensor([float(self._structure[i]),
                                   float(self._informal[i])], dtype=torch.float32),
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


class BuildingValue(BenchDataset):
    """Structure-value regression from aerial/optical patches (city: metro_a)."""

    name = "building-value"
    task = "regression"
    num_classes = 1                    # one continuous target
    multilabel = False
    rgb_bands = ["red", "green", "blue"]
    split_sizes = {"train": 3000, "val": 1000, "test": 1000}
    supports_partitions = False
    regression_target = "structure_value_usd"
    regression_group_names = ["is_informal"]

    #: which synthetic city this dataset draws from (overridden by transfer variant)
    city = "metro_a"
    #: patch geometry for the synthetic generator
    patch_size = 32

    # RGB(+NIR) aerial bands. Stats are placeholder DN ranges for the synthetic
    # generator; replace with real per-band stats when wiring true imagery.
    # fmt: off
    bands = [
        BandSpec("aerial", "red", "R", mean=1500.0, std=600.0, min=0, max=3000, wavelength_um=0.66),
        BandSpec("aerial", "green", "G", mean=1500.0, std=600.0, min=0, max=3000, wavelength_um=0.56),
        BandSpec("aerial", "blue", "B", mean=1500.0, std=600.0, min=0, max=3000, wavelength_um=0.48),
        BandSpec("aerial", "nir", "N", mean=1500.0, std=600.0, min=0, max=3000, wavelength_um=0.84),
    ]
    # fmt: on

    @classmethod
    def data_root(cls) -> Path:
        """Directory for real data drops (unused by the synthetic generator)."""
        return Path("data/building_value")

    def _split_seed(self, split: str) -> int:
        base = {"train": 0, "val": 1, "test": 2}[split]
        # distinct stream per city so transfer city B never overlaps city A
        city_offset = 0 if self.city == "metro_a" else 100
        return base + city_offset

    def get_dataset(
        self,
        split: str,
        *,
        partition: str = "default",
        bands: tuple[str, ...] | None = None,
        transform: Callable | None = None,
    ) -> Dataset:
        """Return a split.

        TODO(real data): replace the synthetic generator with a loader that,
        for the requested spatial split, reads image patches centered on
        Overture/MS building footprints and joins the structure-value target
        (assessor improvement value, or GEM replacement cost). Keep the
        ``label = [structure_value, is_informal]`` convention so metrics and
        slices are unchanged.
        """
        del partition
        n = self.split_sizes[split]
        n_channels = len(self.select_band_specs(bands))
        return _SyntheticBuildingPatches(
            city=self.city, n=n, patch=self.patch_size,
            channels=n_channels, seed=self._split_seed(split), transform=transform,
        )


class BuildingValueTransfer(BuildingValue):
    """Out-of-distribution variant: a *different* city (metro_b).

    Train a probe on ``building-value`` (metro_a) and evaluate it here to
    measure the cross-city generalization gap. Same bands/target/convention;
    only the underlying city changes.
    """

    name = "building-value-transfer"
    city = "metro_b"
    split_sizes = {"train": 3000, "val": 1000, "test": 1000}


class BuildingValueReal(BuildingValue):
    """Real structure-value regression from NAIP patches over Larimer County, CO.

    Prepare the data once with::

        python scripts/prepare_building_value.py \
            --parcels data/raw/larimer_parcels.shp \
            --values  data/raw/larimer_values.csv \
            --out     data/building_value/larimer_co

    then select it with ``dataset=building-value-real``. Everything downstream
    (ridge probe, RMSLE / within-factor / exposure-weighted metrics, informal
    slice) is identical to the synthetic benchmark — only the source of the
    patches and the value target changes.

    If the prepared directory is missing, this raises a clear error rather than
    silently falling back, so a "real" run never quietly scores synthetic data.
    """

    name = "building-value-real"
    #: split_sizes is informational only here; real counts come from the files.
    split_sizes = {"train": 0, "val": 0, "test": 0}

    @classmethod
    def data_root(cls) -> Path:
        return Path("data/building_value/larimer_co")

    def get_dataset(
        self,
        split: str,
        *,
        partition: str = "default",
        bands: tuple[str, ...] | None = None,
        transform: Callable | None = None,
    ) -> Dataset:
        del partition
        root = self.data_root()
        if not (root / split / "patches.npy").exists():
            raise FileNotFoundError(
                f"No prepared data at {root/split}. Run "
                "scripts/prepare_building_value.py first "
                "(or use dataset=building-value for the synthetic benchmark)."
            )
        n_channels = len(self.select_band_specs(bands))
        return _RealBuildingPatches(root, split, n_channels, transform)
