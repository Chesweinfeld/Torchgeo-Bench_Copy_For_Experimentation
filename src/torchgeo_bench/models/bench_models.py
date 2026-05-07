"""Lightweight benchmark model wrappers (RCF and ImageStats)."""

import torch
from torch.utils.data import Dataset
from torchgeo.datasets import NonGeoDataset

from torchgeo_bench.datasets.base import BandSpec

from .interface import BenchModel
from .models import RCF


class _NormalizingDatasetView(Dataset):
    """Wraps a benchmark dataset so ``__getitem__`` returns z-scored images.

    Used by :class:`RCFBench` empirical mode so the patches sampled to seed
    the ZCA-whitened filter bank live in the same distribution as the inputs
    that :meth:`BenchModel.normalize_inputs` will produce at inference time.
    """

    def __init__(self, base: Dataset, mean: torch.Tensor, std: torch.Tensor) -> None:
        self._base = base
        # Per-channel (C, 1, 1) tensors for sample-level normalization.
        self._mean = mean.detach().clone().view(-1, 1, 1).cpu().float()
        self._std = std.detach().clone().clamp_min(1e-8).view(-1, 1, 1).cpu().float()

    def __len__(self) -> int:
        return len(self._base)  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> dict:
        sample = self._base[idx]
        img = sample["image"].float()
        sample = dict(sample)
        sample["image"] = (img - self._mean) / self._std
        return sample


class RCFBench(BenchModel):
    """Wrapper for the existing :class:`RCF` implementation.

    Modes:

    - ``mode="gaussian"``: filters are drawn from a Gaussian; default
      :meth:`BenchModel.normalize_inputs` (per-channel z-score) is applied
      to inference inputs.
    - ``mode="empirical"``: filters are sampled from ``dataset``.  To keep
      the filter bank and inference inputs in the same distribution, the
      passed dataset is wrapped so its samples are pre-normalized with the
      same per-channel z-score this :class:`RCFBench` will use at inference.
    """

    def __init__(
        self,
        bands: list[BandSpec],
        features: int = 512,
        kernel_size: int = 3,
        mode: str = "gaussian",
        stats_mode: str = "mean",
        seed: int | None = None,
        dataset: NonGeoDataset | None = None,
        **_kwargs,
    ) -> None:
        super().__init__(bands=bands)
        if mode == "empirical" and dataset is not None:
            dataset = _NormalizingDatasetView(dataset, self.input_mean, self.input_std)
        self.rcf = RCF(
            in_channels=self.num_channels,
            features=features,
            kernel_size=kernel_size,
            mode=mode,
            stats_mode=stats_mode,
            seed=seed,
            dataset=dataset,
        )

    def _forward_patch_features(
        self,
        images: torch.Tensor,
        bboxes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return RCF embeddings for already-normalized images."""
        del bboxes
        return self.rcf(images)


class ImageStatsBench(BenchModel):
    """BenchModel that returns per-image statistics (mean, std, min, max).

    Returns *raw* sensor statistics: :meth:`normalize_inputs` is overridden
    to identity so the per-band magnitudes are preserved.  Downstream KNN
    distances and the LogisticRegression sweep see large, unscaled
    per-channel values; widen ``eval.c_range`` if the default sweep
    saturates.
    """

    def normalize_inputs(self, images: torch.Tensor) -> torch.Tensor:
        """Identity — this model intentionally exposes raw sensor statistics."""
        return images

    def _forward_patch_features(
        self,
        images: torch.Tensor,
        bboxes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return per-channel image statistics (mean, std, max, min)."""
        del bboxes
        feats = torch.cat(
            [
                torch.mean(images, dim=(2, 3)),
                torch.std(images, dim=(2, 3)),
                torch.amax(images, dim=(2, 3)),
                torch.amin(images, dim=(2, 3)),
            ],
            dim=1,
        )
        return feats
