"""Model interface for torchgeo-bench.

Defines :class:`BenchModel`, the abstract base class that every benchmarkable
model inherits from.  The contract is split into two halves:

1. **Construction**: subclasses receive a
   :class:`list[~torchgeo_bench.datasets.base.BandSpec]` describing the input
   channels.  Per-channel mean/std/min/max statistics are available on each
   :class:`~torchgeo_bench.datasets.base.BandSpec` and are used by
   :meth:`normalize_inputs` to z-score the raw input tensor.

2. **Forward path**:

   * The public :meth:`forward_patch_features` is **sealed** — subclasses
     do not override it.  It always applies :meth:`normalize_inputs` before
     dispatching to :meth:`_forward_patch_features`, so normalization can't
     be silently forgotten.
   * Subclasses implement :meth:`_forward_patch_features` (the abstract
     hook) which receives the **already-normalized** ``(B, C, H, W)`` tensor
     and returns ``(B, K)`` embeddings.

Models whose backbones do their own normalization (e.g. OlmoEarth) override
:meth:`normalize_inputs` to identity.  Models that need a different
normalization strategy (ImageNet-style for pretrained RGB CNNs, weights-bound
``Normalize`` transforms for torchgeo wrappers, etc.) override
:meth:`normalize_inputs` with their own policy.

A small number of models don't run a pixel forward pass at all — they look
up a precomputed embedding by location/time (e.g. GeoTessera). These set
``requires_geolocation = True`` and implement :meth:`_forward_patch_features_geo`
instead of :meth:`_forward_patch_features`; see that method's docstring.
"""

from abc import ABC

import torch
import torch.nn as nn

from torchgeo_bench.datasets.base import BandSpec

from ._input_units import InputUnit
from ._normalization import NormalizationStrategy, build_normalizer


class BenchModel(nn.Module, ABC):
    """Abstract base interface for benchmarkable models.

    Args:
        bands: Ordered list of :class:`BandSpec` describing the input
            channels.  Length determines :attr:`num_channels`.
        normalization: Input-normalisation strategy name (one of
            ``bandspec_zscore`` / ``model_native`` / ``minmax`` /
            ``minmax_zscore`` / ``identity``).  Defaults to
            ``"bandspec_zscore"``.

    Subclasses may declare:

    * ``expected_input_unit`` — what scale the pretrained backbone was
      fed at training (e.g. ``s2_dn``, ``reflectance_0_1``, ``uint8``).
      Used by the ``model_native`` strategy.
    * ``pretrain_mean`` / ``pretrain_std`` — per-channel normalisation
      applied *after* unit conversion under ``model_native``.
    * ``requires_geolocation`` — set ``True`` to receive a ``geo`` dict
      (``lat``/``lon``/``year`` tensors, when the dataset supplies them)
      and dispatch to :meth:`_forward_patch_features_geo` instead of
      :meth:`_forward_patch_features`.
    """

    expected_input_unit: InputUnit | None = None
    pretrain_mean: list[float] | None = None
    pretrain_std: list[float] | None = None
    requires_geolocation: bool = False

    def __init__(
        self,
        bands: list[BandSpec],
        normalization: NormalizationStrategy | str = NormalizationStrategy.BANDSPEC_ZSCORE,
        **_: object,
    ) -> None:
        super().__init__()
        if not bands:
            raise ValueError("BenchModel requires a non-empty list of BandSpec.")
        self.bands: list[BandSpec] = list(bands)
        self.num_channels: int = len(self.bands)
        self.normalization = NormalizationStrategy(normalization)
        self._normalizer = build_normalizer(
            self.normalization,
            bands=self.bands,
            expected_input_unit=self.expected_input_unit,
            pretrain_mean=self.pretrain_mean,
            pretrain_std=self.pretrain_std,
        )

    def normalize_inputs(self, images: torch.Tensor) -> torch.Tensor:
        """Apply the configured normalisation strategy."""
        return self._normalizer(images)

    def _forward_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        """Subclass hook — receives normalized ``(B, C, H, W)``, returns ``(B, K)``.

        Implementations should call only the backbone; the public
        :meth:`forward_patch_features` has already applied
        :meth:`normalize_inputs`. Required for every model except those with
        :attr:`requires_geolocation` set, which implement
        :meth:`_forward_patch_features_geo` instead.

        Args:
            images: Normalized input tensor of shape ``(B, C, H, W)``.

        Returns:
            Embeddings tensor of shape ``(B, K)``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _forward_patch_features "
            "(or set requires_geolocation=True and implement "
            "_forward_patch_features_geo)."
        )

    def _forward_patch_features_geo(
        self, images: torch.Tensor, geo: dict[str, torch.Tensor] | None
    ) -> torch.Tensor:
        """Subclass hook for :attr:`requires_geolocation` models.

        Receives the same normalized ``(B, C, H, W)`` tensor as
        :meth:`_forward_patch_features` (pixel content may be ignored
        entirely) plus ``geo``: a dict of per-sample tensors drawn from
        whatever the dataset's :attr:`~torchgeo_bench.datasets.base.BenchDataset.geo_fields`
        supplies (a subset of ``lat``/``lon``/``year``), or ``None`` if the
        dataset supplies none. Must return ``(B, K)`` embeddings.
        """
        raise NotImplementedError(
            f"{type(self).__name__} sets requires_geolocation=True but does not "
            "implement _forward_patch_features_geo."
        )

    def forward_patch_features(
        self, images: torch.Tensor, geo: dict[str, torch.Tensor] | None = None
    ) -> torch.Tensor:
        """Return a batch of vector embeddings ``(B, K)`` from raw inputs.

        Sealed: applies :meth:`normalize_inputs` then dispatches to
        :meth:`_forward_patch_features`, or to
        :meth:`_forward_patch_features_geo` (passing ``geo`` through) when
        :attr:`requires_geolocation` is set.  Override
        :meth:`normalize_inputs` to change the normalization policy.
        """
        normalized = self.normalize_inputs(images)
        if self.requires_geolocation:
            return self._forward_patch_features_geo(normalized, geo)
        return self._forward_patch_features(normalized)

    def forward(
        self, images: torch.Tensor, geo: dict[str, torch.Tensor] | None = None
    ) -> torch.Tensor:
        """Alias for :meth:`forward_patch_features`."""
        return self.forward_patch_features(images, geo=geo)


_GEO_KEYS = ("lat", "lon", "year")


def extract_geo_from_batch(batch: dict) -> dict[str, torch.Tensor] | None:
    """Pull ``lat``/``lon``/``year`` tensors out of a dataloader batch dict.

    Returns ``None`` when the dataset didn't supply any of them (the common
    case), so callers can pass the result straight through as ``geo=``
    without an extra branch.
    """
    geo = {k: batch[k].float() for k in _GEO_KEYS if k in batch}
    return geo or None
