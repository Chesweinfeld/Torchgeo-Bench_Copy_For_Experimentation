"""Model interface for torchgeo-bench.

This module defines a lightweight base class that geospatial / foundation models
can inherit from (or emulate) in order to be benchmarked with the
``torchgeo_bench.py`` script.

Contract (forward_features):
  Inputs:
    images: torch.Tensor shape (B, C, H, W) with float32 values in [0, 1] (or
            already normalized if the dataset transform performs normalization).
    bboxes: Optional torch.Tensor shape (B, 4) with (minx, miny, maxx, maxy)
            coordinates in EPSG:4326. For pure image models this can be None.
  Output:
    embeddings: torch.Tensor shape (B, K) where K is the embedding dimension.

Optionally a model may implement ``forward_pixel_features`` returning a per-pixel
embedding map of shape (B, K, H, W) or (B, H, W, K). This is future-facing for
semantic segmentation style benchmarks and is not yet consumed by the current
benchmark script.

To integrate an existing timm / torchgeo model you can create a thin wrapper
class implementing ``forward_features`` (and delegating any internal feature
extraction utilities) while leaving the original model untouched.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BenchModel(nn.Module, ABC):
    """Abstract base interface for benchmarkable models.

    Design requirements:
      * ``__init__`` MUST accept ``num_channels: int`` (positional or keyword).
      * ``forward_features`` MUST return a 2D tensor (B, K).
      * ``forward`` defaults to calling ``forward_features``.
      * Implementations may ignore ``bboxes`` if not spatially aware.
    """

    def __init__(self, num_channels: int, *_, **__):  # type: ignore[no-untyped-def]
        super().__init__()
        self.num_channels = num_channels

    @abstractmethod
    def forward_features(
        self,
        images: torch.Tensor,
        bboxes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return a batch of vector embeddings (B, K)."""
        raise NotImplementedError

    # Optional extension point (not yet used by benchmark)
    def forward_pixel_features(  # pragma: no cover - optional
        self,
        images: torch.Tensor,
        bboxes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError("Per-pixel features not implemented.")

    def forward(
        self,
        images: torch.Tensor,
        bboxes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.forward_features(images, bboxes)


class IdentityBenchWrapper(BenchModel):
    """Wrap any existing module whose forward already yields embeddings.

    Automatically handles a variety of common output shapes (dict, 3D tokens,
    4D feature maps) and pools them to (B, K).
    """

    def __init__(self, module: nn.Module, num_channels: int):  # type: ignore[no-untyped-def]
        super().__init__(num_channels=num_channels)
        self.module = module

    def forward_features(
        self,
        images: torch.Tensor,
        bboxes: torch.Tensor | None = None,
    ) -> torch.Tensor:  # noqa: D401
        feats = self.module(images)
        if isinstance(feats, dict):  # handle feature extractor dict outputs
            for key in [
                "global_pool",
                "avgpool",
                "head.global_pool",
                "norm",
                "features",
            ]:
                if key in feats:
                    feats = feats[key]
                    break
            else:  # no-break
                raise ValueError(f"Unsupported feature dict keys: {list(feats.keys())}")
        if feats.dim() == 4:  # (B, C, H, W)
            feats = feats.mean(dim=(2, 3))
        if feats.dim() == 3:  # (B, T, C)
            feats = feats.mean(dim=1)
        if feats.dim() != 2:
            raise ValueError(
                f"Expected 2D feature tensor after pooling, got shape {tuple(feats.shape)}"
            )
        return feats
