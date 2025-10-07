"""Benchmark model implementations wrapping existing architectures.

These classes inherit from ``BenchModel`` and ensure a consistent
``forward_features`` contract.
"""

from __future__ import annotations

import timm
import torch

from . import models as legacy_models  # reuse existing RCF definition
from .interface import BenchModel

__all__ = [
    "RCFBench",
    "TimmResNet50Bench",
    "ImageStatsBench",
]


class RCFBench(BenchModel):
    """Wrapper for the existing ``RCF`` implementation.

    Parameters mirror ``src.models.RCF`` with explicit ``num_channels``.
    """

    def __init__(
        self,
        num_channels: int,
        features: int = 512,
        kernel_size: int = 3,
        mode: str = "gaussian",
        stats_mode: str = "mean",
        seed: int | None = None,
        dataset=None,
    ) -> None:
        super().__init__(num_channels=num_channels)
        self.rcf = legacy_models.RCF(
            in_channels=num_channels,
            features=features,
            kernel_size=kernel_size,
            mode=mode,
            stats_mode=stats_mode,
            seed=seed,
            dataset=dataset,
        )

    def forward_features(
        self,
        images: torch.Tensor,
        bboxes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.rcf(images)


class TimmResNet50Bench(BenchModel):
    """ResNet50 backbone from timm returning pooled embeddings."""

    def __init__(
        self,
        num_channels: int,
        pretrained: bool = True,
        global_pool: str = "avg",
        seed: int | None = None,
        **_,
    ) -> None:  # ignore extra kwargs (e.g., dataset) for compatibility
        super().__init__(num_channels=num_channels)
        if seed is not None:
            torch.manual_seed(seed)
        # num_classes=0 -> feature extraction mode in timm
        self.backbone = timm.create_model(
            "resnet50",
            pretrained=pretrained,
            num_classes=0,
            in_chans=num_channels,
            global_pool=global_pool,
        )

    def forward_features(
        self,
        images: torch.Tensor,
        bboxes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        feats = self.backbone(images)
        if isinstance(feats, dict):  # safety: some timm models return dict
            feats = next(iter(feats.values()))
        if feats.dim() == 4:  # (B,C,H,W)
            feats = feats.mean(dim=(2, 3))
        if feats.dim() != 2:
            raise ValueError(f"Unexpected feature shape {tuple(feats.shape)}")
        return feats


class ImageStatsBench(BenchModel):
    """BenchModel that returns per-image statistics (mean, std, min, max) as features."""

    def __init__(self, num_channels: int, **kwargs) -> None:
        super().__init__(num_channels=num_channels)

    def forward_features(
        self,
        images: torch.Tensor,
        bboxes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # images: (B, C, H, W)
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


class TimmCNNBench(BenchModel):
    """Generic timm CNN backbone returning pooled embeddings.
    Specify any timm model name (e.g., 'resnet18', 'efficientnet_b0').
    """

    def __init__(
        self,
        num_channels: int,
        model_name: str = "resnet18",
        pretrained: bool = True,
        global_pool: str = "avg",
        seed: int | None = None,
        **_,
    ) -> None:
        super().__init__(num_channels=num_channels)
        if seed is not None:
            torch.manual_seed(seed)
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            in_chans=num_channels,
            global_pool=global_pool,
        )

    def forward_features(
        self,
        images: torch.Tensor,
        bboxes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        feats = self.backbone(images)
        if isinstance(feats, dict):
            feats = next(iter(feats.values()))
        if feats.dim() == 4:
            feats = feats.mean(dim=(2, 3))
        if feats.dim() != 2:
            raise ValueError(f"Unexpected feature shape {tuple(feats.shape)}")
        return feats

__all__.append('TimmCNNBench')
