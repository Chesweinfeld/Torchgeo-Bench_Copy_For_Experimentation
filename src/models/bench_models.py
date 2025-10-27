import torch

from .models import RCF
from .interface import BenchModel


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
        **kwargs
    ) -> None:
        super().__init__(num_channels=num_channels)
        self.rcf = RCF(
            in_channels=num_channels,
            features=features,
            kernel_size=kernel_size,
            mode=mode,
            stats_mode=stats_mode,
            seed=seed,
            dataset=dataset,
        )

    def forward_patch_features(
        self,
        images: torch.Tensor,
        bboxes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.rcf(images)


class ImageStatsBench(BenchModel):
    """BenchModel that returns per-image statistics (mean, std, min, max) as features."""

    def __init__(self, num_channels: int, **kwargs) -> None:
        super().__init__(num_channels=num_channels)

    def forward_patch_features(
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
