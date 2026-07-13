"""Tessera v1.1 encoder wrapper for torchgeo-bench.

Ported from ``ucam-eo/tessera``'s ``tessera_infer_QAT/src/models/ssl_model_v1_1.py``
(``v1.1`` branch), against a real, publicly downloadable (CC0-licensed)
checkpoint. Tessera's own checkpoints are hosted on Google Drive, not a
stable auto-fetch target like HuggingFace Hub -- so for ``data_source="mpc"``
this wrapper auto-downloads from an unofficial HF Hub mirror
(``Chesapeakeiw/tessera-v1.1-mpc-encoder``, re-uploaded from the original
CC0-licensed Drive link) via :func:`huggingface_hub.hf_hub_download`, same
pattern as every other auto-downloading model in this package. Pass
``checkpoint_path`` to override with a local file instead (required for
``data_source="aws"``, which has no HF mirror yet).

Tessera is a genuinely per-pixel model: the encoder has no spatial mixing at
all, so this wrapper runs every pixel of a chip through the encoder
independently and mean-pools the resulting embeddings — mean-pooling the raw
pixels *before* the (nonlinear) encoder would silently produce a different,
wrong result.

GeoBench chips are single-timestamp, so the day-of-year the encoder expects
per observation is approximated via ``default_doy`` (documented
approximation, same category as OlmoEarth's ``time_steps`` repeat-hack
elsewhere in this package) — real Tessera inference uses a full year of
observations per pixel.

SAR/S1 support is disabled by default: Tessera expects S1 in a linear
digital-number-like scale (e.g. mean ~5588), which does not match GeoBench's
own dB-scaled S1 bands (e.g. mean ~-19 for VV in ``benv2``). No verified
conversion between the two exists, and feeding dB values through the
linear-scale normalizer would silently produce wrong (not absent) SAR
features. Pass ``enable_sar=True`` to opt into this experimental,
unverified path anyway.
"""

import logging
from typing import Literal

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

from torchgeo_bench.datasets.base import BandSpec

from ._band_mapping import canonical_band_name, map_to_model_bands
from ._tessera_v1_1_modules import TransformerEncoder, build_dim_reducer
from .interface import BenchModel

logger = logging.getLogger(__name__)


# NOT wavelength-ascending (B02..B12) -- this is Tessera's own non-conventional
# training-time channel order, produced by their Rust `s2_stack` preprocessing
# tool. Confirmed against the authoritative table in the upstream README
# (https://github.com/ucam-eo/tessera/pull/22) and the equivalent bug fixed in
# torchgeo's own Tessera wrapper (https://github.com/torchgeo/torchgeo/pull/3673,
# reported by the Tessera team in torchgeo#3672). NORM_STATS below is copied
# verbatim from v1_1_norm_stats.py and is bound to this exact order -- getting
# it wrong doesn't error, it silently normalizes/feeds bands into the wrong
# slots and degrades embedding quality.
_S2_BAND_ORDER = [
    "red", "blue", "green", "nir", "nir_narrow",
    "rededge1", "rededge2", "rededge3", "swir1", "swir2",
]  # fmt: skip
_S1_BAND_ORDER = ["vv", "vh"]

# Unofficial HF Hub mirrors of Tessera's CC0-licensed Google Drive checkpoints
# (re-uploaded since Drive has no huggingface_hub-style auto-download API).
# The Tessera team has said an official mirror will follow eventually --
# prefer that once it exists.
_HF_MIRRORS: dict[str, tuple[str, str]] = {
    "mpc": ("Chesapeakeiw/tessera-v1.1-mpc-encoder", "tessera_v1_1_mpc_encoder.pt"),
}

# Verbatim from tessera_infer_QAT/src/datasets/v1_1_norm_stats.py (v1.1 branch).
NORM_STATS = {
    "mpc": {
        "s2_mean": [2683.4553, 2223.3630, 2432.0950, 3633.1970, 3602.1755,
                    3006.4324, 3400.2710, 3515.6392, 2456.9163, 1983.8783],
        "s2_std": [2739.5217, 2846.2993, 2690.8250, 2290.0439, 2088.8970,
                   2673.1106, 2381.4521, 2229.5225, 1601.0942, 1495.3545],
        "s1a_mean": [5588.3291, 3025.6270],
        "s1a_std": [1713.4646, 1693.0471],
        "s1d_mean": [5552.9683, 2955.0520],
        "s1d_std": [1685.5857, 1677.6414],
    },
    "aws": {
        "s2_mean": [2793.6589, 2356.7776, 2551.0496, 3741.9229, 3713.7844,
                    3120.1997, 3516.3342, 3637.0342, 2501.0283, 2038.1504],
        "s2_std": [2810.0093, 2933.8835, 2755.6360, 2344.5027, 2145.7986,
                   2743.9019, 2438.8601, 2286.5977, 1680.7367, 1585.5529],
        "s1a_mean": [5697.0859, 2838.6687],
        "s1a_std": [1671.3737, 1789.4116],
        "s1d_mean": [5759.1367, 2873.2854],
        "s1d_std": [1583.2858, 1747.8390],
    },
}  # fmt: skip


class _MultimodalV1_1InferenceModel(nn.Module):
    """Two backbones (S2 + S1) fused through an MLP ``dim_reducer``."""

    def __init__(
        self,
        s2_backbone: nn.Module,
        s1_backbone: nn.Module,
        dim_reducer: nn.Module,
        fusion_method: str = "concat",
    ) -> None:
        super().__init__()
        self.s2_backbone = s2_backbone
        self.s1_backbone = s1_backbone
        self.dim_reducer = dim_reducer
        self.fusion_method = fusion_method

    def forward(self, s2_x: torch.Tensor, s1_x: torch.Tensor) -> torch.Tensor:
        reprs = [self.s2_backbone(s2_x), self.s1_backbone(s1_x)]
        if self.fusion_method == "concat":
            fused = torch.cat(reprs, dim=-1)
        elif self.fusion_method == "sum":
            fused = sum(reprs)
        else:
            raise ValueError(f"Unknown fusion_method: {self.fusion_method}")
        return self.dim_reducer(fused)


def _load_checkpoint(encoder: nn.Module, checkpoint_path: str) -> dict:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    raw = ckpt["model_state"]
    cleaned = {}
    for k, v in raw.items():
        if k.startswith("_orig_mod."):
            k = k[len("_orig_mod.") :]
        if k.startswith("projector.") or k.startswith("segmented_matryoshka_projector."):
            continue
        cleaned[k] = v
    missing, unexpected = encoder.load_state_dict(cleaned, strict=False)
    logger.info(
        "Loaded Tessera v1.1 checkpoint %s: missing=%d unexpected=%d",
        checkpoint_path,
        len(missing),
        len(unexpected),
    )
    if missing:
        logger.info("Missing keys (first 10): %s", missing[:10])
    if unexpected:
        logger.info("Unexpected keys (first 10): %s", unexpected[:10])
    return ckpt.get("config", {})


class TesseraV1_1BenchModel(BenchModel):
    """Real Tessera v1.1 encoder (S2 + optional S1 transformer backbones).

    Args:
        bands: Ordered list of :class:`BandSpec`.
        checkpoint_path: Local path to a ``tessera_v1_1_{mpc,aws}_encoder.pt``
            file. Optional for ``data_source="mpc"`` (auto-downloaded from
            the HF mirror if omitted); required for ``"aws"`` (no mirror
            exists yet).
        data_source: Which checkpoint flavour to use (``"mpc"`` or
            ``"aws"``) — selects the matching normalization stats. Must
            match the actual checkpoint or embeddings silently degrade.
        default_doy: Day-of-year (1-366) used for every sample, since
            GeoBench chips lack real per-observation acquisition dates.
        enable_sar: Opt into feeding GeoBench's dB-scaled S1 bands through
            Tessera's linear-scale S1 normalizer. Experimental/unverified —
            see module docstring.
        pixel_chunk_size: Tessera has no spatial mixing — every pixel of a
            chip is run through the encoder independently, then mean-pooled.
            Flattening a whole batch of chips into one giant
            ``(B*H*W, ...)`` call scales memory with batch_size × H × W and
            can exhaust RAM on repeated calls (observed: a single
            32×64×64 batch peaked at ~4.7GB, and a full dataset pass over
            many such batches was OOM-killed). Processing ``pixel_chunk_size``
            pixels at a time bounds peak memory regardless of batch size or
            chip resolution, at the cost of wall-clock time.
    """

    def __init__(
        self,
        bands: list[BandSpec],
        *,
        checkpoint_path: str | None = None,
        data_source: Literal["mpc", "aws"] = "mpc",
        default_doy: int = 182,
        enable_sar: bool = False,
        pixel_chunk_size: int = 8192,
        **_kwargs: object,
    ) -> None:
        super().__init__(bands=bands, normalization="identity")
        if data_source not in NORM_STATS:
            raise ValueError(
                f"data_source must be one of {sorted(NORM_STATS)}, got {data_source!r}."
            )
        if not checkpoint_path:
            if data_source not in _HF_MIRRORS:
                raise ValueError(
                    f"No HF mirror available for data_source={data_source!r}; pass "
                    "checkpoint_path explicitly. Download "
                    f"tessera_v1_1_{data_source}_encoder.pt from "
                    "https://github.com/ucam-eo/tessera/tree/v1.1 and set, e.g., "
                    f"model.checkpoint_path=/path/to/tessera_v1_1_{data_source}_encoder.pt"
                )
            repo_id, filename = _HF_MIRRORS[data_source]
            logger.info("Auto-downloading Tessera v1.1 checkpoint from %s/%s", repo_id, filename)
            checkpoint_path = hf_hub_download(repo_id=repo_id, filename=filename)
        self.data_source = data_source
        self.default_doy = default_doy
        self.enable_sar = enable_sar
        self.pixel_chunk_size = pixel_chunk_size

        canon = {canonical_band_name(b.name) for b in bands}
        self._has_sar = enable_sar and "vv" in canon and "vh" in canon
        if enable_sar and not self._has_sar:
            logger.warning(
                "enable_sar=True but dataset lacks vv/vh bands; falling back to zero-filled SAR."
            )

        ckpt_cfg = torch.load(checkpoint_path, map_location="cpu", weights_only=True)["config"]
        latent_dim = int(ckpt_cfg.get("latent_dim", 192))
        repr_dim = int(ckpt_cfg.get("representation_dim", latent_dim))
        fusion_method = ckpt_cfg.get("fusion_method", "concat")
        self.eval_repr_dim = int(ckpt_cfg.get("eval_repr_dim", 128))

        s2_backbone = TransformerEncoder(
            band_num=10,
            latent_dim=latent_dim,
            nhead=int(ckpt_cfg["s2_num_heads"]),
            num_encoder_layers=int(ckpt_cfg["s2_num_layers"]),
            dim_feedforward=int(ckpt_cfg["s2_dim_feedforward"]),
            dropout=0.1,
        )
        s1_backbone = TransformerEncoder(
            band_num=2,
            latent_dim=latent_dim,
            nhead=int(ckpt_cfg["s1_num_heads"]),
            num_encoder_layers=int(ckpt_cfg["s1_num_layers"]),
            dim_feedforward=int(ckpt_cfg["s1_dim_feedforward"]),
            dropout=0.1,
        )
        active_backbones = 2 if fusion_method == "concat" else 1
        dim_reducer = build_dim_reducer(latent_dim * 4 * active_backbones, repr_dim)

        self.encoder = _MultimodalV1_1InferenceModel(
            s2_backbone, s1_backbone, dim_reducer, fusion_method
        )
        _load_checkpoint(self.encoder, checkpoint_path)

    def _forward_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        device, dtype = images.device, images.dtype
        b, _, h, w = images.shape
        stats = NORM_STATS[self.data_source]

        s2_img, _ = map_to_model_bands(images, self.bands, _S2_BAND_ORDER, allow_missing=False)
        s2_mean = torch.tensor(stats["s2_mean"], device=device, dtype=dtype).view(1, -1, 1, 1)
        s2_std = torch.tensor(stats["s2_std"], device=device, dtype=dtype).view(1, -1, 1, 1)
        s2_img = (s2_img - s2_mean) / (s2_std + 1e-9)
        s2_pixels = s2_img.permute(0, 2, 3, 1).reshape(-1, 10)  # (B*H*W, 10)

        doy_col = torch.full(
            (s2_pixels.shape[0], 1), float(self.default_doy), device=device, dtype=dtype
        )
        s2_seq = torch.cat([s2_pixels, doy_col], dim=-1).unsqueeze(1)  # (B*H*W, 1, 11)

        if self._has_sar:
            s1_img, _ = map_to_model_bands(images, self.bands, _S1_BAND_ORDER, allow_missing=False)
            s1_mean = torch.tensor(stats["s1a_mean"], device=device, dtype=dtype).view(1, -1, 1, 1)
            s1_std = torch.tensor(stats["s1a_std"], device=device, dtype=dtype).view(1, -1, 1, 1)
            s1_img = (s1_img - s1_mean) / (s1_std + 1e-9)
            s1_pixels = s1_img.permute(0, 2, 3, 1).reshape(-1, 2)
            s1_seq = torch.cat([s1_pixels, doy_col], dim=-1).unsqueeze(1)  # (B*H*W, 1, 3)
        else:
            s1_seq = torch.zeros((s2_pixels.shape[0], 1, 3), device=device, dtype=dtype)

        # No spatial mixing in the encoder -- process pixels in bounded chunks
        # so peak memory doesn't scale with batch_size * H * W (see
        # pixel_chunk_size docstring). Move each chunk's (tiny) output to CPU
        # immediately rather than accumulating on-device: on MPS, PyTorch's
        # async op queue otherwise keeps every chunk's intermediate buffers
        # alive until the final torch.cat forces synchronization, ballooning
        # peak memory far past what a single chunk needs (observed: OOM on a
        # unified-memory system with plenty of headroom for one chunk at a
        # time). extract_features() moves the result to CPU/numpy right
        # after this returns anyway, so this isn't extra work.
        n = s2_seq.shape[0]
        chunks = []
        for start in range(0, n, self.pixel_chunk_size):
            end = min(start + self.pixel_chunk_size, n)
            chunk_out = self.encoder(s2_seq[start:end], s1_seq[start:end])
            chunks.append(chunk_out[:, : self.eval_repr_dim].cpu())
        out = torch.cat(chunks, dim=0)  # (B*H*W, eval_repr_dim), on CPU
        return out.view(b, h, w, self.eval_repr_dim).mean(dim=(1, 2))  # (B, eval_repr_dim)
