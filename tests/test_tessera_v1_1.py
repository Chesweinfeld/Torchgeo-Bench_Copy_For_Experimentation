"""Tests for :class:`TesseraV1_1BenchModel`.

The fast tests build a tiny, freshly-initialized checkpoint on disk (small
``latent_dim``, 1 transformer layer) so the real ``__init__``/checkpoint-loading
code path is exercised without downloading the ~230MB real weights. The slow
test loads the actual checkpoint if present locally.
"""

import os

import pytest
import torch

from torchgeo_bench.datasets.base import BandSpec
from torchgeo_bench.models._band_mapping import map_to_model_bands
from torchgeo_bench.models._tessera_v1_1_modules import (
    CustomTemporalAwarePooling,
    TransformerEncoder,
    build_dim_reducer,
)
from torchgeo_bench.models.tessera_v1_1 import _S2_BAND_ORDER, TesseraV1_1BenchModel

REAL_CHECKPOINT_PATH = os.environ.get("TESSERA_V1_1_MPC_CHECKPOINT", "")


def _s2_bands() -> list[BandSpec]:
    names = [
        "blue", "green", "red", "red_edge_1", "red_edge_2", "red_edge_3",
        "nir", "red_edge_4", "swir_1", "swir_2",
    ]  # fmt: skip
    return [
        BandSpec(
            sensor="s2", name=n, source_name=n.upper(), mean=1500.0, std=600.0, min=0.0, max=10000.0
        )
        for n in names
    ]


def _s1_bands() -> list[BandSpec]:
    return [
        BandSpec(
            sensor="s1", name=n, source_name=n.upper(), mean=-15.0, std=5.0, min=-30.0, max=5.0
        )
        for n in ("vv", "vh")
    ]


def _write_fake_checkpoint(path, *, latent_dim=4, nhead=2, layers=1, ff=8):
    """A tiny, freshly-initialized checkpoint with real Tessera v1.1 shapes/config keys."""
    s2 = TransformerEncoder(
        band_num=10,
        latent_dim=latent_dim,
        nhead=nhead,
        num_encoder_layers=layers,
        dim_feedforward=ff,
    )
    s1 = TransformerEncoder(
        band_num=2,
        latent_dim=latent_dim,
        nhead=nhead,
        num_encoder_layers=layers,
        dim_feedforward=ff,
    )
    reducer = build_dim_reducer(latent_dim * 4 * 2, latent_dim * 4)
    state = {
        **{f"s2_backbone.{k}": v for k, v in s2.state_dict().items()},
        **{f"s1_backbone.{k}": v for k, v in s1.state_dict().items()},
        **{f"dim_reducer.{k}": v for k, v in reducer.state_dict().items()},
    }
    config = {
        "latent_dim": latent_dim,
        "representation_dim": latent_dim * 4,
        "eval_repr_dim": min(8, latent_dim * 4),
        "fusion_method": "concat",
        "s2_num_heads": nhead,
        "s2_num_layers": layers,
        "s2_dim_feedforward": ff,
        "s1_num_heads": nhead,
        "s1_num_layers": layers,
        "s1_dim_feedforward": ff,
    }
    torch.save({"model_state": state, "config": config}, path)


def test_temporal_pooling_t1_is_plain_squeeze():
    """T==1 (GeoBench's case) must bypass the GRU/attention math entirely."""
    pool = CustomTemporalAwarePooling(16)
    x = torch.randn(3, 1, 16)
    assert torch.equal(pool(x), x.squeeze(1))


def test_missing_checkpoint_path_raises_for_aws():
    """aws has no HF mirror yet, so checkpoint_path is still required for it."""
    with pytest.raises(ValueError, match="checkpoint_path"):
        TesseraV1_1BenchModel(bands=_s2_bands(), checkpoint_path="", data_source="aws")


def test_mpc_auto_downloads_when_checkpoint_path_omitted(tmp_path, monkeypatch):
    """mpc has an HF mirror, so omitting checkpoint_path should auto-download."""
    ckpt = tmp_path / "fake.pt"
    _write_fake_checkpoint(ckpt)
    calls = []

    def fake_hf_hub_download(*, repo_id, filename):
        calls.append((repo_id, filename))
        return str(ckpt)

    monkeypatch.setattr("torchgeo_bench.models.tessera_v1_1.hf_hub_download", fake_hf_hub_download)
    model = TesseraV1_1BenchModel(bands=_s2_bands(), data_source="mpc")
    assert calls == [("Chesapeakeiw/tessera-v1.1-mpc-encoder", "tessera_v1_1_mpc_encoder.pt")]
    x = torch.rand(2, 10, 4, 4) * 3000
    with torch.no_grad():
        out = model.forward_patch_features(x)
    assert out.shape == (2, model.eval_repr_dim)


def test_bad_data_source_raises(tmp_path):
    ckpt = tmp_path / "fake.pt"
    _write_fake_checkpoint(ckpt)
    with pytest.raises(ValueError, match="data_source"):
        TesseraV1_1BenchModel(bands=_s2_bands(), checkpoint_path=str(ckpt), data_source="bogus")


def test_forward_shape_s2_only(tmp_path):
    ckpt = tmp_path / "fake.pt"
    _write_fake_checkpoint(ckpt)
    model = TesseraV1_1BenchModel(bands=_s2_bands(), checkpoint_path=str(ckpt), data_source="mpc")
    x = torch.rand(2, 10, 4, 4) * 3000
    with torch.no_grad():
        out = model.forward_patch_features(x)
    assert out.shape == (2, model.eval_repr_dim)
    assert torch.isfinite(out).all()


def test_forward_shape_with_sar_disabled_by_default(tmp_path):
    """Dataset has S1 bands, but enable_sar defaults to False -> zero-filled, no crash."""
    ckpt = tmp_path / "fake.pt"
    _write_fake_checkpoint(ckpt)
    bands = _s2_bands() + _s1_bands()
    model = TesseraV1_1BenchModel(bands=bands, checkpoint_path=str(ckpt), data_source="mpc")
    assert model._has_sar is False
    x = torch.rand(2, 12, 4, 4) * 3000
    with torch.no_grad():
        out = model.forward_patch_features(x)
    assert out.shape == (2, model.eval_repr_dim)


def test_forward_shape_with_sar_enabled(tmp_path):
    ckpt = tmp_path / "fake.pt"
    _write_fake_checkpoint(ckpt)
    bands = _s2_bands() + _s1_bands()
    model = TesseraV1_1BenchModel(
        bands=bands, checkpoint_path=str(ckpt), data_source="mpc", enable_sar=True
    )
    assert model._has_sar is True
    x = torch.rand(2, 12, 4, 4) * 3000
    with torch.no_grad():
        out = model.forward_patch_features(x)
    assert out.shape == (2, model.eval_repr_dim)
    assert torch.isfinite(out).all()


def test_state_dict_round_trip_no_missing_or_unexpected(tmp_path, caplog):
    """The exact scenario that matters most: a real save/load round-trip must be clean."""
    import logging

    ckpt = tmp_path / "fake.pt"
    _write_fake_checkpoint(ckpt)
    with caplog.at_level(logging.INFO, logger="torchgeo_bench.models.tessera_v1_1"):
        TesseraV1_1BenchModel(bands=_s2_bands(), checkpoint_path=str(ckpt), data_source="mpc")
    assert "missing=0 unexpected=0" in caplog.text


def test_s2_band_order_matches_tessera_training_order():
    """Pins Tessera's non-conventional S2 channel order.

    Confirmed authoritative in the upstream README table
    (https://github.com/ucam-eo/tessera/pull/22) and the matching bug fixed in
    torchgeo's own wrapper (torchgeo#3672 / torchgeo#3673, reported by the
    Tessera team). Getting this wrong doesn't raise -- it silently normalizes
    and feeds bands into the wrong slots, degrading embedding quality without
    any error. An earlier version of this file used the wavelength-ascending
    (B02..B12) order, which is wrong.
    """
    assert _S2_BAND_ORDER == [
        "red", "blue", "green", "nir", "nir_narrow",
        "rededge1", "rededge2", "rededge3", "swir1", "swir2",
    ]  # fmt: skip


def test_s2_bands_land_in_correct_channel_slots():
    """End-to-end check that real pixel data ends up in the right slot.

    Builds a source image where each canonical band is filled with a distinct
    constant equal to its *correct* target index, in an arbitrary (non-Tessera)
    source order, then asserts ``map_to_model_bands`` placed each one at the
    index the upstream README table specifies -- not merely that the constant
    list looks right, but that real data actually lands correctly.
    """
    target_index = {name: i for i, name in enumerate(_S2_BAND_ORDER)}
    # Deliberately different order from _S2_BAND_ORDER.
    src_order = ["blue", "green", "red", "rededge1", "rededge2", "rededge3", "nir", "nir_narrow", "swir1", "swir2"]  # fmt: skip
    src_bands = [
        BandSpec(sensor="s2", name=n, source_name=n.upper(), mean=0.0, std=1.0, min=0.0, max=1.0)
        for n in src_order
    ]
    images = torch.zeros(1, len(src_order), 2, 2)
    for i, name in enumerate(src_order):
        images[:, i] = float(target_index[name])

    mapped, missing = map_to_model_bands(images, src_bands, _S2_BAND_ORDER, allow_missing=False)
    assert not any(missing)
    for name, expected_idx in target_index.items():
        assert torch.all(mapped[:, expected_idx] == float(expected_idx)), (
            f"{name} should land at channel {expected_idx}"
        )


@pytest.mark.slow
@pytest.mark.skipif(
    not REAL_CHECKPOINT_PATH, reason="set TESSERA_V1_1_MPC_CHECKPOINT to a local .pt path"
)
def test_real_checkpoint_loads_and_runs():
    model = TesseraV1_1BenchModel(
        bands=_s2_bands(), checkpoint_path=REAL_CHECKPOINT_PATH, data_source="mpc"
    )
    x = torch.rand(1, 10, 8, 8) * 3000
    with torch.no_grad():
        out = model.forward_patch_features(x)
    assert out.shape == (1, 128)
    assert torch.isfinite(out).all()
