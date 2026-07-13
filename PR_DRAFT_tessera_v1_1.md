## Add model: Tessera v1.1 (ucam-eo)

> **Status: draft — architecture verified with one real dataset result, full
> benchmark sweep not yet run.** This PR ports and verifies the real Tessera
> v1.1 encoder against its public checkpoint (state dict loads with **zero
> missing/unexpected keys**, tests pass, and it now has one genuine result:
> **74.4% kNN-5 accuracy** (CI 71.6-77.7%) on `m-eurosat`, ~50 minutes on an
> Apple M5 Max via `device=mps`). What's blocking a non-draft PR is compute
> at scale: this model runs every pixel of a chip through the encoder
> independently (Tessera has no spatial mixing), and the benchmark's standard
> `dataset.image_size=224` resize (applied to every model for cross-model
> comparability — not something this PR should opt out of) multiplies that
> per-pixel cost 12x over `m-eurosat`'s native 64x64 resolution. On CPU a
> single dataset is estimated at ~9 hours; MPS cut that to ~50 minutes for
> `m-eurosat`, but a full `dataset.names=all` sweep across all ~20 GeoBench
> datasets is still likely many hours even there. **Looking for a
> GPU-equipped (CUDA) contributor/maintainer to run the full sweep** — should
> be faster still there. Happy to pair on this or hand off the branch
> entirely.
>
> **Caveat on the one result above**: 74.4% likely *understates* Tessera's
> real quality, since GeoBench's format forces two deviations from how it's
> designed to run: (1) a single timestep instead of a full year of
> observations (Tessera's whole point is temporal compression — we feed a
> fixed placeholder `default_doy=182` since GeoBench chips have no real
> per-observation dates), and (2) S2-only, since SAR defaults off (see caveat
> below) — the real model fuses S1+S2, so it's only getting one of two
> trained input branches here. For reference, `timm/resnet50` (plain
> ImageNet features, no geospatial pretraining) scores 83.5% kNN on the same
> dataset — Tessera scoring lower isn't necessarily informative about
> real-world quality given how far outside its intended regime this
> evaluation runs it.
>
> **Important**: that 74.4% result currently lives in a scratch file
> (`/tmp/tessera_v1_1_smoke_results.csv`) from an early verification run, not
> in `results/all_results.csv` — the file this PR actually needs to update.
> `results/all_results.csv` currently has **zero** Tessera rows. A first
> attempt at a real multi-dataset sweep into that file crashed immediately on
> the first dataset (`dynamic_earthnet`, see #3) before computing anything, so
> as of this draft, no rows have landed in the real target file yet.

Docs:

- [Evaluate your own model](https://torchgeo.org/torchgeo-bench/user/eval_own_model.html)
- [Contribute a model](https://torchgeo.org/torchgeo-bench/user/contribute_model.html)
- [Datasets](https://torchgeo.org/torchgeo-bench/user/datasets.html)
- [Results format](https://torchgeo.org/torchgeo-bench/user/results-format.html)

### 1. Model summary

| Field | Value |
|-------|-------|
| **Model name** | `tessera_v1_1_mpc` (and `tessera_v1_1_aws` — same architecture, different checkpoint/normalization stats) |
| **Class** | `torchgeo_bench.models.TesseraV1_1BenchModel` |
| **Hydra config** | `src/torchgeo_bench/conf/model/tessera_v1_1_mpc.yaml`, `tessera_v1_1_aws.yaml` |
| **Pretraining data** | Global Sentinel-2 (+ Sentinel-1, architecturally supported but disabled by default here) full-year time series, self-supervised (Barlow Twins) |
| **Sensor coverage** | S2, 10 selected bands (B02,B03,B04,B05,B06,B07,B08,B8A,B11,B12). S1/SAR support exists (`enable_sar=true`) but is **experimental/unverified** — see caveat below |
| **Weights URL** | Originally Google Drive (see [ucam-eo/tessera, `v1.1` branch README](https://github.com/ucam-eo/tessera/tree/v1.1#v11-qat-model-weight-latest-recommended), CC0-licensed). Re-uploaded to an unofficial HF Hub mirror, [`Chesapeakeiw/tessera-v1.1-mpc-encoder`](https://huggingface.co/Chesapeakeiw/tessera-v1.1-mpc-encoder), auto-downloaded via `hf_hub_download` for `data_source="mpc"` (same pattern as this repo's `SAM3Encoder`). No mirror yet for `aws` — `checkpoint_path` still required there. |
| **Paper / project page** | [arXiv:2506.20380](https://arxiv.org/abs/2506.20380) |
| **Required extra** | N/A — pure `torch`/`torch.nn`, no new dependency |

**Caveat on SAR/S1**: Tessera expects S1 in a linear/DN-like scale (checkpoint norm stats: mean≈5588, std≈1713), which does not match GeoBench's own dB-scaled S1 bands (e.g. `benv2`'s VV: mean≈-19.4, std≈5.6). No verified dB→Tessera-linear conversion exists, so S1 defaults off (`enable_sar: false`) — datasets get a correct, documented S2-only embedding. `enable_sar: true` opts into feeding GeoBench's dB values through the linear-scale normalizer anyway, which is unverified and likely wrong; flagged clearly in the model docstring and this PR, not silently shipped as correct.

### 2. Add the model

- [x] Class inherits `BenchModel` and implements `_forward_patch_features(images) -> (B, K)`.
- [x] Class is exported from `src/torchgeo_bench/models/__init__.py` and listed in `__all__`.
- [x] Hydra config exists at `src/torchgeo_bench/conf/model/tessera_v1_1_{mpc,aws}.yaml` with the correct `_target_`.
- [x] Model weights are publicly accessible without authentication, and auto-downloaded (`mpc`, via HF mirror) with no manual file placement — verified end-to-end via a real `hf_hub_download` call. `aws` still needs `checkpoint_path` set manually pending its own mirror.
- [x] Optional dependencies — N/A, no new extra needed.
- [x] Tests cover all added code in `tests/test_tessera_v1_1.py`.
- [x] Fast tests use random tensors (a freshly-initialized fake checkpoint with real Tessera shapes) and no network I/O.
- [x] Weight-download test is marked `@pytest.mark.slow` (skipped unless `TESSERA_V1_1_MPC_CHECKPOINT` env var points at a real local checkpoint).

### 3. Run the model on every dataset

**Not done into the real results file yet.** `m-eurosat` succeeded once, but into a scratch file, not `results/all_results.csv` (see note above). A first attempt at the real multi-dataset sweep crashed on the very first dataset before computing anything (see gotcha 3 below) and needs to be rerun with that dataset excluded.

On CPU a single dataset is ~9 hours (estimated); with `device=mps` on Apple Silicon `m-eurosat` took ~50 minutes. Total across the ~12 datasets Tessera can actually run on (see skip table below) is estimated at 40-50+ hours even on MPS (dominated by feature extraction, which scales with total sample count — some V2 datasets like `benv2`/`eurosat`/`so2sat` have 5-7x more samples than `m-eurosat`). This is a genuinely multi-day job; `resume=true` makes it safe to run across multiple sessions since it skips already-completed (dataset, method, model, config) combinations.

Three gotchas hit so far, worth knowing if you pick this up:
- **MPS memory**: the per-pixel chunking loop originally accumulated all chunk outputs on-device before the final `cat`, which OOM'd on MPS (PyTorch's async op queue kept every chunk's intermediates alive until forced to synchronize). Fixed by moving each chunk's (tiny) output to CPU immediately after computing it — verified memory now stays completely flat across repeated batches. (Adds some overhead — see timing above.)
- **`faiss` has no MPS/CUDA-equivalent GPU kNN**: `device=mps` gets inherited by the kNN evaluation step by default, which then tries to call `faiss.StandardGpuResources()` — a CUDA-only API that doesn't exist for Apple Silicon, and crashes. Fix: pass `eval.knn_device=cpu` explicitly (an existing config option, [config.yaml:33](src/torchgeo_bench/conf/config.yaml:33)) to keep the expensive feature extraction on MPS while forcing the cheap kNN step to CPU.
- **`dynamic_earthnet` crashed the first sweep attempt — now genuinely fixed, not just excluded.** It's a change-detection dataset whose Planet imagery is a multi-temporal stack (`(T,C,H,W)`, `T=1` in practice), not the single-timestamp `(C,H,W)` every other dataset provides. Two real, independent bugs, both in shared code used by every model (not Tessera-specific), found by actually running it:
  1. `_ResizeTransform.__call__` ([loading.py:113](src/torchgeo_bench/datasets/loading.py:113)) assumed exactly 3D `(C,H,W)` input and crashed on the extra leading axis (`F.interpolate` saw a 4D tensor and expected 3D spatial dims instead of 2D). Fixed by flattening all leading dims into one batch dim before interpolating and restoring the original leading shape after — a no-op reshape for the standard `(C,H,W)` case, so behavior elsewhere is unchanged.
  2. **Worse, silent bug**: `geobench_v2.py`'s `chained` wrapper ([geobench_v2.py:171](src/torchgeo_bench/datasets/geobench_v2.py:171)), used for every `by_sensor`-strategy dataset, resizes each `image_*` key in a loop but never touched `"mask"` — so for any `by_sensor` + segmentation dataset without its own `canonicalize_sample` override (only `dynamic_earthnet` confirmed so far; `kuro_siwo` dodges this because its `canonicalize_sample` already folds everything into a top-level `"image"` key before the resize check), the mask silently stayed at native resolution while the image got resized — a spatial mismatch, not a crash, that would have quietly corrupted segmentation training/eval on this dataset for *any* model. Fixed by explicitly resizing `"mask"` in that branch too. Covered by 5 new regression tests in `tests/test_silent_bug_regressions.py` and `tests/test_geobench_v2_datasets.py`.

  This fix is generically useful, not Tessera-specific, and arguably belongs in its own PR rather than bundled here — flagging for discussion rather than deciding unilaterally.

Skip list, now empirically confirmed rather than guessed:

| Skipped dataset | Reason |
|-----------------|--------|
| `kuro_siwo` | SAR + DEM only, no S2 — Tessera's S2 backbone has nothing to map (verified via `BandSpec.sensor` inspection) |
| `caffe`, `m-pv4ger` | Aerial imagery only, no S2 |
| `flair2` | Aerial + elevation only, no S2 |
| `m-forestnet` | Landsat only, no S2 |
| `spacenet2` | Pan + WorldView only, no S2 |
| `spacenet7` | Planet only, no S2 |
| `burn_scars`, `forestnet`, `fotw` | Have *some* S2 bands but are missing several of the 10 specific bands Tessera requires (e.g. no rededge/NIR channels) — verified via `_band_mapping.canonical_band_name` against `_S2_BAND_ORDER` |

Confirmed usable (all have the full 10 required S2 bands): `benv2`, `cloudsen12`, `dynamic_earthnet`, `eurosat`, `eurosat-spatial`, `m-bigearthnet`, `m-brick-kiln`, `m-eurosat`, `m-so2sat`, `pastis`, `so2sat`, `treesatai` — 12 datasets. (`dynamic_earthnet` is usable again now that the resize bug above is fixed, but the sweep currently running was launched before that fix and excludes it — a follow-up run should add it back.)

```bash
uv sync --extra dev
uv run torchgeo-bench download geobench_v1
uv run torchgeo-bench download geobench_v2
uv run torchgeo-bench download eurosat

# mpc auto-downloads its checkpoint from the HF mirror; no checkpoint_path needed.
# On Apple Silicon, use device=mps + eval.knn_device=cpu (see gotchas above);
# on an NVIDIA GPU, device=cuda:0 should need neither override.
# dataset.names lists only the confirmed-usable 11 (see skip table above) --
# dataset.names=all will crash on dynamic_earthnet and the other excluded ones.
uv run torchgeo-bench run model=tessera_v1_1_mpc \
  'dataset.names=[pastis,m-eurosat,cloudsen12,treesatai,m-brick-kiln,so2sat,m-so2sat,m-bigearthnet,eurosat,eurosat-spatial,benv2]' \
  dataset.bands=all \
  output=results/all_results.csv \
  resume=true \
  device=<cuda:0|cpu>
```

### 4. Commit results

**Not yet done** — blocked on #3.

- [ ] New rows are committed to `results/all_results.csv`.
- [ ] Added result rows are only for this model and the command above.
- [ ] No existing `results/all_results.csv` rows were reordered, edited, or removed.
- [ ] Result rows match the documented CSV schema.
- [ ] Added row count: TBD

### 5. Reproduction details

```bash
git checkout claude/elated-lumiere-cbf1a1
uv sync --extra dev
uv run torchgeo-bench run model=tessera_v1_1_mpc \
  'dataset.names=[pastis,m-eurosat,cloudsen12,treesatai,m-brick-kiln,so2sat,m-so2sat,m-bigearthnet,eurosat,eurosat-spatial,benv2]' \
  dataset.bands=all \
  output=results/all_results.csv \
  resume=true \
  device=mps \
  eval.knn_device=cpu
```

| Field | Value |
|-------|-------|
| **Commit SHA** | TBD — nothing committed yet on this branch |
| **OS** | macOS 26.5.1 |
| **CPU/GPU** | Apple M5 Max, `device=mps` (a CUDA run would likely be faster still and needs no `eval.knn_device` override) |
| **Python** | 3.13.13 |
| **torchgeo-bench command** | see above |

### 6. Local checks

- [x] `uv run pytest --no-cov tests/test_tessera_v1_1.py` passes locally (8 fast tests).
- [x] `uv run pytest --no-cov -m slow tests/test_tessera_v1_1.py` passes locally against the real downloaded checkpoint.
- [x] `uv run pytest --no-cov` passes locally (368 passed, 64 skipped, 0 failed — full suite, reverified after the pixel-chunking fix, the HF auto-download change, and the `dynamic_earthnet` resize fix).
- [x] `uv run ruff check . && uv run ruff format --check .` passes on all files touched this session.

### Notes

- **`dynamic_earthnet`'s crash is fixed, not just excluded.** It caused a real crash on the first attempt at the full sweep. Root cause and fix are in gotcha 3 of section 3 — two real bugs in shared code (a resize-transform shape assumption, and a mask that silently never got resized for `by_sensor` datasets), both fixed and covered by 5 new regression tests. It's back in the confirmed-usable list; the sweep currently running just predates the fix and doesn't include it.

### Related, parked work in this branch

This branch also contains an unrelated, **parked** feature: a
geolocation-based precomputed-embedding lookup path (`GeoTesseraEmbeddingBenchModel`,
`requires_geolocation` on `BenchModel`, `geo_fields` on `BenchDataset`) that
looks up Tessera's *public, precomputed* embeddings by lat/lon/year instead of
running real inference. It's functional but untested, and orthogonal to this
model — worth splitting into its own PR rather than bundling here.
