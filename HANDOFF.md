# Session handoff — dissertation data foundation

**Branch:** `phase3-l3-rasterization` (merged `nespreso-v2-port` @ `38191d7`)  
**Base:** legacy ISAS production on `nespreso-v2-port` — not replaced by dissertation work  
**Updated:** 2026-06-18  
**Code home:** [`NeSPReSO2_onTemplate/`](NeSPReSO2_onTemplate/)

**Read first:** [`PLAN.md`](PLAN.md), [`PLAN-dissertation-data-foundation.md`](PLAN-dissertation-data-foundation.md), [`PLAN-phase3-l3-rasterization.md`](PLAN-phase3-l3-rasterization.md).  
**L3 download homework (user):** [`L3-DOWNLOAD-HOMEWORK.md`](L3-DOWNLOAD-HOMEWORK.md)
**NEXT-STEPS PLAN (2026-07-15):** [`PLAN-agentic-close-out.md`](PLAN-agentic-close-out.md) — close Track 0.2 (retrain done, evaluate it), **retire Track B**, then RC-4 MC-dropout ensemble → `point_cube` σ₀ diagnosis → `export_field_product.py` fixes → anomaly parity gap. Start here.
**Agentic AI experiment (2026-07-15, Tracks 0 + A DONE, B retired):** [`PLAN-agentic-ai-experiment.md`](PLAN-agentic-ai-experiment.md) — measure `readiness.py` (never run), harden `bench_datacube_speed.py` into a real evaluator, then a controlled hand-vs-OpenEvolve comparison on `PLAN_datacube_speed.md` Phase 5. Source brief: [`agentic-science.html`](agentic-science.html)
**Track 0 results (2026-07-15):** [`HANDOFF-2026-07-15-agentic-track0.md`](HANDOFF-2026-07-15-agentic-track0.md) — RC-1/RC-2 measured for the first time. **Two plan assumptions invalidated:** σ₀ stability is already slack (models are *over-smoothed*, 0.00% violations vs nature's 24.7%; except `point_cube` at 38.5%), and **steric-vs-SLA is saturated** (model r=0.8299 vs true-profile ceiling r=0.8297) — so it is *not* a usable science-loop objective. Fixed two latent `readiness.py` bugs that returned plausible-looking wrong numbers. Anom loss scales re-derived (raw-PC scales confirmed, but only a 9.7% T:S effect).
**Track A results (2026-07-15):** [`HANDOFF-2026-07-15-agentic-track-a.md`](HANDOFF-2026-07-15-agentic-track-a.md) — **the golden gate was BROKEN at HEAD and had been for 10 days.** All three goldens failed against the repo's own cube: they were saved 07-05 14:24–15:28, the cube was rebuilt rev2→rev3 (double-decode fix) at 07-06T00:47Z, and nothing re-derived them. They asserted a **2.87–3.06 °C Gulf of Mexico** (true 22.7–29.6 °C) — the gate would have rejected every *correct* candidate. Goldens regenerated from the **committed** sampler + rev-3 cube; `data_revision` now stamped in `.meta.json` and asserted at check time; plausibility + end-to-end tests added. **Evaluator noise floor: σ = 7.36%** of median → Track B must use min-of-N≥5 and treat sub-10% wins as noise.
**RUNNING now (2026-07-15 15:10):** `anom_point` loss-scale retrain in tmux **`anom_retune`** (GPU 2) — closes the loss-scale question; parallel to Track A. Log: `NeSPReSO2_onTemplate/saved/readiness/retune_retune_0715_anom_point.log`. Beat ANOM-point 0.680/0.104. *Predicted not to reach parity* — a 9.7% T:S rebalance is a small lever.
**Scoped from Track 0:** `point_cube` σ₀ **38.5%** vs nature's 24.7% — the only model less stable than the ocean. Now `PLAN-agentic-ai-experiment.md` Track 0.5 + `PLAN.md` Phase 8 (scoped to `point_cube` only). Diagnose (contrast vs `residual_cube` 2.57%; check cube-feature standardization) **before** any physics-loss term.
**Latest session handoff (2026-07-05):** [`HANDOFF-2026-07-05-full-scratch-notebook.md`](HANDOFF-2026-07-05-full-scratch-notebook.md) — from-scratch all-models notebook (cube rebuild rev 3 + retrain incl. cube-native) running in tmux `scratch_nb`
**Previous (2026-07-03):** [`HANDOFF-2026-07-03-l4-stale-sat.md`](HANDOFF-2026-07-03-l4-stale-sat.md) — L4 patch root cause (stale satellite ≥2021-01), gap downloads running in tmux `satdl`

**Conda env:** `nespreso` (has `netCDF4`, `copernicusmarine`; use for selfcheck/train).

---

## What this branch is

GoM dissertation NeSPReSO: **ARGO/CORA subsurface targets**, **mask-native L3 surface inputs** (not L4 gridded truth), **chronological splits**. Legacy ISAS production (`config/isas/config_isas_patch.json`, PCA-16) lives on `nespreso-v2-port` and is not replaced by this work.

---

## Phase status

| Phase | PLAN step | Status | Key artifacts |
|-------|-----------|--------|---------------|
| 0 Data census | 1–2 | **Done** | `scripts/data_census.py` → `reports/data_census.*`, `reports/split_design.*` |
| 1 Chronological split | 3–4 | **Done** | `base/split_utils.py`; `split_mode: chronological` in ARGO configs |
| 2 ARGO-first path | 5–6 | **Done** | `export_v2_cache.py`, `config_argo*.json`, chronological eval path |
| 3 L3 pipeline | 7–9 | **Done** | `download_l3_products.py`, `l3_rasterize.py`, `export_l3_cache.py`, `build_l3_samples.py` |
| 4 L4 augmentation | 10 | **Done** | `preproc/l4_augment.py`, `preproc/l4_rasterize.py`, `io.l4` wired in `export_l3_cache.py` |
| 5 L3 model channels | 11 | **Done** | `l3_input.sync_arch_with_io`, `verify_l3_cache_layout`, checkpoint L3 metadata |
| 6 Stratified eval | 12 | **Done** | `eval_stratified.py` — RMSE/bias by L3 coverage, track distance, census year subsets |
| 7 Readiness diagnostics | 13 | **Done + RUN 2026-07-15** | `diagnostics/readiness.py` → `saved/readiness/readiness_*.{json,md}` (5 models). RC-1 σ₀ **measured**; RC-2 steric-vs-SLA **wired + measured** (was never wired). Two silent-wrong-number bugs fixed. |
| 8 Physics loss | 14 | **Pending — now scoped to `point_cube` only** | RC-1 says 4 of 5 models are *over-smoothed, not unstable* (0.00–8.99% vs nature's 24.70%) → physics loss unmotivated for them. **`point_cube` 38.52%** is the sole target; diagnose before adding a loss term. |
| 9 Ensemble / RC-4 | 15 | **Pending — highest-value diagnostic left** | The only RC not saturated or slack. Models are **under-dispersed** (high PCs at 0.196× true std) → predict spread-error ratio **< 1**. |

### Merged from `nespreso-v2-port` (legacy ISAS appendix)

| Item | Status | Notes |
|------|--------|-------|
| GoM ML diagnostics | **done** | `scripts/gom_diagnostics.py` — ISAS + ARGO prod checkpoints |
| Results table | **done** | `scripts/results_table.py` — aggregates `saved/eval_*.json` |
| Decoder eval `loss: NaN` | **done** | `DecoderProfileLoss` uses `nanmean` on profile MSE |

ISAS production baseline (`patch16_scales`): T **1.016** / S **5.318**. Not the dissertation primary path.

---

## Key finding (splits)

GoM ARGO export spans **2015–2022** (4145 profiles), not 2002–2020. Candidate A (2002–2015 train) is **empty**.

**Default dissertation split:** chronological **70/15/15** (`config/argo/config_argo.json`, no `split_config`).  
**Explicit dates alternative:** `config/argo/config_argo_chrono_dates.json` (Candidate E).  
See `reports/split_design.md` for high/low observation stress subsets (2020 peak vs 2015–2018 sparse).

---

## L3 input architecture (MVP)

| Item | Value |
|------|--------|
| Variables | `ssh`, `wind_u`, `wind_v` (ERA5 u10/v10) |
| Features per var | `value`, `mask`, `age`, `uncertainty`, `count` |
| Patch grid | ±3° @ 0.25° → **25×25** (`spatial_pad=12`) |
| Time bins | 5 windows: 0, 24, 72, 168, 336 h (`temporal_pad=4`) |
| PatchConvMLP shape | `(C,T,H,W) = (15, 5, 25, 25)` |
| Input dim | **46881** = 6 encodings + 15×5×25×25 |
| Legacy ARGO point mode | `config/argo/config_argo.json` — still 9-D L4 COAPS inputs |

**Cache naming:** `data/cache/train_ready_l3_<config_hash>_<l3_hash>.pkl`  
**Processed samples:** `data/processed/l3_samples_<l3_hash>.pkl`

Without `data/raw/`, rasterization produces **explicit all-mask-zero** bundles (fast path when raw root missing). Download real SSH/ERA5 before expecting non-zero coverage.

---

## Commands

```bash
conda activate nespreso
cd NeSPReSO2_onTemplate

# Gate check (L3 synthetic + dataloader + v2 equivalence)
srun --ntasks=1 --cpus-per-task=8 python3 selfcheck.py

# Census + split reports
srun --ntasks=1 --cpus-per-task=8 python3 scripts/data_census.py -c config/argo/config_argo.json

# Legacy ARGO smoke (L4 point inputs, chronological)
srun --ntasks=1 --cpus-per-task=8 --gres=gpu:1 python3 train.py -c config/argo/config_argo_smoke.json

# L3 download (credentials: copernicusmarine login, ~/.cdsapirc for ERA5)
python3 scripts/download_l3_products.py --product all_scaffold
python3 scripts/download_l3_products.py --product ssh_l3_historical --date 2020-01-15
python3 scripts/download_l3_products.py --product era5_wind --year 2020 --month 1

# L3 processed samples + train cache
srun --ntasks=1 --cpus-per-task=8 python3 scripts/build_l3_samples.py \
  -c config/argo/config_argo_l3_smoke.json --max-samples 20 --export-train-cache --force
# Omit --max-samples for full 4145-profile cache (empty bundles if no raw data)

# L4 mask-augment processed batch (uses real L3 mask geometry on L4 SSH when raw present)
srun --ntasks=1 --cpus-per-task=8 python3 scripts/build_l3_samples.py \
  -c config/argo/config_argo_l3_l4_smoke.json --max-samples 20 --export-train-cache --force

# L3 mask-native smoke train (2 epochs)
srun --ntasks=1 --cpus-per-task=8 --gres=gpu:1 python3 train.py -c config/argo/config_argo_l3_smoke.json

# Stratified eval (needs trained checkpoint paired with its cache)
srun --ntasks=1 --cpus-per-task=8 --gres=gpu:1 python3 eval_stratified.py \
  -c config/argo/config_argo_l3_smoke.json -r saved/smoke_argo_l3/checkpoint-epoch2.pth \
  --split test --out saved/eval_stratified_l3_smoke.json --md-out saved/eval_stratified_l3_smoke.md

# Readiness diagnostics (static stability; needs checkpoint + cache)
srun --ntasks=1 --cpus-per-task=8 --gres=gpu:1 python3 diagnostics/readiness.py \
  -c config/argo/config_argo_l3_smoke.json -r saved/smoke_argo_l3/checkpoint-epoch2.pth \
  --split test --out saved/readiness_l3_smoke.json --md-out saved/readiness_l3_smoke.md
```

---

## File map (dissertation additions)

| Path | Role |
|------|------|
| `base/split_utils.py` | Chronological + random split indices |
| `scripts/data_census.py` | Profile census + split design reports |
| `scripts/download_l3_products.py` | Copernicus SSH, PO.DAAC, ERA5 download scaffold |
| `scripts/build_l3_samples.py` | CLI for L3 rasterization + optional train cache |
| `preproc/l3_rasterize.py` | Sparse obs → mask-native patch tensors |
| `preproc/l3_input.py` | Flatten bundles → PatchConvMLP sat block |
| `preproc/export_l3_cache.py` | Processed batch + `build_argo_l3_train_cache()` |
| `preproc/l4_augment.py` | L4 mask/noise augment + source flags (Phase 4) |
| `preproc/l4_rasterize.py` | DUACS L4 SSH → patch grid sampling |
| `config/argo/config_argo_l3_smoke.json` | L3 patch smoke (15-ch, chronological) |
| `config/argo/config_argo_l3_l4_smoke.json` | L3 + L4 mask-augment smoke (`io.l4.enabled`) |
| `config/argo/config_argo_chrono_dates.json` | Explicit date split alternative |
| `eval_stratified.py` | Stratified RMSE/bias by L3 coverage, track distance, census subsets |
| `diagnostics/readiness.py` | `gsw_torch` σ₀ static-stability readiness on predicted profiles |

---

## Eval rules

1. **Pair checkpoint with its cache** — never mix PCA/checkpoint across caches.
2. **Cross-tag:** `eval_matched.py` only; do not headline ISAS vs ARGO raw RMSE side by side.
3. **Dissertation:** chronological split only for ARGO results.
4. **L4** is augmentation/baseline — ARGO/CORA remains primary target.
5. **L3 cache** uses `dataset_tag: argo_v2_l3`; compare L3 runs against L3 cache, not legacy `argo_v2` 9-D cache.

---

## Known limitations

- SST (VIIRS L3U) and SMAP SSS rasterization **deferred**.
- L4 augment: SSH only (`mask_augment` mode); wind/SST L4 and `auxiliary` merge mode **deferred**.
- Spatially correlated L4 noise **deferred** (independent pixel noise today).
- Full-dataset L3 cache build without raw files is correct but **zero coverage** everywhere.
- `config/argo/config_argo_l3_smoke.json` sets `input_params.sss/sst/ssh/sat: false` — encodings only; sat block comes from L3 tensors.
- ISAS configs still use random split (legacy parity).

---

## Git / worktree

| Location | Branch | Notes |
|----------|--------|-------|
| `ISAS20_project/` | `master` | **active** — dissertation + merged ISAS appendix code |
| `ISAS20_project-phase3-commit/` | `phase3-l3-rasterization` | **delete** — redundant worktree; remove after confirming below |

`compare_v2_vs_template.ipynb` was executed successfully on **2026-06-29**; outputs saved to `NeSPReSO2_onTemplate/notebooks/_compare_v2_vs_template.ipynb`.

```bash
cd /unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project
```

Remove the old worktree when ready:

```bash
cd /unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project
git worktree remove ../ISAS20_project-phase3-commit
```

---

## Next coding tasks (priority order)

1. **[running]** `anom_point` loss-scale retrain — tmux `anom_retune`; then eval + readiness on the new checkpoint.
2. **Track A** — harden `scripts/bench_datacube_speed.py` into a loop-ready evaluator (`PLAN-agentic-ai-experiment.md`).
3. **`point_cube` σ₀ 38.5%** — diagnose vs `residual_cube` (2.57%); check cube-feature standardization. Gates Phase 8.
4. **RC-4 / MC-dropout ensemble** (Phase 9) — now the highest-value diagnostic; predict spread-error < 1.
5. Download real L3 SSH + L4 DUACS + ERA5 for 2020 window; rebuild caches with `--force`; verify non-zero mask cells.
6. SST L3U / SMAP (post-MVP).
7. ~~Steric SSH consistency (RC-2)~~ **done 2026-07-15** — wired + measured; **saturated**, not a usable objective.

---

## Session commits (this branch)

| Commit | Summary |
|--------|---------|
| `bb82a89` | Phases 0–3: census, chronological split, L3 rasterization |
| `b332666` | Phases 4–5 scaffold + L3 batch loading + PatchConvMLP 15-ch forward |
| `b03504e` | HANDOFF update for end of Phase 3–5 session |
