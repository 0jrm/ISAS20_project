# NeSPReSO redesign: anomaly targets, steric consistency, field-to-field U-Net

Project: `/unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project/NeSPReSO2_onTemplate` (conda env `nespreso`; all paths below relative to this unless absolute).

## Context

The model currently regresses **raw-profile PCs**, spending capacity reconstructing the seasonal cycle from harmonic inputs, and the comparison table has **no non-NN baseline** (no climatology, no GEM — verified: no LinearRegression/Ridge anywhere). The user wants a GraphCast-style reframing in three parts: (1) predict climatology **anomalies** + add climatology/GEM baselines, (2) a **steric-height consistency loss** tying predicted T/S to observed SSH, (3) a **field-to-field U-Net** over the full GoM grid with masked sparse supervision at ARGO pixels, making point/patch the degenerate windowed versions. User was asked about scope and field-model channels but was AFK — proceeding with the recommended defaults: **all three phases A→B→C**, **SSH-only primary field model** (revisit if user objects).

Key verified facts:
- PCA is fit per variable on `nan_to_num(profiles,0)` over the **entire dataset** (leakage — fix by fitting on train split only, transform all). Builders: `preproc/preproc_isas_sat.py:416-530`, `preproc/export_v2_cache.py:36-125`, `preproc/export_argo_l4_cache.py:56-211`.
- `config_hash` (`preproc_isas_sat.py:186-195`) covers the `io` section → new io keys ⇒ fresh cache filename ⇒ checkpoint/cache pairing preserved by construction.
- `gsw_torch` (differentiable) is installed: `specvol_anom_standard`, `SA_from_SP`, `CT_from_t`. Working batched usage: `diagnostics/readiness.py:105-113`; unwired `steric_ssh_diagnostic` stub at `readiness.py:159`.
- SSH input is **absolute ADT** (argo_v2 col 8; argo_l4 col 522). `gom_mean_adt_2013_2022.nc` is a basin-mean **time series** (3652 scalars), NOT a spatial map — get SLA directly from the DUACS L4 files (`/unity/g2/jmiranda/SubsurfaceFields/Data/CMEMS/SSH/SSH_YYYY.nc`, vars `adt, sla, ugos, vgos`, 0.25°).
- ISAS `Original/*.nc` are per-profile (not gridded) 2002–2020 analyses with per-profile `*_CLMN` climatology — **not usable** as-is for the 2015–2022 ARGO targets ⇒ fit our own harmonic climatology.
- Gridded L4 on disk at `/unity/g2/jmiranda/SubsurfaceFields/Data/`: DUACS SSH 1999–2020+, OSTIA SST →2021-01, CMEMS SSS →2020-12, NBS wind, GEBCO 2024. **SST/SSS end before the chronological val/test window (~mid-2021→2022-02).** Regridder exists: `ISAS20_project/utils/retrieve_sat.py` (import, don't rewrite).
- GoM domain 18–31N / −98..−81 (`config_argo_patch_l4.json io.basin`) → 52×68 @ 0.25°; 4145 profiles ≈ 1.2/cell total ⇒ masked sparse supervision essential.
- Trainer batch contract `(data, target, indices)` hard-coded in `trainer/trainer.py:151-154,187`; `BaseTrainer.train` (`base/base_trainer.py:119`) is shape-agnostic. Loss forward: `(pcs, targets, indices=None, inputs=None)`; `DensityConstraint` appended in `CombinedPCALoss.forward:518-538` is the physics-penalty template. `make_loss:541-609` wired from `train.py:198-213`.
- Baseline to beat / match: point model test avg RMSE 0.301 (T 0.516 / S 0.087), chronological 70/15/15 (train 2901 / val 621 / test 623).

**Comparability break (intentional):** anomaly targets + train-only PCA make all prior checkpoints historical-only. Keep old caches/JSONs; annotate in results table. Never mutate an existing cache (HANDOFF pairing rule) — every new payload key ships with an `io.cache_version` bump.

---

## Phase A — Anomaly targets + climatology & GEM baselines

### A1. New `preproc/climatology.py`
Per-variable (T,S), per-depth ridge regression on a 30-term tensor-product basis:
- time: `[1, cos/sin annual, cos/sin semiannual]` of day-of-year from JULD (5 terms)
- space: degree-2 polynomial in (lat,lon) normalized to [−1,1] over the GoM box (6 terms)

Design matrix (n_train×30) shared across depths → loop of 1801 ridge solves (30×30), milliseconds; per-depth valid mask for short profiles; optional ~5–11 m vertical smoothing of coefficients. Polynomial (not RBF) because train profiles cluster in the central GoM — RBF would extrapolate wildly onto shelves.

Picklable API (plain dataclass + numpy, no closures):
```python
@dataclass
class Climatology:
    coef: dict[str, np.ndarray]  # var -> (30, n_z)
    pres: np.ndarray; norm: dict; meta: dict
def fit_climatology(profiles, lat, lon, juld, pres, train_idx, cfg) -> Climatology
def eval_climatology(clim, lat, lon, juld) -> dict[str, np.ndarray]  # var -> (n_z, N)
```
**Fit on TRAIN split only** — builders compute the chronological split internally via `base/split_utils.build_split_indices` (deterministic from dl_args + JULD). Fix the PCA leakage at the same time: fit PCA on train anomalies, transform all.

### A2. Cache builders (all three, gated identically)
Gate `io.anomaly_targets: true` (+ `io.cache_version: 2`) → hash changes → fresh cache. When enabled: split → fit clim → `anomalies = nan_to_num(profiles) − clim_profiles` → PCA on train anomalies → targets = anomaly PCs. New payload keys: `climatology`, `clim_profiles` (var → (n_z,N) f32), `anomaly_targets: True`. Extract the shared block into one helper in `preproc/climatology.py` (no triplication).

### A3. Single add-back helper
In `model/loss.py` next to `sklearn_inverse_transform_pcs:47`:
```python
def reconstruct_physical_profiles(pcs, pca_models, outputs, clim_profiles=None, indices=None)
```
plus a torch path: register `clim_profiles` buffers in `CombinedPCALoss` (alongside true-profile buffers, lines 291-300), indexed by `indices`. Call sites (complete): `loss.py _reconstruct_profiles:492-516` (profile-MSE term stays in anomaly space — mathematically identical; add clim back only where physics needs physical profiles: density term 529-536 and Phase B steric), `model/metric.py:20-42`, `eval_run.py:111-118`, `eval_matched.py:83`, `diagnostics/readiness.py` (~105-150 + steric diag). `make_loss` and `train.py:198-213` gain `clim_profiles=`.

### A4. SSH caching (needed by GEM and Phase B)
All three builders store per-sample `cache["ssh_obs_adt"]` and `cache["ssh_obs_sla"]` — **SLA sampled directly from the DUACS `sla` variable** at (lat,lon,date) from `CMEMS/SSH/SSH_YYYY.nc` (do NOT parse input columns; do NOT use gom_mean_adt file, it's a time series). Also fix the known off-by-4 in `surface_residual_feature_col` (`preproc_isas_sat.py:40-59`) to offset by `count_scalar_dims:92`.

### A5. Baselines — new `scripts/eval_baselines.py`
Loads the anomaly cache, recomputes the split, evaluates on test, interpolates to the same 0–1800 m @ 10 m grid as `eval_run.py`, writes JSON matching the `results_table.py collect_eval_rows` schema (`checkpoint, cache, dataset_tag, split, n_samples, raw_profile_rmse{temperature,salinity}`):
1. **Climatology-only**: prediction = `clim_profiles` → `saved/eval_clim_test.json`.
2. **Clim + SLA GEM**: per depth & var, train-fit regression `anomaly(z) = a(z)·SLA + b(z)` using `cache["ssh_obs_sla"]` → `saved/eval_gem_test.json`.
Add labels to `EVAL_LABELS` in `scripts/results_table.py`.

### A6. Selfcheck + configs
New flat `test_*` in `selfcheck.py` (register in `__main__` ~1161-1218): `test_climatology_fit_eval_roundtrip`, `test_climatology_train_only` (poisoned val sample has no influence), `test_anomaly_cache_addback` (add-back ≈ nan_to_num(profiles) within PCA truncation), `test_ssh_obs_cached`. New configs: `config/argo/config_argo_anom.json`, `config_argo_patch_l4_anom.json` + smoke variants. Check anomaly-PCA explained variance at build; adjust `io.outputs` PC counts if needed before training.

### A7. Verify
```
python3 selfcheck.py
srun --ntasks=1 --cpus-per-task=8 --gres=gpu:1 python3 train.py -c config/argo/config_argo_smoke.json   # non-anomaly path unchanged
srun --ntasks=1 --cpus-per-task=8 --gres=gpu:1 python3 train.py -c config/argo/config_argo_anom.json
python3 eval_run.py <ckpt> --split test ; python3 scripts/eval_baselines.py -c config/argo/config_argo_anom.json
python3 scripts/results_table.py
```
Success: anomaly point model ≥ parity with 0.301 avg RMSE; table shows climatology-only + GEM rows; every model row readable as skill-vs-climatology. Repeat for patch config.

---

## Phase B — Steric-height consistency loss

### B1. Physics function (in `model/loss.py`)
```python
def steric_height_anomaly(temp, sal, pres_dbar, lat, lon, subsample_dz=5) -> torch.Tensor  # (B,)
# SA_from_SP → CT_from_t → specvol_anom_standard → (1/g)·trapz(delta, dp[Pa])
```
Batched conventions from `readiness.py:105-113`. 0–1800 dbar integral = dynamic height rel. 1800 dbar ⇒ **anomaly-vs-anomaly only**.

### B2. Reference + calibration (cache-build time)
- `cache["clim_steric"]`: steric height of climatological profile at each sample.
- `cache["steric_calibration"]`: train-split affine fit `(alpha, beta)` of observed SLA on `(steric(true) − clim_steric)` — absorbs deep steric, barotropic, DUACS offsets.
Loss term: `scale · mean((α·(steric(pred_phys) − clim_steric[idx]) + β − ssh_obs_sla[idx])²)`. All cache-indexed via `indices` — no input-column dependence.

### B3. Wiring (mirror DensityConstraint)
`class StericConstraint(nn.Module)` with buffers (ssh_obs_sla, clim_steric, lat, lon, pres, calibration); instantiate in `CombinedPCALoss` when `loss.steric.enabled`; append term after `_reconstruct_profiles` (physical path from A3) next to density (529-536), logged as separate component. Thread `steric_config` + arrays through `make_loss` / `train.py:198-213`. Config: `"steric": {"enabled": true, "scale": <tune>, "subsample_dz": 5}` — identical for point and patch. Wire the real diagnostic into `readiness.py steric_ssh_diagnostic:159` (no-grad; report corr + RMSE of calibrated steric vs SLA).

### B4. Selfcheck
`test_steric_height_sanity` (warm/salty > cold/fresh; O(0.1–1 m)), `test_steric_matches_climatology_adt` (clim steric field vs DUACS mean pattern, r > ~0.5), `test_steric_loss_grad` (finite grads to PCs, both tags), `test_steric_train_calibration` (**hard gate**: train-profile steric vs SLA r > ~0.6, else the loss is mis-specified — don't train with it).

### B5. Verify
Smoke + full runs of point and patch anomaly configs at 2–3 `scale` values (one tiny as control). Success: test RMSE not degraded vs Phase A (within noise) AND steric-consistency diagnostic on test improves vs no-steric run. Report both.

---

## Phase C — Field-to-field U-Net (SSH-only primary)

### C1. Cache builder — new `preproc/export_field_cache.py`
- Domain 18–31N / −98..−81 @ 0.25° → 52×68. Dates = unique ARGO profile dates 2015–2022 (~1000–1500). Regrid via `utils/retrieve_sat.py`.
- **Primary channels** (full-period coverage): SLA (DUACS), static log-bathymetry (GEBCO, normalized), land/sea mask; time harmonics (4 cos/sin channels) computed in the Dataset from JULD, not stored. SST/SSS-augmented config = train-period-only **ablation** (per-channel availability masks); extending downloads = optional follow-up.
- Targets: **load the finalized Phase A argo_v2 anomaly cache** and carry over `pca_models, climatology, clim_profiles, outputs, weights, targets, profiles, PRES, LAT/LON, JULD, ssh_obs_sla, clim_steric` — identical targets for point-vs-field comparison; loss/eval/steric work unchanged. Add `fields (D,C,52,68) f32` (~60 MB), `dates`, `sample_date_idx (N,)`, `sample_pixel_rc (N,2)`. `dataset_tag: "argo_field"`, own config_hash.

### C2. Arch — `model/model.py`
`class FieldUNet(BaseModel)`: 2-level U-Net, base width ~32, GroupNorm+SiLU, `in_channels`/`out_channels=Σk PCs` from config; pad 52×68→56×72 in forward, crop back. ~1–2 M params — small is a feature (1500 images, 4145 supervision pixels).

### C3. Wiring
- `preproc/l3_input.py sync_arch_with_io:134-186`: 4th branch for `FieldUNet` — set `in_channels`/`out_channels`, do NOT set `input_dim`.
- `train.py ensure_cache:37-85` + `resolve_dataloader_batch_size:98-145`: branch on `dataset_tag=="argo_field"` (assert `fields.ndim==4`, channel count; fixed batch_size, skip VRAM probe).
- `data_loader/data_loaders.py`: `FieldDataset.__getitem__(d)` → `(field+time-channels (C,H,W), pc_targets (P_d,Σk), pixel_rc (P_d,2), sample_idx (P_d,))`; `collate_field` stacks fields + concatenates ragged pixel lists with `batch_of_pixel` index. `FieldDataLoader` exposes the attribute surface eval/metrics read (`pca_models, outputs, weights, LAT, LON, PRES, profiles, split_validation(), split_test()`). **Split chronologically by date** (a date lives in exactly one split — no leakage through shared fields); report n_dates + n_profiles per split.
- New `trainer/field_trainer.py`: `FieldTrainer(BaseTrainer)` overriding `_train_epoch/_valid_epoch/_forward_loss/_loss`: `out = model(fields)` → gather `pred_pcs = out[batch_of_pixel,:,r,c]` → existing `CombinedPCALoss(pred_pcs, targets, indices=sample_idx)` — anomaly add-back, density, steric all free via indices. Dispatch in `train.py:246` on `dataset_tag`.

### C4. Eval + deliverable
- `eval_field.py`: test dates → gather at pixels → `reconstruct_physical_profiles` → 10 m grid RMSE vs `cache["profiles"]` → `saved/eval_field_test.json` (results_table schema + label) ⇒ automatic climatology/GEM/point/patch/field table.
- `scripts/export_field_product.py`: run over a date range, inverse-PCA every ocean pixel, add climatology at (pixel, doy) → gridded 3D T/S netCDF (dissertation deliverable). Mask land/shallow pixels.

### C5. Selfcheck + smoke
`test_field_unet_shapes` (pad/crop round-trip), `test_field_gather_matches_loop`, `test_field_date_split_disjoint`, `test_field_cache_targets_match_v2`, `test_field_loss_grad_with_steric`. Configs: `config/argo/config_argo_field_smoke.json` (few dates, 2 epochs) + `config_argo_field.json`.

Success: trains stably; masked test RMSE in the comparison table. Beating the point model is NOT required — an honest point-vs-patch-vs-field comparison is the criterion.

---

## Ordering constraints & risks
- Strict A→B→C: B's calibration needs A's climatology; C1 carries A's finalized argo_v2 anomaly cache (rebuilding it later invalidates the field cache).
- Within A: A4 (ssh_obs) lands with A2 (single cache-version bump); A5 baselines need `ssh_obs_sla`.
- Anomaly PC spectrum flatter than raw — check explained variance, may need more PCs.
- DUACS SLA is smoothed — pointwise steric corr may cap ~0.7–0.8; keep `scale` small initially; hard calibration gate protects.
- Field model data hunger: small U-Net + early stopping; no flips (break geographic priors).
- Long GPU runs: use tmux/srun per HANDOFF pattern.

---

## Operational handoff (2026-07-02)

**Env:** always `conda activate nespreso` (or `/conda/jmiranda/miniconda/envs/nespreso/bin/python3`). GSW usage is **`import gsw_torch as gsw`** everywhere in training/diagnostics (`model/steric.py`, `diagnostics/readiness.py`); selfcheck compares vectorized vs looped `gsw_torch` only (no vanilla `gsw` dependency).

### Progress

| Phase | Status | Key artifacts |
|-------|--------|---------------|
| **A** | **~90% code-complete** | `preproc/climatology.py` (`fit_climatology`, `eval_climatology`, `build_anomaly_targets_block`); `preproc/ssh_obs.py`; anomaly gating in `export_v2_cache.py`, `export_argo_l4_cache.py`, `preproc_isas_sat.py`; `reconstruct_physical_profiles` + clim buffers in `CombinedPCALoss`; `scripts/eval_baselines.py`; configs `config_argo_anom.json`, `config_argo_anom_smoke.json`, `config_argo_patch_l4_anom.json`; `surface_residual_feature_col` offset via `count_scalar_dims` |
| **B** | **~80% code-complete** | `model/steric.py` (`steric_height_anomaly`, `StericConstraint`, train calibration); wired in `make_loss` / `train.py` / `eval_run.py`; `steric_ssh_diagnostic` implemented in `readiness.py` |
| **C** | **~70% code-complete** | `FieldUNet` in `model/model.py`; `preproc/export_field_cache.py`; `FieldDataset`/`FieldDataLoader`; `trainer/field_trainer.py`; `eval_field.py`; `scripts/export_field_product.py`; `train.py` dispatch for `argo_field` |

**Selfcheck (new tests, pass individually under `nespreso`):** `test_climatology_*`, `test_anomaly_cache_addback`, `test_ssh_obs_cached_smoke`, `test_steric_height_sanity`, `test_steric_loss_grad`, `test_field_unet_shapes`. Full `python3 selfcheck.py` not confirmed end-to-end (slow tests / pipe buffering — run with `PYTHONUNBUFFERED=1 python3 -u selfcheck.py`).

**Not run yet:** production anomaly-cache build (DUACS SSH sampling ~4k profiles), GPU training on `config_argo_anom.json`, baseline JSONs, steric ablation sweeps, field-cache build + field training.

### Next steps (priority order)

1. **Build anomaly cache** (one-time, slow):  
   `/conda/jmiranda/miniconda/envs/nespreso/bin/python3 -c` or `train.py` with `-c config/argo/config_argo_anom_smoke.json` first (`max_samples` optional). Confirm printed anomaly-PCA explained variance; bump `outputs` PC counts if variance is low.
2. **A7 verify:** smoke train non-anomaly (`config_argo_smoke.json`) unchanged → full `config_argo_anom.json` train → `eval_run.py` + `scripts/eval_baselines.py` → `scripts/results_table.py`. Repeat for patch (`config_argo_patch_l4_anom.json`).
3. **Close A gaps:** `eval_matched.py` still uses raw `sklearn_inverse_transform_pcs` (no clim add-back); add `config_argo_patch_l4_anom_smoke.json`; optional 0–1800 m @ 10 m interp in `eval_baselines.py` (currently native-depth RMSE).
4. **B hard gates:** add `test_steric_train_calibration` (train steric vs SLA r ≳ 0.6) before enabling `steric.enabled: true`; tune `steric.scale` (start 0.01).
5. **C configs + cache:** add `config_argo_field.json` / `config_argo_field_smoke.json` pointing at `io.field.source_cache` = finalized Phase-A `train_ready_<hash>.pkl`; `build_field_cache` → smoke train → `eval_field.py`.
6. **C selfcheck:** `test_field_gather_matches_loop`, `test_field_date_split_disjoint`, `test_field_cache_targets_match_v2`, `test_field_loss_grad_with_steric`.
7. **`parse_config.validate_config`:** assert steric cache keys when `steric.enabled`; skip `input_dim` check for `FieldUNet`.

### Intentional compromises / tradeoffs

- **Climatology basis:** degree-2 polynomial in normalized (lat,lon) × annual/semiannual harmonics (30 terms), ridge per depth — not RBF (shelf extrapolation) and not ISAS `*_CLMN` (wrong period/coverage).
- **Train-only fit:** climatology + anomaly PCA + steric calibration all use chronological train split inside `build_anomaly_targets_block`; old raw-PCA caches remain valid historical artifacts (`io.cache_version: 2` forces new filenames).
- **Steric physics:** `gsw_torch.specvol_anom_standard(SA,CT,p)` (3-arg API) + trapezoidal ∫α dp / g; no explicit 1800 dbar reference argument in gsw_torch — calibration `(α,β)` on SLA absorbs barotropic/deep/mean offsets. Gradients through steric can NaN for extreme random PCs; loss test uses near-zero PCs / asserts `any` finite grad, not `all`.
- **SSH obs:** sampled at profile (lat,lon,date) from DUACS via `retrieve_satellite_data` (`preproc/ssh_obs.py`), not from model input columns; batch build is I/O heavy (`io.ssh_batch_size`, default 200).
- **Field model scope:** SSH + log-bathy + land mask in cache; 4 time harmonics added in `FieldDataset` (not stored). Small 2-level U-Net (~32 base width); date-level chronological split (not profile-level). Beating point model is explicitly out of scope — comparison table honesty is the goal.
- **Steric module location:** lives in `model/steric.py` (not inline in `loss.py`) to keep `CombinedPCALoss` readable.
- **export_v2_cache:** when `anomaly_targets` is false, behavior unchanged; when true, legacy PCA block still runs then gets overwritten (harmless redundancy, not yet refactored away).

### Quick commands

```bash
cd ISAS20_project/NeSPReSO2_onTemplate
conda activate nespreso   # required

# Unit tests (fast)
PYTHONUNBUFFERED=1 python3 -u selfcheck.py   # or run new tests only — see selfcheck __main__

# Phase A smoke
srun --ntasks=1 --cpus-per-task=8 --gres=gpu:1 python3 train.py -c config/argo/config_argo_anom_smoke.json

# Baselines (after anomaly cache exists)
python3 scripts/eval_baselines.py -c config/argo/config_argo_anom.json
python3 scripts/results_table.py
```

**Checkpoint/cache rule (unchanged):** never mutate an existing `train_ready_*.pkl`; new `io` keys ⇒ new hash. Field cache must be rebuilt if its source anomaly cache is regenerated.
