# Handoff — ARGO L4 patch model: data, training, evaluation (2026-07-04)

**Prior investigation:** [HANDOFF-2026-07-03-l4-stale-sat.md](HANDOFF-2026-07-03-l4-stale-sat.md) (stale satellite root cause, downloads, regeneration).

**Status as of 2026-07-04:** Satellite data fixed and caches rebuilt. Full retrains completed (`patch_l4_fixedsat`, `patch_l4_anom_fixedsat`). **Patch models still do not beat the reference point baseline or climatology on test.** Next lever: input standardization (see §7).

All paths below are under `/unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project` unless noted. Working directory for train/eval: `NeSPReSO2_onTemplate/`. Conda env: **`nespreso`**.

---

## 1. Reference benchmark (target to beat)

The L4 patch model is compared against the **ARGO point model** (NeSPReSO v2 / PCA-argo production run):

| Item | Value |
|------|-------|
| Config | `NeSPReSO2_onTemplate/config/argo/config_argo.json` |
| Checkpoint | `NeSPReSO2_onTemplate/saved/models/NeSPReSO2_ARGO_GoM/argo16_scales/model_best.pth` |
| Architecture | `PatchConvMLP` (9-D point inputs: harmonics + local sss/sst/ssh) |
| Test split | Chronological 15% (~623 profiles) |
| **Golden test RMSE (native depth)** | **T = 0.416 °C**, **S = 0.072 PSU** |
| Source | `NeSPReSO2_onTemplate/saved/eval_argo16_test.json` |

Secondary baselines on the same test split:

| Baseline | T RMSE | S RMSE | Source |
|----------|-------:|-------:|--------|
| Climatology-only | 1.657 | 0.216 | `saved/eval_clim_test.json` |
| ANOM-point | 0.647 | 0.100 | `saved/eval_anom_point_test.json` |

**Success criterion for patch work:** beat climatology on test first, then close the gap to the point model. Patch should justify its extra spatial context vs a 9-D point encoder.

---

## 2. Model and data

### 2.1 What the L4 patch model is

- **Architecture:** `PatchMaskConvMLP` (`NeSPReSO2_onTemplate/model/model.py`)
- **Configs:** `config/argo/config_argo_patch_l4.json` (raw profile PCs), `config/argo/config_argo_patch_l4_anom.json` (anomaly PCs + clim add-back at eval)
- **Scalar branch:** 10-D → `Linear(10 → 128)` (`n_enc=10`, `d_model=128`)
- **Patch branch:** 3 satellite channels × 7 days × 5×5 grid → `Conv3d [32,64]` + `GroupNorm` → `AdaptiveAvgPool3d(1,1,1)` → `Linear → 128`
- **Fusion:** concat → MLP head (`512` hidden, depth 2, dropout 0.2) → **32 outputs** (16 T PCs + 16 S PCs)
- **Loss:** `combined_pca_loss`, mode `pc_mse_only`
- **Mask mode (production):** `use_mask_channels: false` (NaN→0 in preproc; 535-D). With `use_mask_channels: true`, input becomes 1060-D (value+mask interleaved).

### 2.2 Input vector (535-D)

Built by `preproc/preproc_isas_sat.py::build_argo_l4_input_matrix`, cached by `preproc/export_argo_l4_cache.py`.

| Block | Dims | Notes |
|-------|-----:|-------|
| Temporal harmonics | 2 | `timecos`, `timesin` on `JULD % 365` |
| Spatial harmonics | 4 | `latcos`, `latsin`, `loncos`, `lonsin` |
| Basin daily means | 3 | `basin_sss`, `basin_sst` (K→°C), `basin_ssh` |
| Bathymetry | 1 | positive depth (m) at patch center |
| **Scalar subtotal** | **10** | cols 0–9 |
| SSS patch | 175 | CMEMS, 5×5×7 flattened |
| SST patch | 175 | OSTIA, K→°C |
| SSH patch | 175 | CMEMS ADT |
| **Satellite subtotal** | **525** | cols 10–534 |
| **Total** | **535** | `patch_shape: [3, 7, 5, 5]` = (C, T, H, W) |

**Patch geometry:** `spatial_pad: 2` → 5×5 cells; `temporal_pad: 6` → 7 daily steps ending on profile date. Flatten order: `time_major_row_major_lat_lon`.

**Satellite HDF5:** `data/NeSPReSO_v2_ARGO_GoM_sat/satellite_NeSPReSO_v2_ARGO_GoM.h5` — one row per profile, aligned with v2 pickle station order.

### 2.3 Output labels / targets

| Config | Target | Reconstruction at eval |
|--------|--------|------------------------|
| `config_argo_patch_l4.json` | Raw profile **PCA** (16 T + 16 S) from v2 pickle; `refit_pca: true` if dim mismatch | Inverse PCA only |
| `config_argo_patch_l4_anom.json` | **Anomaly PCA** (`io.anomaly_targets: true`, `cache_version: 2`): climatology fit on train split, PCA on train anomalies | Inverse PCA + add `clim_profiles` |

Ground truth for RMSE: physical T/S profiles from v2 pickle (`ds.TEMP`, `ds.SAL`), depth axis `PRES`.

### 2.4 Dataset

| Field | Value |
|-------|-------|
| Tag | `io.dataset_tag: argo_l4` |
| Profiles | **4145** from `io.v2_pickle` (`config_dataset_full.pkl`) |
| Domain | GoM 18–31°N, −98 to −81°W; basin exclusion (23°N, −88°W) |
| Split | Chronological 70/15/15 → ~2901 / 621 / **623** test |
| Raw cache | `data/cache/train_ready_4411c65ee518.pkl` |
| Anom cache | `data/cache/train_ready_0085aed3c9b8.pkl` |

Cache hash = SHA-256 of `{input_params, io, outputs, use_mask_channels}` (first 12 hex chars). Raw and anom configs produce different hashes.

---

## 3. Preprocessing

### 3.1 Pipeline overview

```
v2 pickle (4145 profiles)
    → generate_argo_satellite_data.py (batch HDF5s → merged satellite HDF5)
    → export_argo_l4_cache.py (train_ready_<hash>.pkl)
    → train.py
```

### 3.2 Satellite HDF5 generation

| Item | Path / command |
|------|----------------|
| Script | `utils/generate_argo_satellite_data.py` |
| CWD | `ISAS20_project/utils` |
| Batches | `data/NeSPReSO_v2_ARGO_GoM_sat/argo_sat_batches/argo_sat_batch_b0100_*.h5` |
| Merged | `data/NeSPReSO_v2_ARGO_GoM_sat/satellite_NeSPReSO_v2_ARGO_GoM.h5` |
| Resume | Skips existing batches; deletes stale batches by date before regen (see prior handoff) |

```bash
cd ISAS20_project/utils
srun --ntasks=1 --cpus-per-task=8 python3 generate_argo_satellite_data.py \
  -c ../NeSPReSO2_onTemplate/config/argo/config_argo_patch_l4.json --batch-size 100
```

### 3.3 Satellite sources

| Product | Archive | HDF5 group/var |
|---------|---------|----------------|
| OSTIA SST | `Data/OISST/OSTIA` | `ostia/analysed_sst` |
| CMEMS SSS | `Data/CMEMS/SSS` | `sss/sos` |
| CMEMS SSH | `Data/CMEMS/SSH` (`SSH_YYYY.nc`) | `ssh/adt` |
| Bathymetry | `Data/Bathymetry` | `bathymetry/elevation` |

Selection: `utils/retrieve_sat.py::select_candidate_file` (nearest file, **no date tolerance** — guard not yet implemented). Coverage needed through **2022-02-27** + temporal pad.

### 3.4 Cache export

```bash
cd NeSPReSO2_onTemplate
srun --ntasks=1 --cpus-per-task=8 python3 preproc/export_argo_l4_cache.py \
  -c config/argo/config_argo_patch_l4.json --force
srun --ntasks=1 --cpus-per-task=8 python3 preproc/export_argo_l4_cache.py \
  -c config/argo/config_argo_patch_l4_anom.json --force
```

`train.py` calls `build_argo_l4_cache` automatically if the hashed pickle is missing; use `--force` after satellite or config changes.

### 3.5 Data quality checks

| Script | Purpose | Healthy result |
|--------|---------|----------------|
| `diagnostics/stale_sat/h5_stale_check.py` | Fraction of stations whose 7-day patch is **identical at every time step** | **0.00** all months (post-fix) |
| `diagnostics/stale_sat/split_vs_stale.py` | Stale fraction per train/val/test split | ~0 on val/test after fix |

**Date pitfall:** cache `JULD` is MATLAB datenum (+366 day offset vs truth). Use HDF5 `stations/julian_date` (astropy JD) for profile dates.

### 3.6 Input standardization (not in production yet)

| Cache | Standardized? |
|-------|---------------|
| Production `train_ready_4411c65ee518.pkl` | **No** — raw scales (bathy ~2700, SSS ~36, SST ~25, SSH ~0.4) |
| Diagnostic `diagnostics/stale_sat/train_ready_e2_std.pkl` | **Yes** — z-score from train split (`make_std_cache.py`) |

E2 diagnostic retrain with std cache: train `profile_rmse` **0.648 → 0.150** (`diag_e2_bs512_std`). **Not yet baked into `export_argo_l4_cache.py`.**

---

## 4. Training

### 4.1 Entry point and flags

**Script:** `NeSPReSO2_onTemplate/train.py`

| Flag | Purpose |
|------|---------|
| `-c` | Config JSON |
| `-id` | Run subdirectory under `saved/models/<name>/` |
| `--bs 512` | Override batch size (recommended; config default `0` = auto full-batch) |
| `-r` | Resume checkpoint |
| `-d` | GPU index |

### 4.2 Trainer settings (production configs)

| Setting | Value |
|---------|-------|
| `epochs` | 8000 |
| `early_stop` | 500 (patience on `min val_loss`) |
| `monitor` | `min val_loss` |
| Optimizer | Adam, lr 0.001 |

### 4.3 Commands (fixed-sat retrains, 2026-07-04)

```bash
cd NeSPReSO2_onTemplate
export CUDA_VISIBLE_DEVICES=0   # or srun --gres=gpu:1

python train.py -c config/argo/config_argo_patch_l4.json --bs 512 -id patch_l4_fixedsat
python train.py -c config/argo/config_argo_patch_l4_anom.json --bs 512 -id patch_l4_anom_fixedsat
```

Pipeline script used: `utils/run_patch_fixedsat_gpu0.sh` (sequential raw → anom → eval on GPU 0).

### 4.4 Checkpoints (2026-07-04 runs)

| Run | Path | Epochs | Early-stop reason |
|-----|------|-------:|-------------------|
| Raw fixed-sat | `saved/models/NeSPReSO2_ARGO_GoM_patch_l4/patch_l4_fixedsat/model_best.pth` | 510 | no val improvement ×500 |
| Anom fixed-sat | `saved/models/NeSPReSO2_ARGO_GoM_patch_l4_anom/patch_l4_anom_fixedsat/model_best.pth` | 524 | same |

Best val checkpoint selected at **~epoch 10** (val `profile_rmse` plateau: raw **0.693**, anom **0.635**). TensorBoard: `saved/log/NeSPReSO2_ARGO_GoM_patch_l4/<run_id>/`.

---

## 5. Evaluation

### 5.1 Primary: `eval_run.py`

Test-split PCA loss + native-depth `raw_profile_rmse` vs physical profiles.

```bash
cd NeSPReSO2_onTemplate

python eval_run.py \
  -c config/argo/config_argo_patch_l4.json \
  -r saved/models/NeSPReSO2_ARGO_GoM_patch_l4/patch_l4_fixedsat/model_best.pth \
  --split test --out saved/eval_patch_l4_fixedsat_test.json

python eval_run.py \
  -c config/argo/config_argo_patch_l4_anom.json \
  -r saved/models/NeSPReSO2_ARGO_GoM_patch_l4_anom/patch_l4_anom_fixedsat/model_best.pth \
  --split test --out saved/eval_anom_patch_fixedsat_test.json
```

**Always pass explicit `-r` paths** — do not rely on auto-discovery (see §5.4).

### 5.2 Comparison and diagnostics

| Script | Output | Notes |
|--------|--------|-------|
| `notebooks/run_argo_production_compare.py` | `notebooks/compare_outputs/argo_production_results.json`, depth overlay PNG | Point vs patch; uses `discover_checkpoint` |
| `notebooks/compare_outputs/fixedsat_eval_report.json` | Explicit fixed-sat vs stale vs golden table | Generated 2026-07-04 |
| `notebooks/compare_outputs/fixedsat_vs_golden_depth_rmse.png` | Depth-RMSE curves | Point vs fixed-sat vs stale ckpt |
| `notebooks/compare_outputs/fixedsat_bin_maps.png` | 1° bin RMSE maps | Point vs fixed-sat L4 |
| `scripts/gom_diagnostics.py` | `saved/gom_diagnostics/` depth plots + bin maps | `--keys argo_point argo_patch_l4` |
| `scripts/results_table.py` | `saved/results/eval_table.{json,md}` | Aggregates all `saved/eval_*.json` |
| `scripts/eval_baselines.py` | Clim/GEM baselines (anom configs) | |

### 5.3 Stale-sat diagnostic suite

`NeSPReSO2_onTemplate/diagnostics/stale_sat/`:

| Script | Purpose |
|--------|---------|
| `h5_stale_check.py` | Time-constant patch fingerprint |
| `split_vs_stale.py` | Stale fraction per split |
| `cmp_sat_sources.py` | L4 vs point-cache satellite correlation |
| `diag_patch.py` | Per-split RMSE, ablations |
| `e0_point_equiv.py` | MLP on point-equivalent features |
| `make_std_cache.py` | Build z-scored cache for E2 experiment |

### 5.4 Gotcha: `discover_checkpoint` epoch ranking

`notebooks/nb_checkpoints.py::discover_checkpoint()` picks `max(model_best.pth)` by **(stored epoch, mtime)**, not best test RMSE or recency of run.

- Stale-era run `0701_102436` (epoch 252 in ckpt) can be auto-selected over `patch_l4_fixedsat` (best ckpt from epoch ~10).
- Stale checkpoint evaluated on **new cache** can look artificially better than a fixed-sat retrain.
- **Mitigation:** always use explicit `-r` in `eval_run.py`; pin checkpoints in scripts/notebooks.

---

## 6. Current results (2026-07-04)

Fixed-satellite pipeline complete; caches `4411c65ee518` (raw) and `0085aed3c9b8` (anom); retrains with `--bs 512`.

### 6.1 Test RMSE (native depth, n=623)

| Model | T RMSE | S RMSE | vs clim (1.657 T) | vs golden point (0.416 T) |
|-------|-------:|-------:|-------------------|----------------------------|
| **Golden ARGO-point** | **0.416** | 0.072 | ✓ | baseline |
| Climatology | 1.657 | 0.216 | — | ✗ |
| ANOM-point | 0.647 | 0.100 | ✓ | ✗ |
| **L4 raw fixed-sat (NEW)** | **1.857** | 0.231 | ✗ worse | ✗ |
| **L4 anom fixed-sat (NEW)** | **1.748** | 0.225 | ✗ worse | ✗ |
| L4 raw stale ckpt → NEW cache | 1.395 | 0.198 | ✓ | ✗ |
| L4 anom stale ckpt → NEW cache | 1.035 | 0.151 | ✓ | ✗ |

Common-depth avg RMSE (0–1800 m @ 10 m): point **0.301**, L4 fixed-sat **1.045**, stale ckpt on new cache **0.799**.

Sources: `saved/eval_patch_l4_fixedsat_test.json`, `saved/eval_anom_patch_fixedsat_test.json`, `saved/eval_argo16_test.json`, `notebooks/compare_outputs/fixedsat_eval_report.json`.

### 6.2 Training behavior (fixed-sat retrains)

- Val `profile_rmse` best by **epoch ~10** (raw 0.693, anom 0.635); flat for 500 epochs → early stop.
- Train `profile_rmse` continued improving while val did not → underfit / poor conditioning.
- **Stale-checkpoint paradox:** models trained on stale inputs score better on the fixed-sat test cache than fresh retrains — stale models learned to ignore broken satellite; fixed-sat retrains see real but unscaled inputs and fail to generalize.

---

## 7. Conclusions and next steps

### Fixed (primary root cause)

- **Stale satellite data:** archives extended through 2022; 42 batch HDF5s regenerated; merged HDF5 rebuilt; **0% time-constant patches** on `h5_stale_check.py` (2015–2022).
- Chronological val/test now receive real temporal satellite variation.

### Still broken

- **Test performance:** fixed-sat L4 raw T **1.857** and anom T **1.748** — below climatology and far from point **0.416**.
- **Input standardization:** production cache is not z-scored; E2 diagnostic showed train RMSE **0.65 → 0.15** with std — highest-confidence next fix.
- **Architecture (lower priority):** conv branch pools to patch-level only; local sss/sst/ssh scalars not in encoder (`n_enc=10` = harmonics + basin + bathy only).

### Recommended next steps

1. **Bake train-split z-scoring into `export_argo_l4_cache.py`** (mirror `make_std_cache.py`); rebuild caches with `--force`.
2. **Retrain:**
   ```bash
   python train.py -c config/argo/config_argo_patch_l4.json --bs 512 -id patch_l4_fixedsat_std
   python train.py -c config/argo/config_argo_patch_l4_anom.json --bs 512 -id patch_l4_anom_fixedsat_std
   ```
3. **Re-eval** with explicit checkpoints; target beat climatology, then close gap to point.
4. **Guard rails:** date tolerance in `retrieve_sat.py`; loud failure on missing basin means (no silent zero-fill).
5. **Architecture** only if standardization retrains still fail.

### Key takeaway

Fixing stale satellite invalidated all prior patch-vs-point comparisons. With clean data, the bottleneck is **input scaling / standardization**, not satellite availability. Patch models must beat **T = 0.416** (point) on the same chronological test split to justify the L4 spatial encoder.

---

## 8. Quick file index

| Stage | Main files |
|-------|------------|
| Config | `NeSPReSO2_onTemplate/config/argo/config_argo_patch_l4.json`, `config_argo_patch_l4_anom.json` |
| Satellite gen | `utils/generate_argo_satellite_data.py`, `utils/retrieve_sat.py` |
| Cache | `NeSPReSO2_onTemplate/preproc/export_argo_l4_cache.py`, `preproc/preproc_isas_sat.py`, `preproc/basin_stats.py` |
| Train | `NeSPReSO2_onTemplate/train.py`, `trainer/trainer.py` |
| Eval | `NeSPReSO2_onTemplate/eval_run.py`, `notebooks/run_argo_production_compare.py`, `scripts/results_table.py` |
| Diagnostics | `NeSPReSO2_onTemplate/diagnostics/stale_sat/*` |
| Data | `data/NeSPReSO_v2_ARGO_GoM_sat/satellite_NeSPReSO_v2_ARGO_GoM.h5`, `data/cache/train_ready_*.pkl` |
| Checkpoints (2026-07-04) | `saved/models/NeSPReSO2_ARGO_GoM_patch_l4/patch_l4_fixedsat/`, `.../patch_l4_anom_fixedsat/` |
| Eval JSON (2026-07-04) | `saved/eval_patch_l4_fixedsat_test.json`, `saved/eval_anom_patch_fixedsat_test.json` |
| Plots | `notebooks/compare_outputs/fixedsat_vs_golden_depth_rmse.png`, `fixedsat_eval_report.json` |
