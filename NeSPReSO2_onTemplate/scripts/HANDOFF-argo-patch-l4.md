# Handoff — ARGO L4 patch model (`config/argo/config_argo_patch_l4.json`)

**Updated:** 2026-06-30  
**Goal:** Train `PatchMaskConvMLP` on ARGO targets (16+16 PCA) with 1°×1°×7-day L4 patch inputs (SSS/SST/SSH + masks), bathymetry depth, harmonics, and basin-wide satellite means.

---

## What was done this session

| Item | Status |
|------|--------|
| `preproc/basin_stats.py` — squeeze depth dim, 0–360° lon, SMAP fallback | **Fixed** |
| Smoke cache + train with valid basin SSS (~35 PSU) | **Done** |
| `astropy` installed in `nespreso` (needed by `utils/retrieve_sat.py`) | **Done** |
| Satellite batch HDF5 generation (4145 profiles, batch=100) | **In progress** |
| Full train cache + full training | **Blocked** on satellite HDF5 + GPU node |

### Basin SSS bug (fixed)

CMEMS daily files keep a depth dimension of size 1; the old code compared shape `(1, 1440, 2880)` to a 2-D mask → always NaN → filled as 0 in the train matrix. Fixed by squeezing to 2-D and matching lon on 0–360 grids. SMAP 8-day running mean is fallback if CMEMS lookup fails.

---

## Active tmux sessions

```bash
tmux ls
tmux attach -t argo_sat_gen          # satellite retrieval (login or compute node)
tmux attach -t argo_patch_l4_train   # waits for combined HDF5, then cache + train
```

Logs:

```bash
tail -f ISAS20_project/utils/logs/argo_sat_gen.log
tail -f ISAS20_project/utils/logs/argo_patch_l4_train.log
```

Batch files:

```bash
ls ISAS20_project/data/NeSPReSO_v2_ARGO_GoM_sat/argo_sat_batches/
cat ISAS20_project/data/NeSPReSO_v2_ARGO_GoM_sat/argo_sat_batches/manifest.json
```

---

## Commands (run on compute node with `nespreso`)

All paths relative to repo root `ISAS20_project/`.

### 1. Satellite HDF5 (resumable batches)

```bash
conda activate nespreso
bash utils/run_argo_sat_tmux.sh
# or foreground:
cd utils
srun --ntasks=1 --cpus-per-task=8 python3 generate_argo_satellite_data.py \
  -c NeSPReSO2_onTemplate/config/argo/config_argo_patch_l4.json --batch-size 100
```

**Resume behavior:** existing `argo_sat_batch_b0100_{start}_{end}.h5` files are skipped. To **re-run batches 0–300**, delete those three files and clear `completed_batches` in `manifest.json`, then re-launch.

**Flags:** `--force` (wipe all batches), `--no-resume`, `--combine-only`

**Output:** `data/NeSPReSO_v2_ARGO_GoM_sat/satellite_NeSPReSO_v2_ARGO_GoM.h5` (auto-combined when all 42 batches exist)

### 2. Train cache

```bash
cd NeSPReSO2_onTemplate
srun --ntasks=1 --cpus-per-task=8 python3 preproc/export_argo_l4_cache.py \
  -c config/argo/config_argo_patch_l4.json --force
```

Deletes stale basin cache if needed: `data/cache/basin_daily_means_*.pkl`

### 3. Train full model (GPU)

```bash
cd NeSPReSO2_onTemplate
srun --ntasks=1 --cpus-per-task=8 --gres=gpu:1 python3 train.py \
  -c config/argo/config_argo_patch_l4.json
```

### 4. Eval vs ARGO point baseline

```bash
cd NeSPReSO2_onTemplate
python3 eval_run.py -c config/argo/config_argo_patch_l4.json \
  -r saved/models/NeSPReSO2_ARGO_GoM_patch_l4/<run>/model_best.pth
python3 eval_run.py -c config/argo/config_argo.json \
  -r saved/models/NeSPReSO2_ARGO_GoM/argo16_scales/model_best.pth
```

Or full pipeline script (satellite → cache → train on GPU node):

```bash
cd NeSPReSO2_onTemplate
bash scripts/run_argo_patch_l4.sh
```

Post-satellite waiter (cache + train after combine):

```bash
bash utils/run_argo_patch_l4_post_sat_tmux.sh
```

### Smoke path (16 samples, already validated)

```bash
cd NeSPReSO2_onTemplate
python3 preproc/export_argo_l4_cache.py -c config/argo/config_argo_patch_l4_smoke.json --force
python3 train.py -c config/argo/config_argo_patch_l4_smoke.json
```

---

## Loss — PCA-space MSE only

Training uses **`loss_config.mode: "pc_mse_only"`** — weighted MSE in PC coefficient space only (no profile-space branch, no combined PCA/profile mix).

In `model/loss.py`, this is `genWeightedMSELoss(pred_pcs, target_pcs)` normalized by `combined_mse_scale` (default **0.0255** if omitted from config). Config has **no `loss_scales` block**; defaults are fine unless you want to tune the scale later via:

```json
"loss_scales": { "combined_mse_scale": 0.0255 }
```

Eval still reports **`profile_rmse`** (reconstruct profiles from PCs for monitoring). Profile RMSE is not in the training loss for this run.

---

## Model summary

| Component | Value |
|-----------|-------|
| Config | `config/argo/config_argo_patch_l4.json` |
| Arch | `PatchMaskConvMLP`, input_dim=535 (3 value ch; set `use_mask_channels: true` → 1060), 16+16 PCA out |
| Patch | 5×5 spatial × 7 temporal × (SSS,SST,SSH values; masks optional via `arch.args.use_mask_channels`) |
| Scalars | time/lat/lon harmonics, basin SSS/SST/SSH, bathy depth |
| Truncation | `bathy_truncation: true` in cache |
| Split | chronological 70/15/15 |
| Loss | `pc_mse_only` — weighted PC-space MSE (`combined_mse_scale` default 0.0255) |
| Eval metric | `profile_rmse` (monitoring only, not in loss) |

---

## Blockers / notes

| Item | Notes |
|------|-------|
| Satellite runtime | ~3–4 h for 4145×7-day retrieval; resumable via batch files |
| GPU training | Must use compute node (`srun --gres=gpu:1`); login node has no GPU |
| `srun` on login node | `utils/run_argo_sat_tmux.sh` falls back to bare `python3` when `srun` missing |
| sklearn PCA version | v2 pickle 1.2 vs 1.5+ warning; `refit_pca: true` in config mitigates |
| SMAP fallback env | `NESPRESO_SMAP_SSS_ROOT` overrides default GOFFISH path |

---

## Key files touched

- `NeSPReSO2_onTemplate/preproc/basin_stats.py` — basin mean fix + selfcheck
- `utils/run_argo_sat_tmux.sh` — srun fallback
- `utils/run_argo_patch_l4_post_sat_tmux.sh` — post-satellite waiter
- `NeSPReSO2_onTemplate/scripts/run_argo_patch_l4.sh` — srun fallback
