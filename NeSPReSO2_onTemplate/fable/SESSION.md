# Session record — ARGO L4 patch diagnosis

Date: 2026-07-01. Model: Claude Fable 5. Mode: plan mode (read-only diagnosis).
Repo: `/unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project/NeSPReSO2_onTemplate`.

This is a complete record of the diagnostic session that produced `fable/PLAN.md` and
`fable/SUBPROMPTS.md`. It captures the task, every investigation step, the raw evidence gathered,
and the decisions made.

---

## 1. Task

The user pasted a master prompt: audit the offline PyTorch training codebase, find why
`config_argo_patch_l4.json` (ARGO L4 patch model) underperforms the ARGO point baseline, and
produce a prioritized fix plan with validation gates. The prompt described a three-tier agent
orchestration (Cursor Auto / Composer 2.5 / Claude Sonnet) and a phased plan (Phase 0–3).

Measured gap on the chronological test split (623 profiles), profile RMSE on common 0–1800 m grid:

| Model | Config | Arch | Cache tag | T RMSE | S RMSE | Avg | Best epoch |
|-------|--------|------|-----------|--------|--------|-----|------------|
| Baseline (target) | `config_argo.json` | `PatchConvMLP` (9-D point) | `argo_v2` | 0.516 | 0.087 | 0.301 | 449 |
| Underperformer | `config_argo_patch_l4.json` | `PatchMaskConvMLP` (535-D patch) | `argo_l4` | 1.857 | 0.235 | 1.046 | 162 |

---

## 2. Method

Launched three parallel `Explore` subagents (read-only): (a) loss + training config + run history;
(b) L4 data pipeline (normalization, NaN handling, basin fix timing); (c) model architecture +
eval path. Then verified the load-bearing findings directly with Bash/Python, and launched one
`Plan` subagent to design the fix and independently re-verify the code claims.

---

## 3. Evidence gathered

### 3.1 Architecture / eval agent
- `PatchMaskConvMLP` input unpacking is **correct**: `enc = x[:, :10]`, `sat_flat = x[:, 10:535]`,
  reshaped to `(B,3,7,5,5)` via `_unpack_sat_channels` (`model/model.py:300-323`). Covered by
  `selfcheck.py:87-158` and `scripts/verify_argo_l4_layout.py`. **Column ordering ruled out.**
- Conv3d trunk 3→32→64 + `AdaptiveAvgPool3d([1,1,1])` → `Linear(64,128)`; head `[512,512]` vs
  baseline `[1024,1024]`.
- `eval_run.py` pairs each checkpoint with the cache from its own config hash (`ensure_cache`);
  bathy depth-mask via `bottom_depth`/`PRES`. Same ARGO truth + split → direct comparison valid.
- `argo_production_results.json`: point run cache `train_ready_ff2393a1ea21.pkl`; patch run
  `0701_013207` + cache `train_ready_950b7c12bd46.pkl` (later found deleted).

### 3.2 Loss / training-config agent
- `model/loss.py`: `DEFAULT_COMBINED_MSE_SCALE=0.0255`, `DEFAULT_PROFILE_SCALES={T:37.86,S:0.28}`.
  `pc_mse_only` = `weighted_mse_loss / combined_mse_scale` (PC space only, **no profile
  reconstruction**). `combined` = `(pca_loss/combined_pca_scale + weighted_mse/combined_mse_scale)/2`
  where `pca_loss` reconstructs profiles and MSEs them (includes the eval metric).
- Config diff: patch has `loss_config.mode="pc_mse_only"` and **no `loss_scales`** block; baseline
  has `"combined"` with tuned scales (`combined_mse_scale=0.2174`, `combined_pca_scale=2.0`,
  profile scales T=2.0029/S=0.0313). Only the two `pc_mse_only` configs lack `loss_scales`.
- Both configs: epochs 8000, early_stop 500, Adam lr 1e-3, StepLR effectively flat.
- Surviving patch run `0701_102436`: `batch_size` probe resolved **2755** of n_train=2901; best
  val_loss 5.535 @ ~epoch 252; early-stop @753. Baseline `argo16_scales`: batch 512, val_loss ~0.12.

### 3.3 Data-pipeline agent
- `build_argo_l4_input_matrix` (`preproc/preproc_isas_sat.py:275-351`) stacks raw columns, NaN→0,
  **no per-column standardization**. Scalar order: 6 harmonics, basin_sss/sst/ssh, bathy_depth,
  then 3×175 patch cols. Basin SST converted K→C at build time (line 313).
- `compute_basin_daily_means` (`preproc/basin_stats.py:153-197`) caches a lookup pickle and reuses
  it unless `force=True`; `build_argo_l4_cache` does **not** pass `force` → basin-fix propagation
  gap.
- v2 baseline inputs come **pre-normalized** from the v2 pickle (`export_v2_cache.py:67-69`).
- `config_hash` covers `input_params`, `io`, `outputs`, `use_mask_channels`; not `data_loader`.

### 3.4 Direct verification (my own Bash/Python)
- `git log`: basin fix = commit `280dd68`, 2026-07-01 11:04:36.
- No input scaler anywhere in `train.py` / `data_loader/` / `base/` (only MNIST-style transform and
  an unrelated `normalize_feature_plane` in `l3_input.py`).
- Cache pickles: production L4 cache is `train_ready_4411c65ee518.pkl` (07/01 10:24); the published
  `train_ready_950b7c12bd46.pkl` and run `0701_013207` are **absent** (deleted). Only surviving
  patch run is `0701_102436` (status: done, early_stop, epoch 753, mnt_best 5.535).
- **Cache column statistics** (`train_ready_4411c65ee518.pkl`, inputs (4145, 535)):

```
col 0 timecos      min=  -1.000 max=   1.000 mean=  -0.017 std= 0.710 zeros=0.000
col 1 timesin      min=  -1.000 max=   1.000 mean=   0.077 std= 0.700 zeros=0.000
col 2 latcos       min=   0.539 max=   0.783 mean=   0.641 std= 0.046 zeros=0.000
col 3 latsin       min=   0.622 max=   0.843 mean=   0.765 std= 0.040 zeros=0.000
col 4 loncos       min=  -0.118 max=   0.128 mean=   0.007 std= 0.062 zeros=0.000
col 5 lonsin       min=  -1.000 max=  -0.992 mean=  -0.998 std= 0.002 zeros=0.000
col 6 basin_sss    min=   0.000 max=  35.528 mean=   0.179 std= 2.510 zeros=0.995   <-- BROKEN
col 7 basin_sst    min=   0.000 max=  25.361 mean=   0.119 std= 1.671 zeros=0.995   <-- BROKEN
col 8 basin_ssh    min=   0.000 max=   0.383 mean=   0.002 std= 0.025 zeros=0.995   <-- BROKEN
col 9 bathy_depth  min=  41.000 max=3789.000 mean=2686.9  std=1030.3 zeros=0.000    <-- HUGE SCALE
sss_patch  min=0.000 max=38.423 mean=35.765 std=2.900 zero_frac=0.0062
sst_patch  min=0.000 max=31.950 mean=26.366 std=2.646 zero_frac=0.0010
ssh_patch  min=-0.128 max=1.265 mean=0.426 std=0.211 zero_frac=0.0011
```

This single table confirms root causes 1 (basin cols 99.5% zero) and 2 (bathy std 1030 next to
±1 harmonics; raw patch magnitudes; no standardization).

### 3.5 Plan agent (independent code re-verification)
- Recomputed `config_hash` of the patch config → `4411c65ee518` (matches the stale cache).
- Confirmed splits reproducible at build time via `base/split_utils.py build_split_indices`.
- Confirmed `combined` loss needs nothing the argo_l4 cache lacks (`pca_models`, `bottom_depth`,
  `PRES` all present) — same wiring as argo_v2.
- Confirmed `validate_config` does not whitelist `io` keys → `io.standardize_inputs` is safe.
- Found the stale basin lookup on disk: `../data/cache/basin_daily_means_f9dd5bc4df.pkl`.
- No contradictions with the evidence above.

---

## 4. Root causes (final ranking)

1. Basin features dead (99.5% zeros) — basin `force` not propagated in `export_argo_l4_cache.py`.
2. No input standardization — raw magnitudes span ±1 to ~3800; v2 baseline was pre-normalized.
3. Loss mismatch — `pc_mse_only` + default scales never optimizes profile reconstruction.
4. Near-full-batch (2755/2901) vs baseline 512.
5. Stale published comparison — cited run + cache deleted; surviving run never test-evaluated.

Ruled out: input ordering / unpacking; eval protocol. Secondary A/B: `use_mask_channels`, head
width, pooling.

---

## 5. Fix design (verified)

- Standardize at cache-build time, train-split stats, scaler in payload, gated by new
  `io.standardize_inputs: true` (changes hash → fresh cache → pairing preserved). `fill_nan=False`
  kwarg on `build_argo_l4_input_matrix` so the builder sees raw NaNs; NaN→0 after z-score.
- Pass `force=force` to `compute_basin_daily_means`.
- Config: `mode="combined"` + baseline `loss_scales` + `batch_size 512` (production + smoke).
- Order: honest re-eval of `0701_102436` (old hash, no `--force`) BEFORE any rebuild.
- Fix-all first (smoke → production, tmux GPU), ablations only if it fails 0.301.
- Gates: selfcheck scaler/basin assertions; success = test (T+S)/2 ≤ 0.301, no >5% per-var regress.

Full detail: `fable/PLAN.md`. Executor prompts: `fable/SUBPROMPTS.md`.

---

## 6. Decisions made by the user during the session

- **Execution:** "Report + sub-prompts only" — I make no code/config/cache changes; deliverables are
  the report and paste-ready sub-prompts for the three-tier orchestration.
- **GPU gate:** "Proceed automatically using tmux" — sub-prompts instruct executors to run the
  production retrain without pausing, with long jobs inside tmux.
- **Save:** save the plan, sub-prompts, and a full session copy into a folder named `fable`.

---

## 7. Key file references

| Concern | Path |
|---------|------|
| Patch model | `model/model.py` (`PatchMaskConvMLP` ~200-330; `_unpack_sat_channels` 300-313) |
| L4 input matrix | `preproc/preproc_isas_sat.py:275-351` (`build_argo_l4_input_matrix`); `config_hash` 186-195 |
| Cache builder | `preproc/export_argo_l4_cache.py` (basin call ~117-122; payload ~184-211) |
| Basin stats | `preproc/basin_stats.py:153-197` (`compute_basin_daily_means`); fix in commit 280dd68 |
| Loss | `model/loss.py` (defaults 17-19; `pc_mse_only`/`combined` 520-527; `make_loss` ~599-601) |
| Splits | `base/split_utils.py` (`build_split_indices`) |
| Train | `train.py` (`ensure_cache` 37-85; `resolve_dataloader_batch_size` 98-145) |
| Eval | `eval_run.py` (`raw_profile_rmse` 64-120; main 200-235) |
| Arch sync | `preproc/l3_input.py:134-186` (`sync_arch_with_io`) |
| Selfcheck | `selfcheck.py` (`test_argo_l4_*` 87-251); `scripts/verify_argo_l4_layout.py` |
| Production cache | `../data/cache/train_ready_4411c65ee518.pkl` |
| Stale basin lookup | `../data/cache/basin_daily_means_f9dd5bc4df.pkl` |
| Surviving patch run | `saved/models/NeSPReSO2_ARGO_GoM_patch_l4/0701_102436/` |
| Baseline run | `saved/models/NeSPReSO2_ARGO_GoM/argo16_scales/` |
| Published metrics | `notebooks/compare_outputs/argo_production_results.json` |
