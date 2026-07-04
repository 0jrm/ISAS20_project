# ARGO L4 patch underperformance — diagnosis & fix plan

## Context

The ARGO **L4 patch model** (`config/argo/config_argo_patch_l4.json`, `PatchMaskConvMLP`,
535-dim input = 6 harmonics + basin_sss/sst/ssh + bathy_depth + 3×175 flattened 5×5×7 patches)
scores test RMSE **T=1.857 / S=0.235** on the chronological test split (623 profiles) versus the
point baseline (`config/argo/config_argo.json`, `PatchConvMLP` point mode, 9-dim) at
**T=0.516 / S=0.087** — a ~3.5× / ~2.7× regression despite strictly richer inputs.

Diagnosis was completed read-only this session. This document is the root-cause report and the
verified fix design. Execution is delegated to the three-tier Cursor/Claude orchestration; the
paste-ready sub-prompts are in [`SUBPROMPTS.md`](SUBPROMPTS.md).

---

## Root causes (verified, ranked by likelihood × impact)

| # | Root cause | Type | Evidence |
|---|------------|------|----------|
| 1 | **Basin features are dead (99.5% zeros)** | Bug | Direct inspection of `../data/cache/train_ready_4411c65ee518.pkl`: cols 6–8 (`basin_sss/sst/ssh`) are 99.5% zeros. Fix (commit 280dd68, `basin_stats.py` depth-squeeze + 0–360 lon) is present, but `build_argo_l4_cache` (`preproc/export_argo_l4_cache.py:117-122`) never passes `force=` to `compute_basin_daily_means`, so the stale zero lookup `../data/cache/basin_daily_means_f9dd5bc4df.pkl` was reused. |
| 2 | **No input standardization** | Bug / design | `build_argo_l4_input_matrix` (`preproc/preproc_isas_sat.py:275-351`) stacks raw values: harmonics ±1 alongside `bathy_depth` 41–3789 (std≈1030), raw patches (SSS ~35±3, SST ~26±3, SSH ~0.4±0.2), NaN→0. No scaler exists in the train/eval path. The v2 baseline's 9 inputs arrive **pre-normalized** from the v2 pickle — the comparison was never apples-to-apples. |
| 3 | **Loss objective mismatch** | Bug | Patch uses `loss_config.mode: "pc_mse_only"` with **no `loss_scales`** block (defaults `combined_mse_scale=0.0255`, profile scales T=37.86/S=0.28). It never optimizes profile reconstruction — the eval metric. Baseline uses `"combined"` with tuned scales (`combined_mse_scale=0.2174`, `combined_pca_scale=2.0`, profile scales T=2.0029/S=0.0313). |
| 4 | **Near-full-batch training** | Design | `batch_size: 0` → VRAM probe resolved **2755 of n_train=2901** → ~1 step/epoch, vs baseline's fixed **512**. |
| 5 | **Stale published comparison** | Process | `notebooks/compare_outputs/argo_production_results.json` cites run `0701_013207` + cache `train_ready_950b7c12bd46.pkl` — **both deleted**. Surviving run `saved/models/NeSPReSO2_ARGO_GoM_patch_l4/0701_102436` (best val_loss 5.535 @ ~epoch 252, early-stop @753) was **never test-evaluated**. |

**Ruled out:** input column ordering / `_unpack_sat_channels` reshape (verified correct; covered by
`selfcheck.py` + `scripts/verify_argo_l4_layout.py`); eval protocol (same ARGO truth + same split
mode → direct `eval_run.py` comparison on the test split is valid).

**Secondary A/B candidates (not bugs, defer):** `use_mask_channels=false` (NaN→0 discards
observability), head `[512,512]` vs baseline `[1024,1024]`, `AdaptiveAvgPool3d([1,1,1])`
over-compression of spatial signal.

---

## Verified fix design

Independently re-verified against the code by the Plan agent (recomputed `config_hash` →
`4411c65ee518` matches; confirmed the basin `force` gap; confirmed `combined` loss needs no new
wiring because the argo_l4 cache already stores `pca_models`, `bottom_depth`, `PRES`).

### A. Input standardization (main code change)
- **Where:** cache-build time, **train-split stats only**, scaler stored in the payload.
- **Why not load-time:** by load time NaNs are already zero-filled, so stats would be contaminated
  and 0 PSU SSS would become a large fake z-score. Only the cache builder sees the raw NaN pattern.
- **Config key:** `io.standardize_inputs: true` — hashed by `config_hash`
  (`preproc_isas_sat.py:186-195`) → fresh cache filename → **old checkpoint/cache pairing preserved
  by construction**. Backward compatible (old caches lack `input_scaler`; loader never reads it).
- **NaN handling:** fill 0 **after** z-scoring = mean imputation (simplest correct option).
- **Split reproducibility:** `base/split_utils.py build_split_indices(n, juld, dl_args, ...)` is
  deterministic from `config["data_loader"]["args"]` + `juld`, both available at build time.

Code touch points:
- `preproc/preproc_isas_sat.py` — add `fill_nan: bool = True` kwarg to `build_argo_l4_input_matrix`
  (default = byte-identical to today; `False` preserves NaNs for the builder to standardize).
- `preproc/export_argo_l4_cache.py` — after building `inputs`: if `standardize_inputs`, compute
  train-split `nanmean`/`nanstd`, z-score, `nan_to_num(...,0.0)`, store
  `payload["input_scaler"] = {"mean", "std"}`.

### B. Basin fix propagation
- One line in `preproc/export_argo_l4_cache.py:117-122`: pass `force=force` to
  `compute_basin_daily_means(...)`. (Optional belt-and-suspenders: delete
  `../data/cache/basin_daily_means_f9dd5bc4df.pkl` before rebuild.)

### C. Loss + batch alignment (config-only, production + smoke)
```json
"io":          { ..., "standardize_inputs": true },
"loss_config": { "mode": "combined" },
"loss_scales": { "profile_scales": { "temperature": 2.0029, "salinity": 0.0313 },
                 "combined_pca_scale": 2.0, "combined_mse_scale": 0.2174 },
"data_loader": { "args": { ..., "batch_size": 512 } }
```
No code change for combined loss — identical path to the working argo_v2 baseline.

---

## Ordering constraint (hard)

Run the **honest re-eval of `0701_102436`** (under old hash `4411c65ee518`, no `--force`) **before
any `--force` rebuild**. After `standardize_inputs` lands the hash differs, so there is no cache
collision and no risk of pairing a checkpoint with a rebuilt cache.

---

## Experiment sequence

1. **Phase 0 — honest baseline** (CPU): re-eval `0701_102436` on the test split via its own
   `config.json`; inspect the basin lookup pickle. Replaces the stale published number.
2. **Phase 1 — code + config + selfcheck** then `python3 selfcheck.py` (CPU); smoke rebuild+train
   on `config_argo_patch_l4_smoke.json`; verify gates.
3. **Phase 2 — production** (auto-proceed per user; long jobs in **tmux**): `export_argo_l4_cache.py
   --force` (CPU; basin lookup recompute is the slow part), verify basin/scaler gates on the NEW
   cache, then `srun --gres=gpu:1 train.py`, then `eval_run.py --split test`.
4. **Phase 3 — fallback ablations** (only if fix-all doesn't beat 0.301), under
   `config/argo/ablations/`: center-pixel `PatchConvMLP` (isolates data vs conv path), single-fix
   ablations, then `use_mask_channels: true` / `head_hidden: 1024`.

**Fix-all first** (not one-at-a-time): the three defects are independent and each individually
sufficient to explain a large gap; one-at-a-time would burn 3+ GPU runs re-learning what's known.

---

## Validation gates

- `selfcheck.py` additions: scaler sanity (`mean[6]∈(25,40)` PSU, `mean[7]∈(10,35)` °C, `std[9]>100`
  bathy), standardized-column std ≈1 on train rows, `fill_nan` backward-compat test, basin lookup
  >99% finite in 25–40 PSU.
- **Success criterion:** test `(raw_profile_rmse.temperature + raw_profile_rmse.salinity)/2 ≤ 0.301`
  (baseline 0.3015), with no >5% regression on either variable.

---

## Risk register

- Never `--force`-rebuild under old hash `4411c65ee518` until the `0701_102436` re-eval is archived.
- `resolve_dataloader_batch_size` short-circuits for `batch_size>0`, so 512 skips the VRAM probe.
- PCA refit is deterministic given identical profiles; each checkpoint is only evaluated against the
  cache its hash points to — pairing invariant holds.
- Patch may legitimately lose: at fixed ARGO point locations, 5×5×7 spatial-temporal context can add
  variance without bias reduction. If fix-all still trails after ablations, pivot to an honest
  stratified narrative (`eval_stratified.py` patterns) on where spatial context helps vs hurts.
