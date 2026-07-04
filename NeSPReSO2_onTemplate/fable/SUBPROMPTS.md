# Paste-ready sub-prompts (three-tier orchestration)

Repo: `/unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project/NeSPReSO2_onTemplate`
Conda env: `nespreso`. HPC: cap CPU with `srun --ntasks=1 --cpus-per-task=8`; GPU needs `--gres=gpu:1`.
Long training runs go inside **tmux** so they survive the shell. Full root cause: see `fable/PLAN.md`.

Order is strict: **1 → 2 → 3**. Sub-prompt 1 must complete (and its result be archived) before any
`--force` cache rebuild in sub-prompt 3.

---

## Sub-prompt 1 — Cursor Auto (light eval, read-only + one eval)

```
Read-only diagnostics for the ARGO L4 patch investigation. Conda env: nespreso. Do NOT edit
any code, config, or cache. cd into
/unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project/NeSPReSO2_onTemplate.

1. Honest re-eval of the surviving patch run using ITS OWN config (this pins config_hash
   4411c65ee518 and reuses the existing stale cache — do NOT add --force):
     python3 eval_run.py \
       -c saved/models/NeSPReSO2_ARGO_GoM_patch_l4/0701_102436/config.json \
       -r saved/models/NeSPReSO2_ARGO_GoM_patch_l4/0701_102436/model_best.pth \
       --split test --out saved/models/NeSPReSO2_ARGO_GoM_patch_l4/0701_102436/eval_test.json
   Report raw_profile_rmse.temperature, .salinity, and (T+S)/2. This is the honest current
   patch number for the checkpoint as trained (the published 0701_013207 run and its cache
   were deleted).

2. Confirm the basin bug in the production cache: load ../data/cache/train_ready_4411c65ee518.pkl,
   print for columns 6,7,8 (basin_sss/sst/ssh) the fraction of exact zeros and the max value.
   Expected: ~0.995 zeros (broken). Also load ../data/cache/basin_daily_means_f9dd5bc4df.pkl and
   report the fraction of finite values in lookup["sss"] and their min/max (expect mostly
   non-finite or 0 if stale; a healthy lookup is >99% finite in 25-40 PSU).

3. Report both results as a short table. Do not change anything.
```

---

## Sub-prompt 2 — Cursor Composer 2.5 (code + config implementation)

```
Implement the ARGO L4 standardization + basin + loss/batch fix. Conda env: nespreso. Minimal
diffs. cd /unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project/NeSPReSO2_onTemplate.
Do NOT overwrite production caches or run any --force rebuild in this task.

(1) preproc/preproc_isas_sat.py — build_argo_l4_input_matrix (lines ~275-351):
    Add kwarg `fill_nan: bool = True`. When True, behavior is byte-identical to today. When False:
    in the satellite-patch loop append the raw `val` columns (do NOT apply np.where(finite, val,
    0.0)); and skip the final np.nan_to_num(stacked, nan=0.0) at line ~351 so basin/bathy NaNs also
    survive. No other signature or call-site changes.

(2) preproc/export_argo_l4_cache.py — build_argo_l4_cache:
    (a) Pass force=force to compute_basin_daily_means(...) (currently ~lines 117-122 it is not
        passed — this is the basin-fix propagation gap).
    (b) Read `standardize = bool(io_cfg.get("standardize_inputs", False))`. Call
        build_argo_l4_input_matrix(..., fill_nan=not standardize). When standardize:
          from base.split_utils import build_split_indices
          dl_args = (config.get("data_loader") or {}).get("args", {})
          train_idx = build_split_indices(inputs.shape[0], juld, dl_args,
                        dataset_tag=io_cfg.get("dataset_tag","argo_l4"),
                        v2_src=io_cfg.get("v2_src"))["train"]
          mean = np.nan_to_num(np.nanmean(inputs[train_idx], axis=0))
          std  = np.nanstd(inputs[train_idx], axis=0); std = np.where(std > 1e-6, std, 1.0)
          inputs = np.nan_to_num(((inputs - mean) / std).astype(np.float32), nan=0.0)
        Add to payload: payload["input_scaler"] = {"mean": mean, "std": std} (only when standardizing).
        Add a comment noting split settings are not in config_hash (benign; chronological split frozen).

(3) config/argo/config_argo_patch_l4.json AND config/argo/config_argo_patch_l4_smoke.json:
    - io: add  "standardize_inputs": true
    - loss_config.mode: "combined"   (was "pc_mse_only")
    - add loss_scales block:
        {"profile_scales": {"temperature": 2.0029, "salinity": 0.0313},
         "combined_pca_scale": 2.0, "combined_mse_scale": 0.2174}
    - data_loader.args.batch_size: 512  (production was 0; keep smoke's small batch, e.g. 8)
    Do NOT touch config_argo.json or any argo_v2 config.

(4) selfcheck.py — add gates (guarded to skip when data absent, like existing test_argo_l4_*):
    - In/near test_argo_l4_cache_smoke: if cache.get("input_scaler") present, assert
      np.isfinite(cache["inputs"]).all(); assert scaler mean[6] in (25,40), mean[7] in (10,35),
      std[9] > 100; assert per-column std of standardized inputs in (0.05,3) for cols with
      scaler std > 1e-6.
    - New test next to test_argo_l4_input_dim_and_forward: build_argo_l4_input_matrix(...,
      fill_nan=False) preserves injected NaNs; default fill_nan=True output unchanged vs today.

Test command: python3 selfcheck.py   (must pass; it will skip data-dependent gates if the
standardized cache does not exist yet — that is fine at this stage).
```

---

## Sub-prompt 3 — Cursor Auto / Claude Code (rebuild, train, eval; auto-proceed via tmux)

```
Run the ARGO L4 fix-all pipeline end to end. Conda env: nespreso. Only proceed if Sub-prompt 1 has
been archived (the 0701_102436 test eval exists). cd
/unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project/NeSPReSO2_onTemplate.

1) selfcheck: python3 selfcheck.py  (must pass).

2) SMOKE (CPU):
   srun --ntasks=1 --cpus-per-task=8 python3 preproc/export_argo_l4_cache.py \
     -c config/argo/config_argo_patch_l4_smoke.json --force
   srun --ntasks=1 --cpus-per-task=8 python3 train.py -c config/argo/config_argo_patch_l4_smoke.json
   Gate: cache has input_scaler; basin scaler mean[6] in 25-40 PSU; training loss finite & decreasing.
   Stop and report if any gate fails.

3) PRODUCTION cache rebuild (CPU; slow part is basin lookup recompute):
   srun --ntasks=1 --cpus-per-task=8 python3 preproc/export_argo_l4_cache.py \
     -c config/argo/config_argo_patch_l4.json --force
   Verify on the NEW cache file (new hash): basin cols now non-zero (load cache, but note inputs are
   z-scored — check cache["input_scaler"]["mean"][6] in (25,40), [7] in (10,35), [9] in (40,3800));
   basin_daily_means lookup >99% finite in 25-40 PSU. Stop and report if it still looks broken.

4) PRODUCTION train (GPU, in tmux so it survives disconnects):
   tmux new-session -d -s argo_l4_train \
     'conda activate nespreso && cd <repo> && \
      srun --ntasks=1 --cpus-per-task=8 --gres=gpu:1 \
      python3 train.py -c config/argo/config_argo_patch_l4.json \
      2>&1 | tee saved/log/argo_l4_fixall.log'
   Poll the log / status.json for early_stop or done. Model is small, batch 512, ~6 steps/epoch.

5) EVAL the new run:
   python3 eval_run.py -c config/argo/config_argo_patch_l4.json \
     -r saved/models/NeSPReSO2_ARGO_GoM_patch_l4/<new_run>/model_best.pth \
     --split test --out saved/models/NeSPReSO2_ARGO_GoM_patch_l4/<new_run>/eval_test.json
   SUCCESS if (temperature + salinity)/2 <= 0.301 with no >5% per-variable regression vs baseline
   (T 0.516, S 0.087). Regenerate notebooks/compare_outputs/argo_production_results.json.

If the criterion is NOT met, do NOT keep tuning blindly — stop and report the numbers so the
hard-reasoning tier can decide between ablations (config/argo/ablations/: center-pixel PatchConvMLP,
single-fix isolation, use_mask_channels=true) and an honest stratified negative-result narrative.
```
