# Next session prompt — A_CRPS_z32 + RONI / ops / heave

Paste this as the user message. Read [`reports/A_CRPS_z32.json`](A_CRPS_z32.json) and [`reports/eval_acrps_phys_pca32_b.md`](eval_acrps_phys_pca32_b.md) first. Ponytail. Conda `nespreso`. Do not retrain frozen Phase-5 `A_CRPS`. Do not mix 16-PC and 32-PC caches.

## Role

You are continuing NeSPReSO v2 on branch `residual_cube`. The current **z-space CRPS baseline** is registered as **A_CRPS_z32** (prose: A×CRPS-z).

It is PatchConvMLP, 9-d inputs, PCA **32+32**, heteroscedastic head, loss in **physical T/S** after a differentiable PCA inverse (`μ_x = μ_z @ V + mean`, `σ_x = sqrt((σ_z²) @ V²)`). Training recipe (keep unless a cell cannot): `equal_var`, `band_equal` (four evalphys bands), `pc_crps_scale=0.1`, two-stage protocol v2, stage-2 stop on **val ENCE(T)**, val-only σ α with picker val ENCE(T) (`none` / `global_var` / `depth_band_var`). Eval: `eval_run.py` RMSE vs raw `profiles` + `scripts/eval_acrps_phys.py`.

**s42 test pins (n=623, cache `train_ready_4ee013852d33.pkl`):** T RMSE **0.560**, S **0.095**, ENCE(T) raw **0.164**, ENCE(T) after `depth_band_var` α **0.135**. Surface and 50–200 m ENCE(T) still miss 0.20. Checkpoint in the registry JSON (lab path `acrps_phys_pca32_b_s42_s2`). Config: `NeSPReSO2_onTemplate/config/argo/config_argo_A_CRPS_z32.json`.

Frozen **A_CRPS** (PCA-16, PC-space CRPS, 9-d, no ONI/RONI) stays the OSSE ingest cell. A_CRPS_z32 is the skill/calibration baseline for this ablation.

## Question

Do **RONI** (with ONI), the **19 regional operators** from the heave **ops** cache, and/or the **heave** (warp + residual-PC) strategy improve on A_CRPS_z32?

## Run one factor at a time (seed 42, chronological split)

1. **RONI/ONI only.** Same 32-PC cache and z32 loss. Splice CPC ONI+RONI at load (`preproc/enso.py` `inject_enso_columns`, after the 6 harmonics; files `data/indices/`). `input_params.oni/roni=true`, `n_enc=8`, `input_dim=11`, `n_sat=3`. No new PCA. Compare to A_CRPS_z32 pins.

2. **Ops (19 operators).** Heave ops extras live in `data/cache/train_ready_heave_ops.pkl` / `config/argo/config_argo_heave_fast_ops.json` (`cache_kind: heave_ops`, `n_sat=19`, `n_enc=11`, `input_dim=30` after ONI/RONI splice). Operator names are `OP_NAMES` in `preproc/export_heave_ablation_cache.py`. **Do not** silently train HeaveResidualFast. First cell: **A_CRPS_z32 head + z32 loss** on the ops input layout (PatchConvMLP, output 64 = 32+32 μ, PCA from the **32-PC** cache or a new hash if ops features force a new cache — pair checkpoint to that cache). If you must refit PCA, say so; do not reuse `3adcff404b0b` (16-PC A_CRPS).

3. **Heave strategy.** Wrap `HeaveResidualFast`: warp (3) + residual T/S PCs, `decode_warp` then unwarp to physical z. **Keep the z32 probabilistic recipe on the decoded physical T/S** (band-equal CRPS, equal T/S, ENCE(T) stop), not the old `heave_residual_fast` CRPS-on-warp mix unless a cell is labeled “stock HeaveFast” as a control. Residual PCA rank: start at **16+16 on canonical z** (existing heave caches) as control vs stock HeaveFast; a 32+32 residual PCA is a second cell only if (1) beats A_CRPS_z32 on the gates below. HeaveFast already has ONI/RONI; do not confound heave with ops in the same first heave cell.

Optional last cell: ops + heave + z32 loss, only if at least one of (1)–(3) beats baseline on a gate.

## What “improve” means (pre-commit)

Win if **any** of these, same chrono test, seed 42:

- T RMSE < **0.560** by more than seed-noise (~0.01; if only one seed, require ≤ 0.545 or a clear 50–200 m T drop).
- Pooled ENCE(T) raw < **0.164**, or after the same val-α recipe < **0.135**, without tanking T RMSE.
- **50–200 m ENCE(T)** after val-α < **0.415** (A_CRPS_z32 still fails this band). This is the interesting gate.
- D26 / 50–200 T RMSE vs A_CRPS_z32 on `scripts/thermocline_scorecard.py` if you decode both.

Report S RMSE but do not win on S alone. Concat T+S ENCE is not a headline. Do not call a cell an ingest product if 0–50 or 50–200 ENCE(T) still miss 0.20.

## How to train / eval

Reuse `scripts/train_prob_twostage.py --prob-mode crps --stage2-stop val_ence`. Train in **tmux**, pin an **idle** GPU with `CUDA_VISIBLE_DEVICES` and **do not** `srun` without `--gres` (it remaps devices). `eval_run.py` + `scripts/eval_acrps_phys.py`. One short `reports/` md table vs A_CRPS_z32 pins. Register new cells in `reports/A_CRPS_z32.json` or a sibling registry, not by overwriting the baseline checkpoint.

## Constraints

AGENTS.md / ponytail. Chronological split. Paths in config JSON. `selfcheck.py` if you touch loss/heave decode. No commit unless asked. No 3-seed matrix unless a cell clearly wins. Heave N² / D26 caveats in `reports/heave_da_compare.md` still apply — a heave cell can win T RMSE and lose level N²; say so.
