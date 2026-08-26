#!/bin/bash
# Balanced physical A×CRPS (32+32): equal T/S, band CRPS, PC regularizer, ENCE(T) stop.
set -euo pipefail
cd /unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project/NeSPReSO2_onTemplate
PY=/conda/jmiranda/miniconda/envs/nespreso/bin/python3
CFG=config/argo/config_argo_acrps_phys_pca32_b.json
CKPT=saved/acrps_phys_pca32_b/models/NeSPReSO2_ARGO_GoM_acrps_phys_pca32_b_acrps_phys_pca32_b_s42_s2/acrps_phys_pca32_b_s42_s2/model_best.pth
LOG=saved/log/tmux_acrps_phys_pca32_b.log
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "START $(date -Is) host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"

# srun without --gres remaps CUDA_VISIBLE_DEVICES; pin GPU in the environment and run python.
export OMP_NUM_THREADS=8
"$PY" selfcheck.py test_pca_hetero_phys_decode_and_grad

"$PY" scripts/train_prob_twostage.py \
  -c "$CFG" --prob-mode crps --parent-tag acrps_phys_pca32_b_s42 --stage2-stop val_ence

"$PY" eval_run.py \
  -c "$CFG" -r "$CKPT" --split test --out ../reports/eval_acrps_phys_pca32_b_s42.json

"$PY" scripts/eval_acrps_phys.py \
  -c "$CFG" -r "$CKPT" --out ../reports/eval_acrps_phys_pca32_b_cal.json

"$PY" - <<'PY'
import json
from pathlib import Path
rmse = json.loads(Path("../reports/eval_acrps_phys_pca32_b_s42.json").read_text())
cal = json.loads(Path("../reports/eval_acrps_phys_pca32_b_cal.json").read_text())
a = json.loads(Path("../reports/eval_A_CRPS.json").read_text())
v1 = json.loads(Path("../reports/eval_acrps_phys_pca32_s42.json").read_text())
raw, rec = cal["test_raw"], cal["test_recalib"]
md = f"""# Physical A×CRPS 32+32, balanced (seed 42)

Same cache as the first physical run (`4ee013852d33`, 32 PCs). Changes vs that run:

1. Val-only σ α: `none` / `global_var` / `depth_band_var`; pick by val ENCE(T); score test once.
2. Equal T/S: `0.5 L_T + 0.5 L_S` (no MSE profile_scales).
3. Stage-2 stop on ENCE(T) only.
4. Physical term = mean of 4 depth-band means.
5. `0.1 ×` PC-space CRPS (stage 1: PC MSE).
6. Stage 1 uses the same equal/band physical MSE; μ LR × 0.1 in stage 2 unchanged.

Checkpoint: `{cal["checkpoint"]}`

## Test RMSE (raw profiles)

| | T | S |
|--|--:|--:|
| A×CRPS s42 (16+16 PC CRPS) | {a["raw_profile_rmse"]["temperature"]:.3f} | {a["raw_profile_rmse"]["salinity"]:.3f} |
| phys v1 (profile_scales, concat ENCE) | {v1["raw_profile_rmse"]["temperature"]:.3f} | {v1["raw_profile_rmse"]["salinity"]:.3f} |
| phys balanced | {rmse["raw_profile_rmse"]["temperature"]:.3f} | {rmse["raw_profile_rmse"]["salinity"]:.3f} |

## Test analytic calibration

Val recipe **{cal["best_recipe"]}**. Gate ENCE(T) < 0.20.

| | CRPS(T) | ENCE(T) | CRPS(S) | ENCE(S) |
|--|--------:|--------:|--------:|--------:|
| raw σ | {raw["temperature"]["crps_mean"]:.3f} | {raw["temperature"]["ence"]:.3f} | {raw["salinity"]["crps_mean"]:.3f} | {raw["salinity"]["ence"]:.3f} |
| val-α σ | {rec["temperature"]["crps_mean"]:.3f} | {rec["temperature"]["ence"]:.3f} | {rec["salinity"]["crps_mean"]:.3f} | {rec["salinity"]["ence"]:.3f} |

ENCE(T) recalib pass: **{cal["test_ence_T_pass_recalib"]}**. Concat T+S ENCE is not the headline.
"""
Path("../reports/eval_acrps_phys_pca32_b.md").write_text(md)
print("wrote ../reports/eval_acrps_phys_pca32_b.md")
PY
echo "DONE $(date -Is)"
