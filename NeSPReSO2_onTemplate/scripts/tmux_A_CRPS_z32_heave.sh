#!/bin/bash
# HeaveResidualFast + z32 physical T/S CRPS (16+16 residual PCs, ONI/RONI, no ops).
set -euo pipefail
cd /unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project/NeSPReSO2_onTemplate
PY=/conda/jmiranda/miniconda/envs/nespreso/bin/python3
CFG=config/argo/config_argo_A_CRPS_z32_heave.json
TAG=acrps_z32_heave_s42
CKPT=saved/acrps_z32_heave/models/NeSPReSO2_ARGO_GoM_A_CRPS_z32_heave_${TAG}_s2/${TAG}_s2/model_best.pth
LOG=saved/log/tmux_A_CRPS_z32_heave.log
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "START $(date -Is) host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
export OMP_NUM_THREADS=8

"$PY" selfcheck.py test_heave_residual_forward_loss test_pca_hetero_phys_decode_and_grad

"$PY" scripts/train_prob_twostage.py \
  -c "$CFG" --prob-mode crps --parent-tag "$TAG" --stage2-stop val_ence

"$PY" eval_run.py \
  -c "$CFG" -r "$CKPT" --split test --out ../reports/eval_A_CRPS_z32_heave_s42.json

"$PY" scripts/eval_acrps_phys.py \
  -c "$CFG" -r "$CKPT" --out ../reports/eval_A_CRPS_z32_heave_cal.json

echo "DONE $(date -Is)"
