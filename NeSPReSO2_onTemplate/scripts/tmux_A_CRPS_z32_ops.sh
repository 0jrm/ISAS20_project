#!/bin/bash
# A_CRPS_z32 head + z32 loss on 19-operator inputs (paired 32-PC PCA, not HeaveResidualFast).
set -euo pipefail
cd /unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project/NeSPReSO2_onTemplate
PY=/conda/jmiranda/miniconda/envs/nespreso/bin/python3
CFG=config/argo/config_argo_A_CRPS_z32_ops.json
TAG=acrps_z32_ops_s42
CKPT=saved/acrps_z32_ops/models/NeSPReSO2_ARGO_GoM_A_CRPS_z32_ops_${TAG}_s2/${TAG}_s2/model_best.pth
LOG=saved/log/tmux_A_CRPS_z32_ops.log
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "START $(date -Is) host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
export OMP_NUM_THREADS=8

"$PY" scripts/train_prob_twostage.py \
  -c "$CFG" --prob-mode crps --parent-tag "$TAG" --stage2-stop val_ence

"$PY" eval_run.py \
  -c "$CFG" -r "$CKPT" --split test --out ../reports/eval_A_CRPS_z32_ops_s42.json

"$PY" scripts/eval_acrps_phys.py \
  -c "$CFG" -r "$CKPT" --out ../reports/eval_A_CRPS_z32_ops_cal.json

echo "DONE $(date -Is)"
