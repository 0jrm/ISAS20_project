#!/bin/bash
# Frozen-checkpoint permutation importance (time/space/sat/enso/ops).
set -euo pipefail
cd /unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project/NeSPReSO2_onTemplate
PY=/conda/jmiranda/miniconda/envs/nespreso/bin/python3
LOG=saved/log/tmux_ablate_inputs.log
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "START $(date -Is) host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
export OMP_NUM_THREADS=8

"$PY" scripts/ablate_inputs.py \
  -c config/argo/config_argo_stoch_eof.json \
  -r saved/stoch_eof/models/NeSPReSO2_ARGO_GoM_stoch_eof_stoch_eof_s42_s2/stoch_eof_s42_s2/model_best.pth \
  --out ../reports/eval_input_ablation_stoch_eof.json

"$PY" scripts/ablate_inputs.py \
  -c config/argo/config_argo_geoff.json \
  -r saved/geoff/models/NeSPReSO2_ARGO_GoM_geoff_geoff_s42_s2/geoff_s42_s2/model_best.pth \
  --out ../reports/eval_input_ablation_geoff.json

"$PY" scripts/ablate_inputs.py \
  -c config/argo/config_argo_A_CRPS_z32_ops.json \
  -r saved/acrps_z32_ops/models/NeSPReSO2_ARGO_GoM_A_CRPS_z32_ops_acrps_z32_ops_s42_s2/acrps_z32_ops_s42_s2/model_best.pth \
  --out ../reports/eval_input_ablation_z32_ops.json

echo "DONE $(date -Is)"
