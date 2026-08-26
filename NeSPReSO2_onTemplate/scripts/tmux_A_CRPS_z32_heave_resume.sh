#!/bin/bash
# Resume heave stage-2 after TensorBoard histogram OOM. Same run dir.
set -euo pipefail
cd /unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project/NeSPReSO2_onTemplate
PY=/conda/jmiranda/miniconda/envs/nespreso/bin/python3
CFG=config/argo/config_argo_A_CRPS_z32_heave.json
CKPT_S2=saved/acrps_z32_heave/models/NeSPReSO2_ARGO_GoM_A_CRPS_z32_heave_acrps_z32_heave_s42_s2/acrps_z32_heave_s42_s2/checkpoint.pth
BEST=saved/acrps_z32_heave/models/NeSPReSO2_ARGO_GoM_A_CRPS_z32_heave_acrps_z32_heave_s42_s2/acrps_z32_heave_s42_s2/model_best.pth
LOG=saved/log/tmux_A_CRPS_z32_heave.log
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "RESUME_S2 $(date -Is) host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
export OMP_NUM_THREADS=8

"$PY" train.py -r "$CKPT_S2"

"$PY" eval_run.py \
  -c "$CFG" -r "$BEST" --split test --out ../reports/eval_A_CRPS_z32_heave_s42.json

"$PY" scripts/eval_acrps_phys.py \
  -c "$CFG" -r "$BEST" --out ../reports/eval_A_CRPS_z32_heave_cal.json

echo "DONE $(date -Is)"
