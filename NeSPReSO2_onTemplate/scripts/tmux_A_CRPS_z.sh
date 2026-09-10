#!/bin/bash
# A_CRPS recipe in native z (no PCA). Classic 9-d SST/SSS/SSH.
set -euo pipefail
cd /unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project/NeSPReSO2_onTemplate
PY=/conda/jmiranda/miniconda/envs/nespreso/bin/python3
CFG=config/argo/config_argo_A_CRPS_z.json
TAG=acrps_z_s42
CKPT=saved/acrps_z/models/NeSPReSO2_ARGO_GoM_A_CRPS_z_${TAG}_s2/${TAG}_s2/model_best.pth
LOG=saved/log/tmux_A_CRPS_z.log
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "START $(date -Is) host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
export OMP_NUM_THREADS=8

RESUME_ARGS=()
if [[ -n ${RESUME_CKPT:-} ]]; then
  RESUME_ARGS+=(--resume "$RESUME_CKPT")
  echo "RESUME $(date -Is) $RESUME_CKPT"
elif [[ ${STAGE2_ONLY:-} != 1 ]]; then
  "$PY" selfcheck.py test_pca_hetero_zspace_identity test_pca_hetero_phys_decode_and_grad test_acrps_z_configs_validate test_resume_monitor_reset_on_run_name_change
fi
if [[ ${STAGE2_ONLY:-} == 1 ]]; then
  RESUME_ARGS+=(--stage2-only)
  echo "STAGE2_ONLY $(date -Is)"
fi
"$PY" scripts/train_prob_twostage.py \
  -c "$CFG" --prob-mode crps --parent-tag "$TAG" --stage2-stop val_ence \
  "${RESUME_ARGS[@]}"

"$PY" eval_run.py \
  -c "$CFG" -r "$CKPT" --split test --out ../reports/eval_A_CRPS_z_s42.json

"$PY" scripts/eval_acrps_phys.py \
  -c "$CFG" -r "$CKPT" --out ../reports/eval_A_CRPS_z_cal.json

echo "DONE $(date -Is)"
