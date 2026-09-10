#!/bin/bash
# geoff_res: geoff inputs/loss with residual 4x1024 head (capacity test).
set -euo pipefail
cd /unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project/NeSPReSO2_onTemplate
PY=/conda/jmiranda/miniconda/envs/nespreso/bin/python3
CFG=config/argo/config_argo_geoff_res.json
TAG=geoff_res_s42
CKPT=saved/geoff_res/models/NeSPReSO2_ARGO_GoM_geoff_res_${TAG}_s2/${TAG}_s2/model_best.pth
LOG=saved/log/tmux_geoff_res.log
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "START $(date -Is) host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
export OMP_NUM_THREADS=8

RESUME_ARGS=()
if [[ -n ${RESUME_CKPT:-} ]]; then
  RESUME_ARGS+=(--resume "$RESUME_CKPT")
  echo "RESUME $(date -Is) $RESUME_CKPT"
elif [[ ${STAGE2_ONLY:-} != 1 ]]; then
  "$PY" selfcheck.py test_patch_conv_mlp_residual_mode test_residual_linear_dropout test_heave_ablation_pin_arch_dims
fi
if [[ ${STAGE2_ONLY:-} == 1 ]]; then
  RESUME_ARGS+=(--stage2-only)
  echo "STAGE2_ONLY $(date -Is)"
fi
"$PY" scripts/train_prob_twostage.py \
  -c "$CFG" --prob-mode crps --parent-tag "$TAG" --stage2-stop val_ence \
  "${RESUME_ARGS[@]}"

"$PY" eval_run.py \
  -c "$CFG" -r "$CKPT" --split test --out ../reports/eval_geoff_res_s42.json

"$PY" scripts/eval_acrps_phys.py \
  -c "$CFG" -r "$CKPT" --out ../reports/eval_geoff_res_cal.json

echo "DONE $(date -Is)"
