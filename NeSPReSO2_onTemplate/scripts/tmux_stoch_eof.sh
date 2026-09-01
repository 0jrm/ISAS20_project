#!/bin/bash
# Stochastic EOF Emulator: physical CRPS vs raw profiles, truncation floor, whitened PC term.
set -euo pipefail
cd /unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project/NeSPReSO2_onTemplate
PY=/conda/jmiranda/miniconda/envs/nespreso/bin/python3
CFG=config/argo/config_argo_stoch_eof.json
TAG=stoch_eof_s42
CKPT=saved/stoch_eof/models/NeSPReSO2_ARGO_GoM_stoch_eof_${TAG}_s2/${TAG}_s2/model_best.pth
LOG=saved/log/tmux_stoch_eof.log
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "START $(date -Is) host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
export OMP_NUM_THREADS=8

RESUME_ARGS=()
if [[ -n ${RESUME_CKPT:-} ]]; then
  RESUME_ARGS+=(--resume "$RESUME_CKPT")
  echo "RESUME $(date -Is) $RESUME_CKPT"
elif [[ ${STAGE2_ONLY:-} != 1 ]]; then
  "$PY" selfcheck.py test_pca_hetero_phys_decode_and_grad test_decode_mu_matches_sklearn_inverse test_stoch_eof_recipe
fi
if [[ ${STAGE2_ONLY:-} == 1 ]]; then
  RESUME_ARGS+=(--stage2-only)
  echo "STAGE2_ONLY $(date -Is)"
fi
"$PY" scripts/train_prob_twostage.py \
  -c "$CFG" --prob-mode crps --parent-tag "$TAG" --stage2-stop val_ence \
  "${RESUME_ARGS[@]}"

"$PY" eval_run.py \
  -c "$CFG" -r "$CKPT" --split test --out ../reports/eval_stoch_eof_s42.json

"$PY" scripts/eval_acrps_phys.py \
  -c "$CFG" -r "$CKPT" --out ../reports/eval_stoch_eof_cal.json

echo "DONE $(date -Is)"
