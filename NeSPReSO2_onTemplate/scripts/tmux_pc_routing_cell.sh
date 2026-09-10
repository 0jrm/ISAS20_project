#!/bin/bash
# Train one PC-routing cell (two-stage CRPS) then eval.
set -euo pipefail
cd /unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project/NeSPReSO2_onTemplate
PY=/conda/jmiranda/miniconda/envs/nespreso/bin/python3
CFG=${1:?config json}
TAG=${2:?run tag}
GPU=${3:-2}
export CUDA_VISIBLE_DEVICES=$GPU
export OMP_NUM_THREADS=8
mkdir -p saved/log
LOG=saved/log/tmux_pc_${TAG}.log
exec > >(tee -a "$LOG") 2>&1
echo "START $(date -Is) tag=$TAG gpu=$GPU"
"$PY" scripts/train_prob_twostage.py -c "$CFG" --prob-mode crps --parent-tag "$TAG" --stage2-stop val_ence
CKPT=$(find saved -path "*${TAG}_s2*model_best.pth" | tail -1 || true)
if [[ -z "${CKPT}" ]]; then
  CKPT=$(find saved -path "*${TAG}_s2*model_best_train.pth" | tail -1)
fi
echo "CKPT=$CKPT"
"$PY" eval_run.py -c "$CFG" -r "$CKPT" --split test --out "../reports/eval_${TAG}_s42.json"
"$PY" scripts/eval_acrps_phys.py -c "$CFG" -r "$CKPT" --out "../reports/eval_${TAG}_cal.json"
"$PY" scripts/ablate_pc_routing.py append --cell "$TAG" --eval-json "../reports/eval_${TAG}_s42.json" --cal-json "../reports/eval_${TAG}_cal.json"
echo "DONE $(date -Is)"
