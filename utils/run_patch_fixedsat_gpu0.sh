#!/usr/bin/env bash
set -eo pipefail

TEMPLATE="/unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project/NeSPReSO2_onTemplate"
LOGDIR="/unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project/utils/logs"
mkdir -p "$LOGDIR"

source ~/.bashrc 2>/dev/null || true
set +u
conda activate nespreso
cd "$TEMPLATE"
export CUDA_VISIBLE_DEVICES=0

PIPE="$LOGDIR/patch_fixedsat_pipeline.log"
echo "Using GPU 0 ($(date))" | tee "$PIPE"

echo "=== RAW patch_l4 ===" | tee -a "$PIPE"
python train.py -c config/argo/config_argo_patch_l4.json --bs 512 -id patch_l4_fixedsat \
  2>&1 | tee "$LOGDIR/patch_l4_fixedsat_train.log"

echo "=== ANOM patch_l4 ===" | tee -a "$PIPE"
python train.py -c config/argo/config_argo_patch_l4_anom.json --bs 512 -id patch_l4_anom_fixedsat \
  2>&1 | tee "$LOGDIR/patch_l4_anom_fixedsat_train.log"

echo "=== EVAL ===" | tee -a "$PIPE"
RAW_CKPT="saved/models/NeSPReSO2_ARGO_GoM_patch_l4/patch_l4_fixedsat/model_best.pth"
ANOM_CKPT="saved/models/NeSPReSO2_ARGO_GoM_patch_l4_anom/patch_l4_anom_fixedsat/model_best.pth"

python eval_run.py -c config/argo/config_argo_patch_l4.json -r "$RAW_CKPT" \
  --split test --out saved/eval_patch_l4_fixedsat_test.json

python eval_run.py -c config/argo/config_argo_patch_l4_anom.json -r "$ANOM_CKPT" \
  --split test --out saved/eval_anom_patch_fixedsat_test.json

python notebooks/run_argo_production_compare.py
python scripts/gom_diagnostics.py --out-dir saved/gom_diagnostics_fixedsat --keys argo_point argo_patch_l4
python scripts/results_table.py

echo "ALL DONE $(date)" | tee -a "$PIPE"
