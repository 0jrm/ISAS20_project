#!/bin/bash
# Retrain ANOM-point with loss scales re-derived on the anomaly cache (2026-07-15).
# Controlled comparison vs scratch_0705_204716_anom_point: seed 42 + chronological split are pinned
# in the config, so loss_scales (T 2.0029/S 0.0313 -> 1.3998/0.0240) is the ONLY difference.
# Baselines (chronological test, n=623): ANOM-point 0.680/0.104, point raw 0.537/0.090, golden 0.514/0.083.
# No `set -u`: conda's gdal activation hook references unbound GDAL_DATA.
set -o pipefail

export PATH="/usr/local/bin:/usr/bin:/bin:${PATH:-}"
source /conda/jmiranda/miniconda/etc/profile.d/conda.sh
conda activate nespreso
cd /unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project/NeSPReSO2_onTemplate
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

RUN_ID="${RUN_ID:-retune_0715_anom_point}"
LOG="saved/readiness/retune_${RUN_ID}.log"
mkdir -p saved/readiness

echo "=== retune start $(date) | run_id=${RUN_ID} | GPU=${CUDA_VISIBLE_DEVICES} ===" | tee -a "$LOG"
python3 -c "import json;c=json.load(open('config/argo/config_argo_anom.json'));print('loss_scales in use:',c['loss_scales'])" | tee -a "$LOG"
python3 train.py -c config/argo/config_argo_anom.json -id "$RUN_ID" 2>&1 | tee -a "$LOG"
echo "=== retune EXIT=${PIPESTATUS[0]} $(date) ===" | tee -a "$LOG"
