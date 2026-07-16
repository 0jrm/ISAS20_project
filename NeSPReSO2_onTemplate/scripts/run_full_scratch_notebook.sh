#!/bin/bash
# Execute notebooks/full_scratch_all_models.ipynb headlessly (tmux session: scratch_nb).
# No `set -u`: conda's gdal activation hook references unbound GDAL_DATA.
set -o pipefail

export PATH="/usr/local/bin:/usr/bin:/bin:${PATH:-}"
source /conda/jmiranda/miniconda/etc/profile.d/conda.sh
conda activate nespreso
cd /unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project/NeSPReSO2_onTemplate
export CUDA_VISIBLE_DEVICES=0

LOG=notebooks/scratch_outputs/papermill_run.log
mkdir -p notebooks/scratch_outputs

echo "=== papermill start $(date) ===" | tee -a "$LOG"
python -m papermill notebooks/full_scratch_all_models.ipynb \
  notebooks/_executed_full_scratch_all_models.ipynb \
  --log-output --request-save-on-cell-execute --cwd notebooks \
  2>&1 | tee -a "$LOG"
echo "=== papermill EXIT=${PIPESTATUS[0]} $(date) ===" | tee -a "$LOG"
