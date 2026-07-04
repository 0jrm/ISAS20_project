#!/usr/bin/env bash
# Wait for GPU memory, train raw + anom L4 patch models, then run evals.
set -eo pipefail

TEMPLATE="/unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project/NeSPReSO2_onTemplate"
LOGDIR="/unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project/utils/logs"
MIN_FREE_MIB="${MIN_FREE_MIB:-20000}"
POLL_SEC="${POLL_SEC:-60}"

source ~/.bashrc 2>/dev/null || true
set +u
conda activate nespreso
set -u
cd "$TEMPLATE"

wait_for_gpu() {
  local tag="$1"
  echo "[$tag] waiting for GPU with >= ${MIN_FREE_MIB} MiB free ($(date))"
  while true; do
    # Pick GPU with most free memory above threshold.
    local pick
    pick=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null \
      | awk -F', ' -v min="$MIN_FREE_MIB" '$2 >= min {print $1, $2}' \
      | sort -k2 -nr | head -1)
    if [[ -n "$pick" ]]; then
      local gpu free
      gpu=$(echo "$pick" | awk '{print $1}')
      free=$(echo "$pick" | awk '{print $2}')
      echo "[$tag] using GPU $gpu (${free} MiB free)"
      echo "$gpu"
      return 0
    fi
    sleep "$POLL_SEC"
  done
}

train_one() {
  local tag="$1" cfg="$2" run_id="$3" log="$4"
  local gpu
  gpu=$(wait_for_gpu "$tag")
  echo "[$tag] training on GPU $gpu ($(date))" | tee -a "$log"
  CUDA_VISIBLE_DEVICES="$gpu" python train.py -c "$cfg" --bs 512 -id "$run_id" 2>&1 | tee -a "$log"
  echo "[$tag] finished $(date)" | tee -a "$log"
}

mkdir -p "$LOGDIR"

# Parallel training in subshells (each waits for its own GPU).
train_one raw config/argo/config_argo_patch_l4.json patch_l4_fixedsat \
  "$LOGDIR/patch_l4_fixedsat_train.log" &
pid_raw=$!

train_one anom config/argo/config_argo_patch_l4_anom.json patch_l4_anom_fixedsat \
  "$LOGDIR/patch_l4_anom_fixedsat_train.log" &
pid_anom=$!

wait "$pid_raw" "$pid_anom"

echo "[eval] both trainings done $(date)" | tee "$LOGDIR/patch_fixedsat_eval.log"

RAW_CKPT="saved/models/NeSPReSO2_ARGO_GoM_patch_l4/patch_l4_fixedsat/model_best.pth"
ANOM_CKPT="saved/models/NeSPReSO2_ARGO_GoM_patch_l4_anom/patch_l4_anom_fixedsat/model_best.pth"

for ck in "$RAW_CKPT" "$ANOM_CKPT"; do
  [[ -f "$ck" ]] || { echo "missing checkpoint: $ck"; exit 1; }
done

python eval_run.py -c config/argo/config_argo_patch_l4.json -r "$RAW_CKPT" \
  --split test --out saved/eval_patch_l4_fixedsat_test.json \
  2>&1 | tee -a "$LOGDIR/patch_fixedsat_eval.log"

python eval_run.py -c config/argo/config_argo_patch_l4_anom.json -r "$ANOM_CKPT" \
  --split test --out saved/eval_anom_patch_fixedsat_test.json \
  2>&1 | tee -a "$LOGDIR/patch_fixedsat_eval.log"

python scripts/eval_baselines.py -c config/argo/config_argo_patch_l4_anom.json \
  --split test --saved-dir saved \
  2>&1 | tee -a "$LOGDIR/patch_fixedsat_eval.log" || true

python notebooks/run_argo_production_compare.py \
  2>&1 | tee -a "$LOGDIR/patch_fixedsat_eval.log"

python scripts/gom_diagnostics.py --out-dir saved/gom_diagnostics_fixedsat \
  --keys argo_point argo_patch_l4 \
  2>&1 | tee -a "$LOGDIR/patch_fixedsat_eval.log"

python scripts/results_table.py \
  2>&1 | tee -a "$LOGDIR/patch_fixedsat_eval.log"

echo "[eval] complete $(date)" | tee -a "$LOGDIR/patch_fixedsat_eval.log"
