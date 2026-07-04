#!/usr/bin/env bash
# Launch residual training in tmux on GPU 0 (~6 GiB preflight, bs=256).
set -euo pipefail

ROOT="/unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project/NeSPReSO2_onTemplate"
SESSION="${SESSION:-residual_train}"
RUN_ID="${RUN_ID:-my_run}"
LOGDIR="${ROOT}/../utils/logs"
mkdir -p "$LOGDIR"
LOG="${LOGDIR}/residual_${RUN_ID}.log"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already exists. Attach: tmux attach -t $SESSION"
  exit 0
fi

tmux new-session -d -s "$SESSION" -c "$ROOT" bash -lc "
  source ~/.bashrc 2>/dev/null || true
  set +u
  conda activate nespreso
  export GPU_MODE=local
  export CUDA_VISIBLE_DEVICES=0
  export BATCH_SIZE=256
  export MIN_FREE_MIB=6000
  export SKIP_CACHE=1
  export RUN_ID=${RUN_ID}
  echo 'Starting residual pipeline (GPU 0, bs=256, MIN_FREE_MIB=6000)' | tee '${LOG}'
  utils/run_residual.sh 2>&1 | tee -a '${LOG}'
  echo 'DONE' | tee -a '${LOG}'
  read -p 'Press enter to close...'
"

echo "Launched tmux session: $SESSION"
echo "  attach: tmux attach -t $SESSION"
echo "  log:    $LOG"
