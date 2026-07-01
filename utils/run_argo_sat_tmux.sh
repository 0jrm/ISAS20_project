#!/usr/bin/env bash
# Launch resumable ARGO satellite HDF5 generation in tmux.
set -euo pipefail
SESSION="${SESSION:-argo_sat_gen}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$ROOT/.." && pwd)"
CFG="${PROJECT}/NeSPReSO2_onTemplate/config/argo/config_argo_patch_l4.json"
LOG="${ROOT}/logs/argo_sat_gen.log"
mkdir -p "${ROOT}/logs"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already exists. Attach with: tmux attach -t $SESSION"
  exit 0
fi

SRUN=""
if command -v srun >/dev/null 2>&1; then
  SRUN="srun --ntasks=1 --cpus-per-task=8"
fi

CMD="cd ${ROOT} && source ~/.bashrc 2>/dev/null; conda activate nespreso && \
${SRUN} python3 generate_argo_satellite_data.py \
  -c ${CFG} --batch-size 100 2>&1 | tee -a ${LOG}"

tmux new-session -d -s "$SESSION" bash -lc "$CMD"
echo "Started tmux session: $SESSION"
echo "Log: $LOG"
echo "Attach: tmux attach -t $SESSION"
