#!/usr/bin/env bash
# After satellite HDF5 combine finishes: export full cache + train (tmux).
set -euo pipefail
SESSION="${SESSION:-argo_patch_l4_train}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$ROOT/.." && pwd)"
TEMPLATE="${PROJECT}/NeSPReSO2_onTemplate"
CFG="${TEMPLATE}/config/argo/config_argo_patch_l4.json"
SAT="${PROJECT}/data/NeSPReSO_v2_ARGO_GoM_sat/satellite_NeSPReSO_v2_ARGO_GoM.h5"
LOG="${ROOT}/logs/argo_patch_l4_train.log"
mkdir -p "${ROOT}/logs"

SRUN=""
TRAIN_SRUN=""
if command -v srun >/dev/null 2>&1; then
  SRUN="srun --ntasks=1 --cpus-per-task=8"
  TRAIN_SRUN="srun --ntasks=1 --cpus-per-task=8 --gres=gpu:1"
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already exists. Attach with: tmux attach -t $SESSION"
  exit 0
fi

CMD="cd ${ROOT} && source ~/.bashrc 2>/dev/null; conda activate nespreso && \
echo \"Waiting for ${SAT} ...\" | tee -a ${LOG}; \
while [ ! -f ${SAT} ]; do sleep 120; done; \
echo \"Satellite HDF5 ready. Building cache...\" | tee -a ${LOG}; \
${SRUN} bash -lc \"conda activate nespreso && cd ${TEMPLATE} && python3 preproc/export_argo_l4_cache.py -c ${CFG} --force\" 2>&1 | tee -a ${LOG}; \
echo \"Training...\" | tee -a ${LOG}; \
${TRAIN_SRUN} bash -lc \"conda activate nespreso && cd ${TEMPLATE} && python3 train.py -c ${CFG}\" 2>&1 | tee -a ${LOG}"

tmux new-session -d -s "$SESSION" bash -lc "$CMD"
echo "Started tmux session: $SESSION"
echo "Log: $LOG"
echo "Attach: tmux attach -t $SESSION"
