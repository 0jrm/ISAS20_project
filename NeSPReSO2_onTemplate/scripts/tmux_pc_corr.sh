#!/usr/bin/env bash
# Run PC–input correlation analysis
SESSION="pc_corr"
tmux new-session -d -s "$SESSION" 2>/dev/null || true
tmux send-keys -t "$SESSION" "cd /unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project/NeSPReSO2_onTemplate && conda activate nespreso && srun --ntasks=1 --cpus-per-task=8 python3 scripts/pc_input_correlation.py 2>&1 | tee /tmp/pc_corr.log && echo DONE" Enter
echo "Running in tmux session: $SESSION"
echo "Monitor: tmux attach -t $SESSION"
echo "Or: tail -f /tmp/pc_corr.log"
