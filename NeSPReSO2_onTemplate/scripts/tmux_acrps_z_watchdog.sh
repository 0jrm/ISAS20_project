#!/bin/bash
# If a two-stage train dies without DONE, respawn tmux pane from latest checkpoint.
# Does not start a second job while the original wrapper / train.py is alive.
set -u
cd /unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project/NeSPReSO2_onTemplate
WLOG=saved/log/tmux_acrps_z_watchdog.log
mkdir -p saved/log
exec >>"$WLOG" 2>&1

job_alive() {
  local cfg_json=$1 tag=$2 wrap=$3
  pgrep -f "${wrap}" >/dev/null && return 0
  pgrep -f "train_prob_twostage.py -c config/argo/${cfg_json}" >/dev/null && return 0
  pgrep -f "train.py .*-id ${tag}_s" >/dev/null && return 0
  pgrep -f "eval_run.py -c config/argo/${cfg_json}" >/dev/null && return 0
  pgrep -f "eval_acrps_phys.py -c config/argo/${cfg_json}" >/dev/null && return 0
  return 1
}

respawn() {
  local sess=$1 gpu=$2 wrap=$3 envextra=$4
  local cmd="export CUDA_VISIBLE_DEVICES=${gpu} OMP_NUM_THREADS=8 ${envextra}; bash ${wrap}; echo EXIT \$?"
  echo "WATCH_RESPAWN $(date -Is) sess=$sess gpu=$gpu ${envextra}"
  if tmux has-session -t "$sess" 2>/dev/null; then
    tmux respawn-pane -k -t "${sess}:0.0" -- bash -lc "$cmd"
  else
    tmux new-session -d -s "$sess" "$cmd"
    tmux set-option -t "${sess}:0" remain-on-exit on
  fi
}

echo "WATCH_START $(date -Is) host=$(hostname) pid=$$"

while true; do
  sleep 90
  while read -r sess gpu wrap cfg_json tag log s1ckpt s2ckpt; do
    [[ -z ${sess:-} ]] && continue
    grep -q "^DONE " "$log" 2>/dev/null && continue
    job_alive "$cfg_json" "$tag" "$wrap" && continue
    s2best="${s2ckpt%checkpoint.pth}model_best.pth"
    s1best="${s1ckpt%checkpoint.pth}model_best.pth"
    extra=""
    if [[ -f $s2best ]]; then
      extra="RESUME_CKPT=${s2ckpt}"
    elif [[ -f $s1best ]]; then
      extra="STAGE2_ONLY=1"
    else
      echo "WATCH_NO_CKPT $(date -Is) sess=$sess — skip"
      continue
    fi
    echo "WATCH_DEAD $(date -Is) sess=$sess — $extra"
    respawn "$sess" "$gpu" "$wrap" "$extra"
  done <<'JOBS'
acrps_z 0 scripts/tmux_A_CRPS_z.sh config_argo_A_CRPS_z.json acrps_z_s42 saved/log/tmux_A_CRPS_z.log saved/acrps_z/models/NeSPReSO2_ARGO_GoM_A_CRPS_z_acrps_z_s42_s1/acrps_z_s42_s1/checkpoint.pth saved/acrps_z/models/NeSPReSO2_ARGO_GoM_A_CRPS_z_acrps_z_s42_s2/acrps_z_s42_s2/checkpoint.pth
acrps_z_roni_ops 1 scripts/tmux_A_CRPS_z_roni_ops.sh config_argo_A_CRPS_z_roni_ops.json acrps_z_roni_ops_s42 saved/log/tmux_A_CRPS_z_roni_ops.log saved/acrps_z_roni_ops/models/NeSPReSO2_ARGO_GoM_A_CRPS_z_roni_ops_acrps_z_roni_ops_s42_s1/acrps_z_roni_ops_s42_s1/checkpoint.pth saved/acrps_z_roni_ops/models/NeSPReSO2_ARGO_GoM_A_CRPS_z_roni_ops_acrps_z_roni_ops_s42_s2/acrps_z_roni_ops_s42_s2/checkpoint.pth
JOBS
done
