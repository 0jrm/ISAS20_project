#!/bin/bash
# Fixed 3×3@1° / 2-day lookback conv: cache → train → test eval. Run in tmux on skynet.
set -euo pipefail
cd /unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project/NeSPReSO2_onTemplate
PY=/conda/jmiranda/miniconda/envs/nespreso/bin/python3
CKPT=saved/models/NeSPReSO2_ARGO_GoM_heave_fast_conv3x3/heave_conv3x3_s42
CACHE=/unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project/data/cache/train_ready_heave_conv_3x3_fix.pkl
LOG=saved/log/tmux_heave_conv3x3.log
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "START $(date -Is) host=$(hostname) job=${SLURM_JOB_ID:-none}"

srun --ntasks=1 --cpus-per-task=8 "$PY" preproc/export_heave_ablation_cache.py \
  --kind conv3 --out "$CACHE"

"$PY" - <<'PY'
import pickle
from pathlib import Path
p = Path("/unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project/data/cache/train_ready_heave_conv_3x3_fix.pkl")
with open(p, "rb") as f:
    c = pickle.load(f)
x = c["inputs"]
assert x.shape == (4145, 9 + 81), x.shape
assert c.get("sat_patch_shape") == [3, 3, 3, 3]
assert c.get("cache_kind") == "heave_conv3"
assert c.get("patch_norm") == "center_rel_channel_train_std"
sat = x[:, 9:].reshape(-1, 3, 3, 3, 3)
assert abs(float(sat[:, :, :, 1, 1].max())) < 1e-5
print("cache_ok", x.shape, "extra_std", c.get("extra_std"))
PY

srun --ntasks=1 --cpus-per-task=8 --gres=gpu:1 "$PY" train.py \
  -c config/argo/config_argo_heave_fast_conv3x3.json \
  -id heave_conv3x3_s42

srun --ntasks=1 --cpus-per-task=8 --gres=gpu:1 "$PY" eval_run.py \
  -c config/argo/config_argo_heave_fast_conv3x3.json \
  -r "$CKPT/model_best.pth" \
  --split test \
  --out ../reports/eval_heave_conv3x3_s42.json
echo "DONE $(date -Is)"
