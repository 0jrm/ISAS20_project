#!/usr/bin/env bash
# Train cube-interpolated point baseline (point_cube anchor) and record test RMSE (local, no Slurm)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/conda/jmiranda/miniconda/envs/nespreso/bin/python
CFG=config/argo/config_argo_point_cube.json
RUN_ID=point_cube
TRAIN_BS="${TRAIN_BS:-256}"  # PatchConvMLP 9-D; fits ~6 GB VRAM at 256

echo "=== export point cache ==="
"$PY" preproc/features/export_feature_cache.py -c "$CFG" --point-only --force

echo "=== train point_cube (local GPU, bs=${TRAIN_BS}) ==="
GPU_MODE=local "$PY" train.py -c "$CFG" --bs "$TRAIN_BS" -id "$RUN_ID"

CKPT="saved/models/NeSPReSO2_ARGO_GoM/${RUN_ID}/model_best.pth"

echo "=== eval test RMSE ==="
"$PY" eval_run.py -c "$CFG" -r "$CKPT" --split test \
  --out "saved/eval_${RUN_ID}_test.json"

echo "Done."
