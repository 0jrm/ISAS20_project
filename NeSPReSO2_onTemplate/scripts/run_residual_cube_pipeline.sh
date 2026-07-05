#!/usr/bin/env bash
# Full residual-cube pipeline: cube build -> point_cube anchor -> cache -> train -> eval (local, no Slurm)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/conda/jmiranda/miniconda/envs/nespreso/bin/python
POINT_CFG=config/argo/config_argo_point_cube.json
CFG=config/argo/config_argo_residual_cube.json
RUN_ID=residual_v1
POINT_RUN_ID=point_cube
# PointAnchoredResidual 41-D + frozen base; keep under ~6 GB VRAM
TRAIN_BS="${TRAIN_BS:-128}"
CUBE_WORKERS="${CUBE_WORKERS:-8}"

POINT_CKPT="saved/models/NeSPReSO2_ARGO_GoM/${POINT_RUN_ID}/model_best.pth"
GOLDEN_CKPT="saved/models/NeSPReSO2_ARGO_GoM/argo16_scales/model_best.pth"

echo "=== M1: build cube ==="
"$PY" preproc/cube/build_cube.py --product all --workers "$CUBE_WORKERS" --resume

echo "=== M2: export point cache ==="
"$PY" preproc/features/export_feature_cache.py -c "$POINT_CFG" --point-only --force

echo "=== M3: train point_cube anchor (local GPU, bs=${TRAIN_BS}) ==="
GPU_MODE=local "$PY" train.py -c "$POINT_CFG" --bs "$TRAIN_BS" -id "$POINT_RUN_ID"

echo "=== M4: export feature cache ==="
"$PY" preproc/features/export_feature_cache.py -c "$CFG" --force

echo "=== M5/M6: train residual_v1 (local GPU, bs=${TRAIN_BS}) ==="
GPU_MODE=local "$PY" train.py -c "$CFG" --bs "$TRAIN_BS" -id "$RUN_ID"

CKPT="saved/models/NeSPReSO2_ARGO_GoM_residual_cube/${RUN_ID}/model_best.pth"

echo "=== M6: eval ==="
"$PY" eval_run.py -c "$CFG" -r "$CKPT" --split test \
  --out "saved/eval_${RUN_ID}_test.json"
"$PY" diagnostics/residual_cube/eval_residual_cube.py \
  -c "$CFG" -r "$CKPT" --point-ckpt "$POINT_CKPT" --golden-ckpt "$GOLDEN_CKPT" --split test \
  --out "saved/eval_${RUN_ID}_interpret.json"

echo "Done."
