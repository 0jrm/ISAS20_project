#!/usr/bin/env bash
# Continue cube-native anchor plan locally (no Slurm). GPU 0, batch sizes tuned for <6 GB VRAM.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/conda/jmiranda/miniconda/envs/nespreso/bin/python
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
POINT_CFG=config/argo/config_argo_point_cube.json
RES_CFG=config/argo/config_argo_residual_cube.json
POINT_RUN=point_cube
RES_RUN=residual_v1
POINT_BS="${POINT_BS:-256}"
RES_BS="${RES_BS:-128}"
POINT_CKPT="saved/models/NeSPReSO2_ARGO_GoM/${POINT_RUN}/model_best.pth"
GOLDEN_CKPT="saved/models/NeSPReSO2_ARGO_GoM/argo16_scales/model_best.pth"
RES_CKPT="saved/models/NeSPReSO2_ARGO_GoM_residual_cube/${RES_RUN}/model_best.pth"
RES_HASH="$("$PY" -c "from base.util import read_json; from preproc.features.export_feature_cache import cube_feature_hash; print(cube_feature_hash(read_json('${RES_CFG}')))")"
RES_CACHE="data/cache/train_ready_${RES_HASH}.pkl"
LOG="${LOG:-/tmp/cube_native_continue.log}"

exec > >(tee -a "$LOG") 2>&1
echo "=== cube-native continue $(date -Is) GPU=${CUDA_VISIBLE_DEVICES} point_bs=${POINT_BS} res_bs=${RES_BS} ==="

POINT_TRAIN_PID=""
if [[ ! -f "$RES_CACHE" ]]; then
  echo "=== export residual cache (${RES_HASH}) ==="
  "$PY" preproc/features/export_feature_cache.py -c "$RES_CFG" --force
else
  echo "=== residual cache present: ${RES_CACHE} ==="
fi

if [[ ! -f "$POINT_CKPT" ]]; then
  if pgrep -f "train.py -c ${POINT_CFG}.*-id ${POINT_RUN}" >/dev/null 2>&1; then
    POINT_TRAIN_PID="$(pgrep -f "train.py -c ${POINT_CFG}.*-id ${POINT_RUN}" | head -1)"
    echo "=== point_cube already training (pid=${POINT_TRAIN_PID}); waiting ==="
  else
    echo "=== train point_cube ==="
    "$PY" train.py -c "$POINT_CFG" --bs "$POINT_BS" -id "$POINT_RUN" &
    POINT_TRAIN_PID="$!"
  fi
else
  echo "=== point_cube checkpoint present: ${POINT_CKPT} ==="
fi

if [[ -n "$POINT_TRAIN_PID" ]]; then
  wait "$POINT_TRAIN_PID"
  echo "=== point_cube test eval ==="
  "$PY" eval_run.py -c "$POINT_CFG" -r "$POINT_CKPT" --split test \
    --out "saved/eval_${POINT_RUN}_test.json"
fi

echo "=== S0b gate ==="
"$PY" -m pytest tests/test_residual_init.py -m s0b_gate -q || {
  echo "WARN: S0b gate failed; continuing anyway for diagnostics"
}

if [[ ! -f "$RES_CKPT" ]]; then
  echo "=== train residual_v1 ==="
  "$PY" train.py -c "$RES_CFG" --bs "$RES_BS" -id "$RES_RUN"
fi

echo "=== M6 eval ==="
"$PY" eval_run.py -c "$RES_CFG" -r "$RES_CKPT" --split test \
  --out "saved/eval_${RES_RUN}_test.json"
"$PY" diagnostics/residual_cube/eval_residual_cube.py \
  -c "$RES_CFG" -r "$RES_CKPT" \
  --point-ckpt "$POINT_CKPT" --golden-ckpt "$GOLDEN_CKPT" \
  --split test --out "saved/eval_${RES_RUN}_interpret.json"

echo "=== M7 ablations ==="
TRAIN_BS="$RES_BS" bash scripts/run_residual_ablations.sh

echo "=== done $(date -Is) ==="
