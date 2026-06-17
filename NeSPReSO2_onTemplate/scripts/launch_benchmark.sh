#!/usr/bin/env bash
# Run ML optimization benchmarks (single-GPU sweep + optional DDP).
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${1:-config_isas.json}"
OUT_DIR="saved/benchmarks"
mkdir -p "$OUT_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
OUT_JSON="$OUT_DIR/ml_opts_${STAMP}.json"

echo "=== Single-GPU variant sweep ($CONFIG) ==="
srun --ntasks=1 --cpus-per-task=8 --gres=gpu:1 \
  python3 scripts/benchmark_ml_opts.py -c "$CONFIG" --profile --out "$OUT_JSON"

if command -v torchrun >/dev/null 2>&1; then
  echo "=== DDP 2-GPU variant ($CONFIG) ==="
  DDP_JSON="$OUT_DIR/ml_opts_ddp_${STAMP}.json"
  srun --ntasks=1 --cpus-per-task=8 --gres=gpu:2 \
    torchrun --nproc_per_node=2 scripts/benchmark_ml_opts.py \
    -c "$CONFIG" --variant ddp --out "$DDP_JSON" || echo "DDP benchmark skipped (need 2 GPUs)"
else
  echo "torchrun not found; skipping DDP benchmark"
fi

echo "Results: $OUT_JSON"
