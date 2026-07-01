#!/usr/bin/env bash
# Full ARGO L4 patch pipeline: satellite HDF5 → cache → train
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT="$(cd "$ROOT/.." && pwd)"
CFG="${ROOT}/config/argo/config_argo_patch_l4.json"
SRUN=""
if command -v srun >/dev/null 2>&1; then
  SRUN="srun --ntasks=1 --cpus-per-task=8"
fi
TRAIN_SRUN=""
if command -v srun >/dev/null 2>&1; then
  TRAIN_SRUN="srun --ntasks=1 --cpus-per-task=8 --gres=gpu:1"
fi

echo "=== Step 1: ARGO satellite HDF5 (resumable batches under data_path/argo_sat_batches/) ==="
$SRUN bash -lc "conda activate nespreso && python3 ${PROJECT}/utils/generate_argo_satellite_data.py -c ${CFG} --batch-size 100"

echo "=== Step 2: Train cache ==="
$SRUN bash -lc "conda activate nespreso && cd ${ROOT} && python3 preproc/export_argo_l4_cache.py -c ${CFG} --force"

echo "=== Step 3: Train ==="
$TRAIN_SRUN bash -lc "conda activate nespreso && cd ${ROOT} && python3 train.py -c ${CFG}"

echo "=== Done ==="
