#!/usr/bin/env bash
# Staged training for point-anchored residual patch model.
#
# GPU scheduling on Skynet (bfs-v13-skynet):
#   - Slurm tracks 3 GPUs (gres/gpu:a100:3), not all 4 physical devices.
#   - `srun --gres=gpu:1` queues when those 3 are allocated to other jobs.
#   - GPU_MODE=local skips srun and uses CUDA_VISIBLE_DEVICES directly (like
#     run_patch_fixedsat_gpu0.sh) when you are already on the node.
#   - GPU_MODE=srun always requests a Slurm GPU (may queue).
#   - GPU_MODE=auto (default): local on bfs-v13-skynet, else srun.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BASE_CFG="config/argo/config_argo_residual.json"
STAGE3_CFG="config/argo/config_argo_residual_stage3.json"
STAGE4_CFG="config/argo/config_argo_residual_stage4.json"
RUN_ID="${RUN_ID:-residual_run}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
GPU_MODE="${GPU_MODE:-auto}"
BATCH_SIZE="${BATCH_SIZE:-256}"
MIN_FREE_MIB="${MIN_FREE_MIB:-6000}"
SKIP_CACHE="${SKIP_CACHE:-0}"

cpu_srun() {
  srun --ntasks=1 --cpus-per-task=8 "$@"
}

gpu_preflight() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "WARN: nvidia-smi not found; skipping GPU memory preflight" >&2
    return 0
  fi
  local free_mib
  free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')"
  echo "GPU ${GPU}: ${free_mib} MiB free (need >= ${MIN_FREE_MIB} MiB for training)"
  if [[ "${free_mib}" -lt "${MIN_FREE_MIB}" ]]; then
    echo "ERROR: GPU ${GPU} has only ${free_mib} MiB free." >&2
    echo "  Slurm GPUs (a100:0-2) may be held by other jobs: squeue -p gpu" >&2
    echo "  Physical GPU 0 is often used by non-Slurm jobs (e.g. VLLM)." >&2
    echo "  Options: wait for queue (GPU_MODE=srun), pick another GPU, or lower --bs." >&2
    return 1
  fi
}

resolve_gpu_mode() {
  case "$GPU_MODE" in
    local|srun) echo "$GPU_MODE" ;;
    auto)
      if [[ "${HOSTNAME:-}" == bfs-v13-skynet* ]] || [[ -n "${SLURMD_NODENAME:-}" ]]; then
        echo local
      else
        echo srun
      fi
      ;;
    *)
      echo "Unknown GPU_MODE=${GPU_MODE} (use auto|local|srun)" >&2
      return 1
      ;;
  esac
}

run_train() {
  local mode
  mode="$(resolve_gpu_mode)"
  if [[ "$mode" == local ]]; then
    gpu_preflight
    echo "Training on GPU ${GPU} (GPU_MODE=local, no srun allocation)"
    env CUDA_VISIBLE_DEVICES="$GPU" python3 train.py "$@"
  else
    echo "Training via srun --gres=gpu:a100:1 (may queue if GPUs are busy)"
    srun --ntasks=1 --cpus-per-task=8 --gres=gpu:a100:1 \
      env CUDA_VISIBLE_DEVICES=0 python3 train.py "$@"
  fi
}

echo "== Stage 1: reuse golden point checkpoint (warmstart in config) =="

echo "== Stage 2: build cache + verify baseline at epoch 0 =="
if [[ "$SKIP_CACHE" == "1" ]]; then
  echo "SKIP_CACHE=1: using existing cache"
else
  cpu_srun python3 preproc/export_argo_residual_cache.py -c "$BASE_CFG" --force
fi

echo "== Stage 3: train residual branch only (freeze base, bs=${BATCH_SIZE}) =="
run_train -c "$STAGE3_CFG" --bs "$BATCH_SIZE" -id "${RUN_ID}_stage3"

echo "== Stage 4: optional joint fine-tune (bs=${BATCH_SIZE}) =="
STAGE3_CKPT="saved/models/NeSPReSO2_ARGO_GoM_residual/${RUN_ID}_stage3/model_best.pth"
run_train -c "$STAGE4_CFG" -r "$STAGE3_CKPT" --bs "$BATCH_SIZE" -id "${RUN_ID}_stage4"

echo "== Evaluation =="
cpu_srun python3 scripts/eval_residual.py \
  -c "$BASE_CFG" \
  -r "saved/models/NeSPReSO2_ARGO_GoM_residual/${RUN_ID}_stage4/model_best.pth" \
  --point-ckpt saved/models/NeSPReSO2_ARGO_GoM/argo16_scales/model_best.pth \
  --split test \
  --out "saved/eval_residual_${RUN_ID}_test.json"

cpu_srun python3 scripts/residual_diagnostics.py \
  -c "$BASE_CFG" \
  -r "saved/models/NeSPReSO2_ARGO_GoM_residual/${RUN_ID}_stage4/model_best.pth" \
  --out-dir "saved/residual_diagnostics/${RUN_ID}"
