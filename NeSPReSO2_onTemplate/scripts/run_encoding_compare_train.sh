#!/usr/bin/env bash
# Encoding-compare training: Phase A (ISAS AE-128) then Phase B (6 surface models in parallel).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

AE_EPOCHS="${AE_EPOCHS:-500}"
SURFACE_EPOCHS="${SURFACE_EPOCHS:-2000}"
SRUN="${SRUN:-srun --ntasks=1 --cpus-per-task=8 --gres=gpu:1}"
MANIFEST="saved/compare_runs/manifest.json"
mkdir -p saved/compare_runs

echo "=== Phase A: train ISAS profile AE dim=128 (${AE_EPOCHS} epochs) ==="
$SRUN python3 scripts/train_profile_ae.py \
  -c config/compare/isas_patch_pca16.json \
  --encoding-dim 128 --arch-tag Autoencoder_dim128 --epochs "$AE_EPOCHS"

echo "=== Phase A2: export AE latents into ISAS point + patch caches ==="
for ae_cfg in config/compare/isas_point_ae128.json config/compare/isas_patch_ae128.json; do
  echo "  export -> $ae_cfg"
  $SRUN python3 scripts/export_ae_latents.py \
    -c "$ae_cfg" \
    --decoder-dir saved/decoders/isas20/Autoencoder_dim128 \
    --target-key ae_targets_dim128 --weight-key ae_weights_dim128
done

COMPARE_KEYS=(
  isas_pt_pca16
  isas_pch_pca16
  isas_pt_ae128
  isas_pch_ae128
  argo_pca15
  argo_pca16
)
COMPARE_CONFIGS=(
  config/compare/isas_point_pca16.json
  config/compare/isas_patch_pca16.json
  config/compare/isas_point_ae128.json
  config/compare/isas_patch_ae128.json
  config/compare/argo_point_pca15.json
  config/compare/argo_point_pca16.json
)

echo "=== Phase B: train ${#COMPARE_KEYS[@]} surface models in parallel (${SURFACE_EPOCHS} epochs each) ==="
pids=()
for i in "${!COMPARE_KEYS[@]}"; do
  key="${COMPARE_KEYS[$i]}"
  cfg="${COMPARE_CONFIGS[$i]}"
  log="saved/compare_runs/${key}.train.log"
  echo "  launching $key -> $cfg (log: $log)"
  $SRUN python3 train.py -c "$cfg" >"$log" 2>&1 &
  pids+=($!)
done

status=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "FAILED: ${COMPARE_KEYS[$i]}" >&2
    status=1
  fi
done

echo "=== Writing manifest: $MANIFEST ==="
python3 scripts/write_compare_manifest.py -o "$MANIFEST"

exit $status
