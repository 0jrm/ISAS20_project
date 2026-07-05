#!/usr/bin/env bash
# Run residual-cube ablation configs and summarize RMSE into saved/results/ (local, no Slurm)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/conda/jmiranda/miniconda/envs/nespreso/bin/python
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
RESULTS=saved/results/residual_cube_ablations
TRAIN_BS="${TRAIN_BS:-128}"
RETRAIN="${RETRAIN:-1}"
POINT_CKPT="saved/models/NeSPReSO2_ARGO_GoM/point_cube/model_best.pth"
GOLDEN_CKPT="saved/models/NeSPReSO2_ARGO_GoM/argo16_scales/model_best.pth"
mkdir -p "$RESULTS"

run_one() {
  local cfg="$1"
  local tag="$2"
  echo "=== $tag ==="
  "$PY" preproc/features/export_feature_cache.py -c "$cfg" --force
  local model_dir="saved/models/NeSPReSO2_ARGO_GoM_residual_cube/${tag}"
  local log_dir="saved/log/NeSPReSO2_ARGO_GoM_residual_cube/${tag}"
  if [[ "$RETRAIN" == "1" ]]; then
    rm -rf "$model_dir" "$log_dir"
  fi
  "$PY" train.py -c "$cfg" --bs "$TRAIN_BS" -id "${tag}"
  local ckpt="${model_dir}/model_best.pth"
  "$PY" eval_run.py -c "$cfg" -r "$ckpt" --split test \
    --out "$RESULTS/${tag}_test.json"
  "$PY" diagnostics/residual_cube/eval_residual_cube.py \
    -c "$cfg" -r "$ckpt" \
    --point-ckpt "$POINT_CKPT" \
    --golden-ckpt "$GOLDEN_CKPT" \
    --split test --out "$RESULTS/${tag}_interpret.json"
}

run_one config/argo/ablations/config_residual_drop_sss_grads.json drop_sss_grads
run_one config/argo/ablations/config_residual_local_only.json local_only
run_one config/argo/ablations/config_residual_gate_scalar.json gate_scalar
run_one config/argo/ablations/config_residual_sigma_2x.json sigma_2x

"$PY" - <<'PY'
import json
from pathlib import Path

results = Path("saved/results/residual_cube_ablations")
tags = ["drop_sss_grads", "local_only", "gate_scalar", "sigma_2x"]
rows = []
for tag in tags:
    test = json.loads((results / f"{tag}_test.json").read_text())
    interp = json.loads((results / f"{tag}_interpret.json").read_text())
    rows.append({
        "tag": tag,
        "residual_rmse_T": test["raw_profile_rmse"]["temperature"],
        "residual_rmse_S": test["raw_profile_rmse"]["salinity"],
        "point_cube_rmse_T": interp["raw_profile_rmse_point"]["temperature"],
        "point_cube_rmse_S": interp["raw_profile_rmse_point"]["salinity"],
        "golden_rmse_T": interp.get("raw_profile_rmse_golden", {}).get("temperature"),
        "golden_rmse_S": interp.get("raw_profile_rmse_golden", {}).get("salinity"),
        "s3a_delta_rmse_ci95": interp["s3a_paired_significance"]["delta_rmse_ci95"],
    })
out = results / "ablation_summary.json"
out.write_text(json.dumps({"ablations": rows}, indent=2) + "\n")
print(f"wrote {out}")
PY

echo "Ablation outputs in $RESULTS"
