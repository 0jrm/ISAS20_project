#!/usr/bin/env bash
# ponytail: agent can inline this; script avoids shell-quoting pain on launch.
set -euo pipefail
cd "$(dirname "$0")/.."

RUN_ID=$(date +%Y%m%d_%H%M%S)
RUN_ROOT="saved/runs/${RUN_ID}"
mkdir -p "$RUN_ROOT"
echo "$RUN_ID" > "$RUN_ROOT/run_id.txt"

srun --ntasks=1 --cpus-per-task=8 --gres=gpu:1 \
  python3 train.py -c config/isas/config_isas.json -id "${RUN_ID}_isas20" -d 1 \
  > "$RUN_ROOT/isas20.log" 2>&1 &
PID_ISAS=$!

srun --ntasks=1 --cpus-per-task=8 --gres=gpu:1 \
  python3 train.py -c config/argo/config_argo.json -id "${RUN_ID}_argo_v2" -d 2 \
  > "$RUN_ROOT/argo_v2.log" 2>&1 &
PID_ARGO=$!

python3 - "$RUN_ID" "$PID_ISAS" "$PID_ARGO" <<'PY'
import json, sys
from pathlib import Path
run_id, pid_isas, pid_argo = sys.argv[1:4]
run_root = Path("saved/runs") / run_id
manifest = {
    "run_id": run_id,
    "runs": [
        {"tag": "isas20", "config": "config/isas/config_isas.json", "pid": int(pid_isas)},
        {"tag": "argo_v2", "config": "config/argo/config_argo.json", "pid": int(pid_argo)},
    ],
}
(run_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"RUN_ID={run_id} PID_ISAS={pid_isas} PID_ARGO={pid_argo} RUN_ROOT={run_root}")
PY
