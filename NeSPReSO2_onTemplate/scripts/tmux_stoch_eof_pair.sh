#!/bin/bash
# Stochastic EOF + causal neighbor-pair inputs (76-d).
set -euo pipefail
cd /unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project/NeSPReSO2_onTemplate
PY=/conda/jmiranda/miniconda/envs/nespreso/bin/python3
CFG=config/argo/config_argo_stoch_eof_pair.json
TAG=stoch_eof_pair_s42
CKPT=saved/stoch_eof_pair/models/NeSPReSO2_ARGO_GoM_stoch_eof_pair_${TAG}_s2/${TAG}_s2/model_best.pth
LOG=saved/log/tmux_stoch_eof_pair.log
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "START $(date -Is) host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
export OMP_NUM_THREADS=8

RESUME_ARGS=()
if [[ -n ${RESUME_CKPT:-} ]]; then
  RESUME_ARGS+=(--resume "$RESUME_CKPT")
  echo "RESUME $(date -Is) $RESUME_CKPT"
elif [[ ${STAGE2_ONLY:-} != 1 ]]; then
  "$PY" selfcheck.py test_pca_hetero_phys_decode_and_grad test_decode_mu_matches_sklearn_inverse test_stoch_eof_recipe test_pair_table_causal test_pair_dataset_index_is_target
fi
if [[ ${STAGE2_ONLY:-} == 1 ]]; then
  RESUME_ARGS+=(--stage2-only)
  echo "STAGE2_ONLY $(date -Is)"
fi
"$PY" scripts/train_prob_twostage.py \
  -c "$CFG" --prob-mode crps --parent-tag "$TAG" --stage2-stop val_ence \
  "${RESUME_ARGS[@]}"

"$PY" eval_run.py \
  -c "$CFG" -r "$CKPT" --split test --out ../reports/eval_stoch_eof_pair_s42.json

"$PY" scripts/eval_acrps_phys.py \
  -c "$CFG" -r "$CKPT" --out ../reports/eval_stoch_eof_pair_cal.json

"$PY" - << 'PY'
import json, pickle, sys
from pathlib import Path
sys.path.insert(0, ".")
from base.split_utils import build_split_indices
from preproc.pair_table import build_pair_table, persistence_rmse
from base.util import read_json
cfg = read_json("config/argo/config_argo_stoch_eof_pair.json")
path = Path(cfg["data_loader"]["args"]["cache_path"])
if not path.is_absolute():
    path = Path(".").resolve() / path
with open(path, "rb") as f:
    cache = pickle.load(f)
n = cache["inputs"].shape[0]
split = build_split_indices(
    n, cache["JULD"],
    cfg["data_loader"]["args"],
    dataset_tag=cache.get("dataset_tag", "argo_v2"),
    v2_src=cfg["io"].get("v2_src"),
)
pt = build_pair_table(cache, split, k_train=4, k_eval=1)
payload = {
    "n_pairs": {k: int(pt[k].shape[0]) for k in ("train", "val", "test")},
    "test_persistence": persistence_rmse(cache["profiles"], pt["test"]),
    "note": "time-causal earlier Argo copy. Compare to eval_stoch_eof_pair_s42.json raw T RMSE.",
}
out = Path("../reports/eval_stoch_eof_pair_persist.json")
out.write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
print("wrote", out)
PY

echo "DONE $(date -Is)"
