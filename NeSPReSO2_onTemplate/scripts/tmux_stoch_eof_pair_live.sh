#!/bin/bash
# Unfrozen 9-d encoder + pair head. L_pair + L_direct on the same Argo target.
set -euo pipefail
cd /unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project/NeSPReSO2_onTemplate
PY=/conda/jmiranda/miniconda/envs/nespreso/bin/python3
CFG=config/argo/config_argo_stoch_eof_pair_live.json
TAG=stoch_eof_pair_live_s42
CKPT=saved/stoch_eof_pair_live/models/NeSPReSO2_ARGO_GoM_stoch_eof_pair_live_${TAG}_s2/${TAG}_s2/model_best.pth
LOG=saved/log/tmux_stoch_eof_pair_live.log
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "START $(date -Is) host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
export OMP_NUM_THREADS=8

RESUME_ARGS=()
if [[ -n ${RESUME_CKPT:-} ]]; then
  RESUME_ARGS+=(--resume "$RESUME_CKPT")
  echo "RESUME $(date -Is) $RESUME_CKPT"
elif [[ ${STAGE2_ONLY:-} != 1 ]]; then
  "$PY" selfcheck.py test_pair_table_causal test_pair_dataset_index_is_target test_pair_joint_live test_stoch_eof_recipe
fi
if [[ ${STAGE2_ONLY:-} == 1 ]]; then
  RESUME_ARGS+=(--stage2-only)
  echo "STAGE2_ONLY $(date -Is)"
fi
"$PY" scripts/train_prob_twostage.py \
  -c "$CFG" --prob-mode crps --parent-tag "$TAG" --stage2-stop val_ence \
  "${RESUME_ARGS[@]}"

"$PY" eval_run.py \
  -c "$CFG" -r "$CKPT" --split test --out ../reports/eval_stoch_eof_pair_live_s42.json

"$PY" scripts/eval_acrps_phys.py \
  -c "$CFG" -r "$CKPT" --out ../reports/eval_stoch_eof_pair_live_cal.json

"$PY" - << 'PY'
"""Score the live 9-d encoder path on the same test pairs (not the pair product)."""
import json, sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, ".")
from collections import OrderedDict
from base.util import prepare_device, read_json
from eval_run import raw_profile_rmse
from model.loss import make_loss, uses_native_z_profiles
from parse_config import ConfigParser, validate_config
from train import ensure_cache, set_seed
import model.model as module_arch
import data_loader.data_loaders as module_data

cfg_path = "config/argo/config_argo_stoch_eof_pair_live.json"
ckpt_path = (
    "saved/stoch_eof_pair_live/models/NeSPReSO2_ARGO_GoM_stoch_eof_pair_live_"
    "stoch_eof_pair_live_s42_s2/stoch_eof_pair_live_s42_s2/model_best.pth"
)
cfg_dict = read_json(cfg_path)
validate_config(cfg_dict)
config = ConfigParser(cfg_dict, run_id="")
set_seed(config.config.get("seed", 42))
ensure_cache(config)
device, _ = prepare_device(1)
model = config.init_obj("arch", module_arch).to(device)
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
model.load_state_dict(ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)))
model.eval()
dl = dict(config["data_loader"]["args"])
dl["split"] = "test"
dl["shuffle"] = False
loader = getattr(module_data, config["data_loader"]["type"])(**dl)
loss_cfg = config.config.get("loss_config") or {}
loss_fn = make_loss(
    pca_models=loader.cache.get("pca_models"),
    outputs=OrderedDict(config["outputs"]),
    weights=loader.cache.get("weights"),
    device=device,
    loss_config=loss_cfg,
    loss_scales=config.config.get("loss_scales"),
    pres_levels=loader.cache.get("PRES"),
    true_profiles=loader.cache.get("profiles") if uses_native_z_profiles(loss_cfg) or str(loss_cfg.get("crps_space", "pca")) == "stoch_eof" else None,
    targets=loader.cache.get("targets"),
    train_idx=np.asarray(loader.split_indices["train"], dtype=np.int64),
)
pair_mu, dir_mu, idx_all = [], [], []
with torch.no_grad():
    for data, target, indices in loader:
        data = data.to(device)
        indices = indices.to(device)
        out = model(data)
        mu_p, _ = loss_fn.phys_mu_sigma(out, indices)
        mu_d, _ = loss_fn.phys_mu_sigma(model.last_direct, indices)
        pair_mu.append(mu_p.cpu().numpy())
        dir_mu.append(mu_d.cpu().numpy())
        idx_all.append(indices.cpu().numpy())
pair_mu = np.concatenate(pair_mu)
dir_mu = np.concatenate(dir_mu)
idx = np.concatenate(idx_all)
T = np.asarray(loader.cache["profiles"]["temperature"])
S = np.asarray(loader.cache["profiles"]["salinity"])
yT, yS = (T[:, idx].T, S[:, idx].T) if T.shape[0] != loader.cache["inputs"].shape[0] else (T[idx], S[idx])
nt = pair_mu.shape[1] // 2

def rmse(mu):
    return {
        "T": float(np.sqrt(np.nanmean((mu[:, :nt] - yT) ** 2))),
        "S": float(np.sqrt(np.nanmean((mu[:, nt:] - yS) ** 2))),
    }

payload = {"n": int(idx.size), "pair_product": rmse(pair_mu), "enc_direct_target": rmse(dir_mu)}
out = Path("../reports/eval_stoch_eof_pair_live_direct.json")
out.write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
print("wrote", out)
PY

echo "DONE $(date -Is)"
