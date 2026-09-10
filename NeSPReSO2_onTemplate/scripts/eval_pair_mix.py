#!/usr/bin/env python3
"""Score a mix-trained pair net on nearest-1 test: Argo source, synth source, 50/50 pool."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from collections import OrderedDict

from base.split_utils import build_split_indices
from base.util import prepare_device, read_json
from model.loss import make_loss, uses_native_z_profiles
from parse_config import ConfigParser, validate_config
from preproc.pair_table import MIX_IN_DIM, PAIR_IN_DIM, build_pair_table, pack_pair_inputs
from train import ensure_cache, set_seed


def _load(cfg_path, ckpt_path, device):
    import model.model as module_arch

    cfg_dict = read_json(cfg_path)
    validate_config(cfg_dict)
    config = ConfigParser(cfg_dict, run_id="")
    set_seed(config.config.get("seed", 42))
    ensure_cache(config)
    model = config.init_obj("arch", module_arch).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)))
    model.eval()
    return config, model


def _loss(config, cache, device, split_indices):
    loss_cfg = config.config.get("loss_config") or {}
    train_idx = np.asarray(split_indices["train"], dtype=np.int64)
    return make_loss(
        pca_models=cache.get("pca_models"),
        outputs=OrderedDict(config["outputs"]),
        weights=cache.get("weights"),
        device=device,
        loss_config=loss_cfg,
        loss_scales=config.config.get("loss_scales"),
        pres_levels=cache.get("PRES"),
        true_profiles=(
            cache.get("profiles")
            if uses_native_z_profiles(loss_cfg) or str(loss_cfg.get("crps_space", "pca")) == "stoch_eof"
            else None
        ),
        targets=cache.get("targets"),
        train_idx=train_idx,
    )


def _with_flag(x76, flag: float) -> np.ndarray:
    n = x76.shape[0]
    x = np.empty((n, MIX_IN_DIM), dtype=np.float32)
    x[:, :PAIR_IN_DIM] = x76
    x[:, PAIR_IN_DIM] = flag
    return x


def _score(pair_model, loss_fn, x, target_i, cache, device):
    mus = []
    with torch.no_grad():
        for i in range(0, x.shape[0], 512):
            xb = torch.as_tensor(x[i : i + 512], device=device)
            idx = torch.as_tensor(target_i[i : i + 512], device=device)
            out = pair_model(xb)
            mu, _ = loss_fn.phys_mu_sigma(out, idx)
            mus.append(mu.cpu().numpy())
    mu = np.concatenate(mus, axis=0)
    nt = mu.shape[1] // 2
    T = np.asarray(cache["profiles"]["temperature"])
    S = np.asarray(cache["profiles"]["salinity"])
    yT, yS = (T[:, target_i].T, S[:, target_i].T) if T.shape[0] != cache["inputs"].shape[0] else (T[target_i], S[target_i])
    return {
        "T": float(np.sqrt(np.nanmean((mu[:, :nt] - yT) ** 2))),
        "S": float(np.sqrt(np.nanmean((mu[:, nt:] - yS) ** 2))),
    }


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-c", "--config", default="config/argo/config_argo_stoch_eof_pair_mix.json")
    ap.add_argument(
        "-r",
        "--checkpoint",
        default=(
            "saved/stoch_eof_pair_mix/models/NeSPReSO2_ARGO_GoM_stoch_eof_pair_mix_"
            "stoch_eof_pair_mix_s42_s2/stoch_eof_pair_mix_s42_s2/model_best.pth"
        ),
    )
    ap.add_argument("--out", default="../reports/eval_pair_mix_heads.json")
    args = ap.parse_args()

    device, _ = prepare_device(1)
    config, model = _load(args.config, args.checkpoint, device)
    import data_loader.data_loaders as module_data

    dl = dict(config["data_loader"]["args"])
    dl["split"] = "test"
    dl["shuffle"] = False
    dl["pair"] = False
    for k in ("k_train", "k_eval", "max_km", "min_dt", "max_dt", "pair_source", "synth_pcs_path", "sample_train"):
        dl.pop(k, None)
    loader = getattr(module_data, config["data_loader"]["type"])(**dl)
    cache = loader.cache
    splits = loader.split_indices
    loss_fn = _loss(config, cache, device, splits)
    synth = np.load(_ROOT / Path(config["data_loader"]["args"]["synth_pcs_path"]))
    argo = np.asarray(cache["targets"], dtype=np.float32)
    base = np.asarray(cache["inputs"], dtype=np.float32)
    fwd = build_pair_table(cache, splits, k_train=8, k_eval=1, sample_train="nearest")["test"]
    ti = fwd["target_i"]
    xa = _with_flag(pack_pair_inputs(base, argo, fwd), 1.0)
    xs = _with_flag(pack_pair_inputs(base, synth, fwd), -1.0)
    xm = np.concatenate([xa, xs], axis=0)
    tm = np.concatenate([ti, ti])
    rows = {
        "n_test_pairs": int(fwd.shape[0]),
        "argo_flag_p1": _score(model, loss_fn, xa, ti, cache, device),
        "synth_flag_m1": _score(model, loss_fn, xs, ti, cache, device),
        "pool_50_50": _score(model, loss_fn, xm, tm, cache, device),
    }
    out = Path(args.out)
    if not out.is_absolute():
        out = _ROOT / out
    out.write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
