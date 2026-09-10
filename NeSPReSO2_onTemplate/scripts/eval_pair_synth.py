#!/usr/bin/env python3
"""Swap pair-net source PCs: Argo vs sat-only synthetic. Forward and reverse time."""
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
from eval_run import raw_profile_rmse
from model.loss import make_loss, uses_native_z_profiles
from parse_config import ConfigParser, validate_config
from preproc.pair_table import (
    PAIR_IN_DIM,
    build_pair_table,
    pack_pair_inputs,
    persistence_rmse,
)
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


def _synth_pcs(config, model, cache, device):
    x = torch.as_tensor(np.asarray(cache["inputs"], dtype=np.float32), device=device)
    pcs = []
    with torch.no_grad():
        for i in range(0, x.shape[0], 512):
            out = model(x[i : i + 512])
            d = int(config["arch"]["args"]["output_dim"])
            pcs.append(out[:, :d].cpu().numpy())
    return np.concatenate(pcs, axis=0)


def _score_pair(pair_model, loss_fn, x, target_i, cache, device):
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
    if T.shape[0] == cache["inputs"].shape[0]:
        yT, yS = T[target_i], S[target_i]
    else:
        yT, yS = T[:, target_i].T, S[:, target_i].T
    t = float(np.sqrt(np.nanmean((mu[:, :nt] - yT) ** 2)))
    s = float(np.sqrt(np.nanmean((mu[:, nt:] - yS) ** 2)))
    return {"T": t, "S": s}


def _pack(base, source_pcs, pairs, abs_dt=False):
    x = pack_pair_inputs(base, source_pcs, pairs)
    if abs_dt and pairs.shape[0]:
        x = np.array(x, copy=True)
        x[:, PAIR_IN_DIM - 1] = np.abs(np.asarray(pairs["dt_days"], dtype=np.float32)) / 30.0
    return x


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-c", "--pair-config", default="config/argo/config_argo_stoch_eof_pair.json")
    ap.add_argument(
        "-r",
        "--pair-ckpt",
        default=(
            "saved/stoch_eof_pair/models/NeSPReSO2_ARGO_GoM_stoch_eof_pair_"
            "stoch_eof_pair_s42_s2/stoch_eof_pair_s42_s2/model_best.pth"
        ),
    )
    ap.add_argument("--out", default="../reports/eval_pair_synth_swap.json")
    args = ap.parse_args()

    device, _ = prepare_device(1)
    pair_cfg = args.pair_config
    pair_ckpt = args.pair_ckpt
    sat_cfg = "config/argo/config_argo_stoch_eof.json"
    sat_ckpt = (
        "saved/stoch_eof/models/NeSPReSO2_ARGO_GoM_stoch_eof_"
        "stoch_eof_s42_s2/stoch_eof_s42_s2/model_best.pth"
    )
    pconfig, pmodel = _load(pair_cfg, pair_ckpt, device)
    sconfig, smodel = _load(sat_cfg, sat_ckpt, device)
    import data_loader.data_loaders as module_data

    dl = dict(pconfig["data_loader"]["args"])
    dl["split"] = "test"
    dl["shuffle"] = False
    dl["pair"] = False
    for k in ("k_train", "k_eval", "max_km", "min_dt", "max_dt", "pair_source", "synth_pcs_path"):
        dl.pop(k, None)
    loader = getattr(module_data, pconfig["data_loader"]["type"])(**dl)
    cache = loader.cache
    splits = loader.split_indices
    loss_fn = _loss(pconfig, cache, device, splits)
    synth = _synth_pcs(sconfig, smodel, cache, device)
    argo_pcs = np.asarray(cache["targets"], dtype=np.float32)
    base = np.asarray(cache["inputs"], dtype=np.float32)
    assert synth.shape == argo_pcs.shape

    fwd = build_pair_table(cache, splits, k_train=4, k_eval=1, later=False)["test"]
    rev = build_pair_table(cache, splits, k_train=4, k_eval=1, later=True)["test"]
    rows = {
        "n_forward": int(fwd.shape[0]),
        "n_reverse": int(rev.shape[0]),
        "persist_forward_argo": persistence_rmse(cache["profiles"], fwd),
        "persist_reverse_argo": persistence_rmse(cache["profiles"], rev) if rev.size else None,
    }

    def run(name, pairs, src, abs_dt=False):
        if pairs.shape[0] == 0:
            rows[name] = None
            return
        x = _pack(base, src, pairs, abs_dt=abs_dt)
        rows[name] = _score_pair(pmodel, loss_fn, x, pairs["target_i"], cache, device)

    run("forward_argo", fwd, argo_pcs)
    run("forward_synth", fwd, synth)
    run("reverse_argo", rev, argo_pcs)
    run("reverse_synth", rev, synth)
    run("reverse_synth_absdt", rev, synth, abs_dt=True)

    test_idx = np.asarray(splits["test"], dtype=int)
    sat_only = raw_profile_rmse(
        synth[test_idx],
        cache["profiles"],
        cache["pca_models"],
        OrderedDict(pconfig["outputs"]),
        test_idx,
    )
    rows["sat_only_stoch_eof_on_test"] = sat_only
    rows["note"] = (
        "forward: earlier neighbor. reverse: later neighbor, signed dt (OOD) "
        "or absdt. synth = 9-d stoch_eof mu PCs. No retrain."
    )
    out = Path(args.out)
    if not out.is_absolute():
        out = _ROOT / out
    out.write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
