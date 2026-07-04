#!/usr/bin/env python3
"""Evaluate residual patch model with paired stats vs point baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import data_loader.data_loaders as module_data
import model.model as module_arch
from base.split_utils import build_split_indices
from eval_run import _resolve_eval_batch_size, raw_profile_rmse, set_seed
from model.loss import reconstruct_physical_profiles
from parse_config import ConfigParser
from train import ensure_cache


def _per_profile_rmse(pred_pcs, true_profiles, pca_models, outputs, indices, *, clim_profiles=None):
    idx = np.asarray(indices, dtype=int)
    pred = reconstruct_physical_profiles(pred_pcs, pca_models, outputs, clim_profiles=clim_profiles, indices=idx)
    out = {}
    for name in outputs:
        diff = pred[name] - true_profiles[name][:, idx]
        rmse = np.sqrt(np.nanmean(diff ** 2, axis=0))
        out[name] = rmse.astype(np.float64)
    return out


from residual_utils.stats import paired_stats as _paired_stats
    dl_args = dict(config["data_loader"]["args"])
    dl_args["split"] = split
    dl_args["shuffle"] = False
    _resolve_eval_batch_size(dl_args, split)
    loader = getattr(module_data, config["data_loader"]["type"])(**dl_args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = config.init_obj("arch", module_arch).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
    model.load_state_dict(state)
    model.eval()

    preds = []
    indices = []
    with torch.no_grad():
        for data, _target, idx in loader:
            out = model(data.to(device)).cpu().numpy()
            preds.append(out)
            indices.append(idx.numpy())
    pred_pcs = np.vstack(preds)
    all_idx = np.concatenate(indices)
    return pred_pcs, all_idx, loader


def main():
    parser = argparse.ArgumentParser(description="Residual model eval with paired stats")
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-r", "--resume", required=True)
    parser.add_argument("--point-ckpt", required=True, help="Golden point model checkpoint")
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", default=None)
    parser.add_argument("-d", "--device", default=None)
    args = parser.parse_args()

    if args.device is not None:
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

    config = ConfigParser.from_args(
        argparse.Namespace(
            config=args.config,
            resume=None,
            device=None,
            run_id=None,
            learning_rate=None,
            batch_size=None,
            log_interval=None,
        )
    )
    set_seed(config.config.get("seed", 42))
    ensure_cache(config)

    residual_pcs, idx, loader = _run_model(config, args.resume, args.split)

    point_cfg_path = ROOT / "config/argo/config_argo.json"
    point_config = ConfigParser.from_args(
        argparse.Namespace(
            config=str(point_cfg_path),
            resume=None,
            device=None,
            run_id=None,
            learning_rate=None,
            batch_size=None,
            log_interval=None,
        )
    )
    ensure_cache(point_config)
    point_pcs, point_idx, point_loader = _run_model(point_config, args.point_ckpt, args.split)

    if not np.array_equal(idx, point_idx):
        raise ValueError("Residual and point splits/index order differ")

    outputs = loader.outputs
    pca_models = loader.pca_models
    clim_profiles = loader.cache.get("clim_profiles")
    profiles = loader.profiles

    residual_overall = raw_profile_rmse(
        residual_pcs,
        profiles,
        pca_models,
        outputs,
        idx,
        bottom_depth=loader.cache.get("bottom_depth"),
        pres_levels=loader.cache.get("PRES"),
        clim_profiles=clim_profiles,
    )
    point_overall = raw_profile_rmse(
        point_pcs,
        profiles,
        pca_models,
        outputs,
        idx,
        bottom_depth=loader.cache.get("bottom_depth"),
        pres_levels=loader.cache.get("PRES"),
        clim_profiles=clim_profiles,
    )

    per_res = _per_profile_rmse(
        residual_pcs, profiles, pca_models, outputs, idx, clim_profiles=clim_profiles
    )
    per_point = _per_profile_rmse(
        point_pcs, profiles, pca_models, outputs, idx, clim_profiles=clim_profiles
    )

    report = {
        "split": args.split,
        "n_profiles": int(len(idx)),
        "residual_checkpoint": args.resume,
        "point_checkpoint": args.point_ckpt,
        "overall_rmse": {
            "residual": residual_overall,
            "point": point_overall,
        },
        "paired_stats": {
            "temperature": _paired_stats(per_res["temperature"], per_point["temperature"]),
            "salinity": _paired_stats(per_res["salinity"], per_point["salinity"]),
        },
        "regression_distribution": {
            "temperature": {
                "worse_count": int(np.sum(per_res["temperature"] > per_point["temperature"])),
                "improved_count": int(np.sum(per_res["temperature"] < per_point["temperature"])),
                "max_regression": float(np.max(per_res["temperature"] - per_point["temperature"])),
            },
            "salinity": {
                "worse_count": int(np.sum(per_res["salinity"] > per_point["salinity"])),
                "improved_count": int(np.sum(per_res["salinity"] < per_point["salinity"])),
                "max_regression": float(np.max(per_res["salinity"] - per_point["salinity"])),
            },
        },
    }

    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
