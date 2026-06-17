#!/usr/bin/env python3
"""Balanced ISAS vs ARGO comparison (apples-to-apples).

Two complementary 50/50 evaluations on the common 0-1800 m grid (10 m), all RMSE
masked to finite truth:

  (A) Balanced-truth, shared colocation sites: held-out matched pairs; each model
      scored 50% vs ISAS analysis and 50% vs ARGO in-situ at the SAME locations.
      This is the true head-to-head — neither model is judged only on its own truth.

  (B) Balanced pooled holdout: equal-sized native test samples from each regime,
      each model scored on its own truth. A balanced report card per regime.
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

import data_loader.data_loaders as module_data
import model.model as module_arch
from model.loss import sklearn_inverse_transform_pcs
from parse_config import ConfigParser, validate_config
from playground import prepare_device, read_json
from preproc.overlap import (
    depth_grid_m,
    find_matched_pairs,
    holdout_mask,
    interp_profiles,
    load_cache,
    overlap_summary,
)
from train import ensure_cache, set_seed

VARS = ("temperature", "salinity")


def _prep(config_path: str, checkpoint_path: str, device):
    cfg = read_json(config_path)
    validate_config(cfg)
    config = ConfigParser(cfg)
    ensure_cache(config)
    cache_path = config.config["data_loader"]["args"]["cache_path"]
    cache = load_cache(cache_path)

    dl_args = dict(config["data_loader"]["args"])
    dl_args["split"] = "test"
    dl_args["shuffle"] = False
    loader = getattr(module_data, config["data_loader"]["type"])(**dl_args)
    test_idx = np.asarray(loader.split_indices["test"], dtype=int)

    model = config.init_obj("arch", module_arch).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state_dict)
    model.eval()
    pca_models = ckpt.get("pca_models", cache["pca_models"])
    outputs = OrderedDict(ckpt.get("outputs", dict(cache["outputs"])))
    tag = cache["dataset_tag"]
    z_native = cache["PRES"].astype(np.float64) if tag == "isas20" else np.arange(
        cache["min_depth"], cache["max_depth"] + 1, dtype=np.float64
    )
    return dict(
        model=model, cache=cache, pca=pca_models, outputs=outputs,
        z=z_native, tag=tag, cache_path=str(cache_path), test_idx=test_idx,
    )


def _predict(ctx, rows: np.ndarray, device, chunk: int = 1024) -> dict[str, np.ndarray]:
    """Predict native-depth profiles for given cache rows. Returns name -> (n_z, len(rows))."""
    inp = ctx["cache"]["inputs"][rows]
    parts = []
    with torch.no_grad():
        for s in range(0, len(rows), chunk):
            x = torch.tensor(inp[s : s + chunk], dtype=torch.float32, device=device)
            parts.append(ctx["model"](x).cpu().numpy())
    pcs = np.vstack(parts)
    return sklearn_inverse_transform_pcs(pcs, ctx["pca"], ctx["outputs"])


def _rmse_from_sq(sq: list[np.ndarray]) -> dict[str, float] | None:
    if not sq:
        return None
    pooled = np.concatenate(sq)
    return float(np.sqrt(np.nanmean(pooled))) if np.isfinite(pooled).any() else None


def eval_balanced_shared(isas, argo, device, z_dst, dt_max_days, holdout_frac, seed):
    """(A) head-to-head on held-out matched pairs, 50/50 truth."""
    pairs = find_matched_pairs(isas["cache"], argo["cache"], dt_max_days=dt_max_days)
    mask = holdout_mask(pairs, isas["cache"], frac=holdout_frac, seed=seed)
    eval_pairs = [p for p, m in zip(pairs, mask) if m]
    i_rows = np.array([i for i, _j, _ in eval_pairs])
    j_rows = np.array([j for _i, j, _ in eval_pairs])

    pred_i = _predict(isas, i_rows, device)  # ISAS model
    pred_a = _predict(argo, j_rows, device)  # ARGO model

    sq = {m: {t: {v: [] for v in VARS} for t in ("isas", "argo")} for m in ("isas_model", "argo_model")}
    for k in range(len(eval_pairs)):
        for v in VARS:
            truth_i = interp_profiles(isas["cache"]["profiles"][v][:, i_rows[k]], isas["z"], z_dst)
            truth_a = interp_profiles(argo["cache"]["profiles"][v][:, j_rows[k]], argo["z"], z_dst)
            pi = interp_profiles(pred_i[v][:, k], isas["z"], z_dst)
            pa = interp_profiles(pred_a[v][:, k], argo["z"], z_dst)
            sq["isas_model"]["isas"][v].append((pi - truth_i) ** 2)
            sq["isas_model"]["argo"][v].append((pi - truth_a) ** 2)
            sq["argo_model"]["isas"][v].append((pa - truth_i) ** 2)
            sq["argo_model"]["argo"][v].append((pa - truth_a) ** 2)

    def per_truth(model, truth):
        return {v: _rmse_from_sq(sq[model][truth][v]) for v in VARS}

    def balanced(model):
        # 50% ISAS truth + 50% ARGO truth pooled by squared error
        return {v: _rmse_from_sq(sq[model]["isas"][v] + sq[model]["argo"][v]) for v in VARS}

    return {
        "n_eval_pairs": len(eval_pairs),
        "holdout_frac": holdout_frac,
        "balanced_50_50": {"isas_model": balanced("isas_model"), "argo_model": balanced("argo_model")},
        "rmse_vs_isas_analysis": {"isas_model": per_truth("isas_model", "isas"), "argo_model": per_truth("argo_model", "isas")},
        "rmse_vs_argo_in_situ": {"isas_model": per_truth("isas_model", "argo"), "argo_model": per_truth("argo_model", "argo")},
    }


def eval_balanced_pooled(isas, argo, device, z_dst, seed):
    """(B) equal-sized native test holdout from each regime; each model on its own truth."""
    k = min(len(isas["test_idx"]), len(argo["test_idx"]))
    rng = np.random.default_rng(seed)
    i_rows = np.sort(rng.choice(isas["test_idx"], size=k, replace=False))
    a_rows = np.sort(rng.choice(argo["test_idx"], size=k, replace=False))

    def home_rmse(ctx, rows):
        pred = _predict(ctx, rows, device)
        sq = {v: [] for v in VARS}
        for col in range(len(rows)):
            for v in VARS:
                truth = interp_profiles(ctx["cache"]["profiles"][v][:, rows[col]], ctx["z"], z_dst)
                p = interp_profiles(pred[v][:, col], ctx["z"], z_dst)
                sq[v].append((p - truth) ** 2)
        return {v: _rmse_from_sq(sq[v]) for v in VARS}

    isas_home = home_rmse(isas, i_rows)
    argo_home = home_rmse(argo, a_rows)
    balanced = {
        v: float(np.mean([x for x in (isas_home[v], argo_home[v]) if x is not None]))
        for v in VARS
    }
    return {
        "n_per_source": int(k),
        "isas_model_on_isas": isas_home,
        "argo_model_on_argo": argo_home,
        "balanced_mean": balanced,
    }


def eval_matched(isas_config, isas_ckpt, argo_config, argo_ckpt, *, device=None,
                 dt_max_days=1.0, holdout_frac=0.15, seed=42):
    set_seed(seed)
    argo_cfg = read_json(argo_config)
    v2_src = argo_cfg.get("io", {}).get("v2_src")
    if v2_src:
        import sys

        sys.path.insert(0, str(v2_src))
    if device is not None:
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = device
    dev, _ = prepare_device(1)

    isas = _prep(isas_config, isas_ckpt, dev)
    argo = _prep(argo_config, argo_ckpt, dev)
    z_dst = depth_grid_m(0.0, 1800.0, 10.0)

    return {
        "eval_type": "balanced_apples_to_apples",
        "notes": [
            "All RMSE on common 0-1800 m grid (10 m), masked to finite truth (no fill zeros).",
            "(A) shared colocation sites, 50/50 truth = true head-to-head.",
            "(B) equal-sized native test sample per regime, each model on own truth.",
            "Native per-tag eval_*.json is NOT cross-comparable; use this file.",
        ],
        "match": {"dt_max_days": dt_max_days, "grid_deg": 0.5, **overlap_summary(isas["cache"], argo["cache"])},
        "checkpoints": {"isas20": isas_ckpt, "argo_v2": argo_ckpt},
        "caches": {"isas20": isas["cache_path"], "argo_v2": argo["cache_path"]},
        "A_shared_sites_balanced_truth": eval_balanced_shared(isas, argo, dev, z_dst, dt_max_days, holdout_frac, seed),
        "B_pooled_holdout_50_50": eval_balanced_pooled(isas, argo, dev, z_dst, seed),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Balanced ISAS/ARGO comparison")
    parser.add_argument("--isas-config", default="config_isas.json")
    parser.add_argument("--isas-checkpoint", required=True)
    parser.add_argument("--argo-config", default="config_argo.json")
    parser.add_argument("--argo-checkpoint", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("-d", "--device", default=None)
    parser.add_argument("--dt-max-days", type=float, default=1.0)
    args = parser.parse_args()
    report = eval_matched(
        args.isas_config, args.isas_checkpoint, args.argo_config, args.argo_checkpoint,
        device=args.device, dt_max_days=args.dt_max_days,
    )
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")
