#!/usr/bin/env python3
"""Post-train interpretability + S3 paired significance for residual cube models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import model.model as module_arch
from base.split_utils import build_split_indices
from base.util import read_json
from model.metric import per_variable_rmse
from parse_config import ConfigParser
from train import ensure_cache


def _load_point_cube_inputs(residual_cache: dict, indices: np.ndarray) -> np.ndarray:
    """Z-scored 9-D inputs matching point_cube training (not raw residual-cache base block)."""
    import pickle

    from preproc.features.export_feature_cache import build_point_cube_cache

    point_cfg_path = _ROOT / "config/argo/config_argo_point_cube.json"
    if not point_cfg_path.is_file():
        raise FileNotFoundError(f"point_cube config not found: {point_cfg_path}")
    point_cfg = read_json(point_cfg_path)
    point_cache_path = build_point_cube_cache(point_cfg, force=False)
    with open(point_cache_path, "rb") as f:
        point_cache = pickle.load(f)
    if point_cache["inputs"].shape[0] != residual_cache["inputs"].shape[0]:
        raise ValueError(
            f"point cache N={point_cache['inputs'].shape[0]} != "
            f"residual cache N={residual_cache['inputs'].shape[0]}"
        )
    idx = np.asarray(indices, dtype=int)
    return point_cache["inputs"][idx].astype(np.float32)


def _load_golden_v2_inputs(residual_cache: dict, indices: np.ndarray) -> np.ndarray:
    """Legacy v2 pickle point inputs for S3b (argo16_scales / config_argo)."""
    import pickle

    from preproc.export_v2_cache import build_argo_cache

    golden_cfg_path = _ROOT / "config/argo/config_argo.json"
    if not golden_cfg_path.is_file():
        raise FileNotFoundError(f"golden point config not found: {golden_cfg_path}")
    golden_cfg = read_json(golden_cfg_path)
    golden_cache_path = build_argo_cache(golden_cfg, force=False)
    with open(golden_cache_path, "rb") as f:
        golden_cache = pickle.load(f)
    if golden_cache["inputs"].shape[0] != residual_cache["inputs"].shape[0]:
        raise ValueError(
            f"golden cache N={golden_cache['inputs'].shape[0]} != "
            f"residual cache N={residual_cache['inputs'].shape[0]}"
        )
    idx = np.asarray(indices, dtype=int)
    return golden_cache["inputs"][idx].astype(np.float32)


def _per_profile_se(pred: np.ndarray, true: np.ndarray, pca_models, outputs, profiles, indices):
    from model.loss import sklearn_inverse_transform_pcs

    idx = np.asarray(indices, dtype=int)
    pred_prof = sklearn_inverse_transform_pcs(pred[idx], pca_models, outputs)
    true_prof = {k: profiles[k][:, idx] for k in outputs}
    se = []
    for i in range(len(idx)):
        errs = []
        for name in outputs:
            d = pred_prof[name][:, i] - true_prof[name][:, i]
            errs.append(np.nanmean(d ** 2))
        se.append(float(np.mean(errs)))
    return np.asarray(se, dtype=np.float64)


def permutation_importance(model, inputs, targets, indices, feat_offset, feat_dim, feat_names, n_perm=5):
    model.eval()
    indices = np.asarray(indices, dtype=int)
    base_loss = float(np.mean((model(torch.tensor(inputs[indices])).detach().numpy() - targets[indices]) ** 2))
    scores = {}
    rng = np.random.default_rng(0)
    for j, name in enumerate(feat_names):
        col = feat_offset + j
        deltas = []
        for _ in range(n_perm):
            x = inputs.copy()
            perm = rng.permutation(len(indices))
            x[indices, col] = x[indices[perm], col]
            pred = model(torch.tensor(x[indices])).detach().numpy()
            deltas.append(float(np.mean((pred - targets[indices]) ** 2)) - base_loss)
        scores[name] = float(np.mean(deltas))
    return scores


def paired_significance(se_a: np.ndarray, se_b: np.ndarray, *, n_boot: int = 2000):
    diff = se_a - se_b
    wilk = stats.wilcoxon(diff) if diff.size else (np.nan, np.nan)
    ttest = stats.ttest_rel(se_a, se_b) if diff.size else (np.nan, np.nan)
    rng = np.random.default_rng(1)
    boots = []
    n = len(diff)
    for _ in range(n_boot):
        samp = rng.integers(0, n, size=n)
        boots.append(float(np.sqrt(np.mean(se_a[samp])) - np.sqrt(np.mean(se_b[samp]))))
    ci = np.percentile(boots, [2.5, 97.5])
    return {
        "wilcoxon_stat": float(wilk.statistic) if np.isfinite(wilk.statistic) else None,
        "wilcoxon_p": float(wilk.pvalue),
        "paired_t_p": float(ttest.pvalue),
        "delta_rmse_ci95": [float(ci[0]), float(ci[1])],
        "mean_delta_se": float(np.mean(diff)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-r", "--resume", required=True, help="residual checkpoint")
    parser.add_argument("--point-ckpt", required=True, help="point_cube baseline checkpoint")
    parser.add_argument("--golden-ckpt", default=None, help="optional golden (argo16_scales) baseline")
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    cfg = read_json(args.config)
    config = ConfigParser(cfg)
    cache_path = ensure_cache(config)
    import pickle

    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    inputs = cache["inputs"].astype(np.float32)
    targets = cache["targets"].astype(np.float32)
    dl = cfg["data_loader"]["args"]
    indices = build_split_indices(
        inputs.shape[0],
        cache.get("JULD"),
        dl,
        dataset_tag=cache.get("dataset_tag"),
        v2_src=dl.get("v2_src"),
    )[args.split]

    residual = config.init_obj("arch", module_arch)
    ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
    residual.load_state_dict(ckpt["state_dict"], strict=False)
    residual.eval()

    point = module_arch.PatchConvMLP(input_dim=9, output_dim=32, n_enc=6, n_sat=3, patch_shape=None, head_layers=[1024, 1024])
    pck = torch.load(args.point_ckpt, map_location="cpu", weights_only=False)
    point.load_state_dict(pck["state_dict"], strict=True)
    point.eval()

    x = torch.tensor(inputs[indices])
    x_point = torch.tensor(_load_point_cube_inputs(cache, indices))
    with torch.no_grad():
        pred_res = residual(x).numpy()
        pred_pt = point(x_point).numpy()

    pca_models = cache["pca_models"]
    outputs = cache["outputs"]
    profiles = cache["profiles"]
    from model.loss import sklearn_inverse_transform_pcs

    pred_res_prof = sklearn_inverse_transform_pcs(pred_res, pca_models, outputs)
    pred_pt_prof = sklearn_inverse_transform_pcs(pred_pt, pca_models, outputs)
    idx = np.asarray(indices, dtype=int)
    rmse_res = {}
    rmse_pt = {}
    for name in outputs:
        true = profiles[name][:, idx]
        rmse_res[name] = float(np.sqrt(np.nanmean((pred_res_prof[name] - true) ** 2)))
        rmse_pt[name] = float(np.sqrt(np.nanmean((pred_pt_prof[name] - true) ** 2)))

    se_res = _per_profile_se(pred_res, targets, pca_models, outputs, profiles, np.arange(len(indices)))
    se_pt = _per_profile_se(pred_pt, targets, pca_models, outputs, profiles, np.arange(len(indices)))

    rmse_golden = None
    se_golden = None
    if args.golden_ckpt:
        golden = module_arch.PatchConvMLP(
            input_dim=9, output_dim=32, n_enc=6, n_sat=3, patch_shape=None, head_layers=[1024, 1024]
        )
        gck = torch.load(args.golden_ckpt, map_location="cpu", weights_only=False)
        golden.load_state_dict(gck["state_dict"], strict=True)
        golden.eval()
        x_golden = torch.tensor(_load_golden_v2_inputs(cache, indices))
        with torch.no_grad():
            pred_golden = golden(x_golden).numpy()
        pred_golden_prof = sklearn_inverse_transform_pcs(pred_golden, pca_models, outputs)
        rmse_golden = {}
        for name in outputs:
            true = profiles[name][:, idx]
            rmse_golden[name] = float(np.sqrt(np.nanmean((pred_golden_prof[name] - true) ** 2)))
        se_golden = _per_profile_se(pred_golden, targets, pca_models, outputs, profiles, np.arange(len(indices)))

    feat_offset = int(cache.get("base_dim", 9))
    feat_names = cache.get("residual_feature_names", cache.get("feature_names", []))
    feat_dim = int(cache.get("feat_dim", len(feat_names)))
    featimp = permutation_importance(
        residual, inputs, targets, indices, feat_offset, feat_dim, feat_names, n_perm=3
    )

    gate = residual.gate.detach().cpu().numpy().tolist() if hasattr(residual, "gate") else []

    report = {
        "checkpoint": args.resume,
        "point_checkpoint": args.point_ckpt,
        "golden_checkpoint": args.golden_ckpt,
        "split": args.split,
        "n_samples": len(indices),
        "raw_profile_rmse_residual": rmse_res,
        "raw_profile_rmse_point": rmse_pt,
        "s3a_note": "S3a: paired residual vs point_cube on z-scored point cache inputs",
        "s3a_paired_significance": paired_significance(se_res, se_pt),
        "feature_importance": featimp,
        "gate_values": gate,
    }
    if rmse_golden is not None:
        report["s3b_golden_rmse"] = {
            "note": "unpaired vs residual; golden uses legacy v2 point inputs, not cube point_cube",
            "raw_profile_rmse": rmse_golden,
        }
        report["raw_profile_rmse_golden"] = rmse_golden
        if se_golden is not None:
            report["s3b_golden_rmse"]["paired_significance_vs_residual"] = paired_significance(se_res, se_golden)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
