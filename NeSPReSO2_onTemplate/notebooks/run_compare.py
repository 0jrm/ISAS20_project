#!/usr/bin/env python3
"""Headless runner for compare_v2_vs_template.ipynb."""
from __future__ import annotations

import json
import pickle
import sys
import time
import warnings
from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

warnings.filterwarnings("ignore")

V2_REPO = Path("/unity/g2/jmiranda/v2-nespreso")
TEMPLATE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TEMPLATE_ROOT.parent
CACHE_PATH = PROJECT_ROOT / "data/cache/train_ready_c0f62f13ca33.pkl"
V2_CHECKPOINT = Path(
    "/unity/g2/jmiranda/SubsurfaceFields/GEM_SubsurfaceFields/saved_models/"
    "model_Test Loss: 0.8847_2024-10-09 20:45:20_sat.pth"
)
V2_DATASET_PICKLE = Path(
    "/unity/g2/jmiranda/SubsurfaceFields/GEM_SubsurfaceFields/config_dataset_full.pkl"
)

sys.path.insert(0, str(V2_REPO / "src"))
sys.path.insert(0, str(TEMPLATE_ROOT))

from nespreso.data.pickle_compat import load_dataset_pickle
from nespreso.data.splits import split_dataset
from nespreso.metrics import bias, rmse
from nespreso.models.mlp import PredictionModel as V2PredictionModel
from nespreso.viz.maps import calculate_average_in_bin

from data_loader.data_loaders import NeSPReSODataset, _collate_with_index
from model.loss import sklearn_inverse_transform_pcs
from model.model import PredictionModel as TemplatePredictionModel
from playground import read_json

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
N_REPEAT = 10
BATCH_SIZE = 512
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15
DEPTH_MAX = 1800
OUT_DIR = Path(__file__).parent / "compare_outputs"


def depth_rmse_bias(residual, axis=1):
    return np.sqrt(np.mean(residual**2, axis=axis)), np.mean(residual, axis=axis)


def time_inference(model, loader, device, n_repeat, inverse_fn):
    model.eval().to(device)
    t0 = time.perf_counter()
    for _ in range(n_repeat):
        pcs_list = []
        with torch.no_grad():
            for batch in loader:
                pcs_list.append(model(batch[0].to(device)).cpu().numpy())
        inverse_fn(np.vstack(pcs_list))
    return (time.perf_counter() - t0) / n_repeat


def make_val_loader(cache, batch_size=512, seed=42):
    ds = NeSPReSODataset(
        torch.tensor(cache["inputs"], dtype=torch.float32),
        torch.tensor(cache["targets"], dtype=torch.float32),
    )
    n = len(ds)
    val_len = int(n * VAL_FRAC)
    train_len = int(n * TRAIN_FRAC)
    test_len = n - train_len - val_len
    g = torch.Generator().manual_seed(seed)
    _, val_sub, _ = random_split(ds, [train_len, val_len, test_len], generator=g)
    loader = DataLoader(val_sub, batch_size=batch_size, shuffle=False, collate_fn=_collate_with_index)
    return loader, np.array(val_sub.indices)


def pcs_to_profiles(pcs, pca_models, outputs):
    prof = sklearn_inverse_transform_pcs(pcs, pca_models, outputs)
    return prof["temperature"], prof["salinity"]


def align_profiles_to_depths(prof, depth_src_m, depth_tgt_m):
    depth_src_m = np.asarray(depth_src_m, dtype=float).squeeze()
    depth_tgt_m = np.asarray(depth_tgt_m, dtype=float).squeeze()
    if prof.shape[0] == len(depth_tgt_m):
        return prof
    src_x = depth_src_m if depth_src_m.size == prof.shape[0] else np.arange(prof.shape[0], dtype=float)
    out = np.empty((len(depth_tgt_m), prof.shape[1]), dtype=prof.dtype)
    for j in range(prof.shape[1]):
        out[:, j] = np.interp(depth_tgt_m, src_x, prof[:, j])
    return out


def main():
    OUT_DIR.mkdir(exist_ok=True)
    results = {}

    t0 = time.perf_counter()
    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    cache_load_s = time.perf_counter() - t0

    outputs = OrderedDict(cache["outputs"])
    input_dim = cache["inputs"].shape[1]
    n_total = cache["inputs"].shape[0]
    val_loader, val_idx = make_val_loader(cache, BATCH_SIZE, SEED)
    n_val = len(val_idx)

    cfg = read_json(TEMPLATE_ROOT / "config.json")
    t0 = time.perf_counter()
    ckpt = torch.load(V2_CHECKPOINT, map_location="cpu", weights_only=False)
    ckpt_load_s = time.perf_counter() - t0

    layers = cfg["arch"]["args"]["layers_config"]
    dropout = cfg["arch"]["args"]["dropout_prob"]
    out_dim = sum(outputs.values())

    v2_model = V2PredictionModel(input_dim, layers, out_dim, dropout)
    tpl_model = TemplatePredictionModel(input_dim, layers, out_dim, dropout)
    v2_model.load_state_dict(ckpt["model_state_dict"])
    tpl_model.load_state_dict(ckpt["model_state_dict"])
    v2_model.eval()
    tpl_model.eval()

    pca_v2 = {"temperature": ckpt["pca_temp"], "salinity": ckpt["pca_sal"]}
    pca_tpl = cache["pca_models"]

    tpl_s = time_inference(
        tpl_model, val_loader, DEVICE, N_REPEAT, lambda pcs: pcs_to_profiles(pcs, pca_tpl, outputs)
    )
    v2_s = time_inference(
        v2_model, val_loader, DEVICE, N_REPEAT, lambda pcs: pcs_to_profiles(pcs, pca_v2, outputs)
    )

    v2_pickle_s = None
    if V2_DATASET_PICKLE.exists():
        t0 = time.perf_counter()
        load_dataset_pickle(V2_DATASET_PICKLE)
        v2_pickle_s = time.perf_counter() - t0

    results["runtime"] = {
        "device": str(DEVICE),
        "cache_load_ms": cache_load_s * 1e3,
        "checkpoint_load_ms": ckpt_load_s * 1e3,
        "template_inference_ms": tpl_s * 1e3,
        "v2_package_inference_ms": v2_s * 1e3,
        "template_us_per_profile": tpl_s / n_val * 1e6,
        "v2_us_per_profile": v2_s / n_val * 1e6,
        "template_over_v2_ratio": tpl_s / v2_s,
        "v2_pickle_load_ms": (v2_pickle_s or 0) * 1e3,
    }

    tpl_pcs, v2_pcs = [], []
    with torch.no_grad():
        for x, _, _ in val_loader:
            x = x.to(DEVICE)
            tpl_pcs.append(tpl_model(x).cpu().numpy())
            v2_pcs.append(v2_model(x).cpu().numpy())
    tpl_pcs = np.vstack(tpl_pcs)
    v2_pcs = np.vstack(v2_pcs)

    results["forward"] = {
        "max_abs_pcs_diff": float(np.max(np.abs(tpl_pcs - v2_pcs))),
        "mean_abs_pcs_diff": float(np.mean(np.abs(tpl_pcs - v2_pcs))),
    }

    tgt_pcs = cache["targets"][val_idx].astype(np.float64)
    true_T, true_S = pcs_to_profiles(tgt_pcs, pca_tpl, outputs)
    pred_T_tpl, pred_S_tpl = pcs_to_profiles(tpl_pcs, pca_tpl, outputs)
    pred_T_v2, pred_S_v2 = pcs_to_profiles(v2_pcs, pca_tpl, outputs)

    depths_m = np.asarray(cache["PRES"], dtype=float).squeeze()
    d_mask = depths_m <= DEPTH_MAX
    dpt_range = np.where(d_mask)[0].astype(int)

    results["metrics_template_pca"] = {
        "T_rmse": rmse(pred_T_tpl[d_mask], true_T[d_mask]),
        "S_rmse": rmse(pred_S_tpl[d_mask], true_S[d_mask]),
        "T_bias": bias(pred_T_tpl[d_mask], true_T[d_mask]),
        "S_bias": bias(pred_S_tpl[d_mask], true_S[d_mask]),
        "cross_impl_T_rmse": rmse(pred_T_tpl[d_mask], pred_T_v2[d_mask]),
        "cross_impl_S_rmse": rmse(pred_S_tpl[d_mask], pred_S_v2[d_mask]),
    }

    pred_T_ckpt, pred_S_ckpt = pcs_to_profiles(tpl_pcs, pca_v2, outputs)
    v2_depth_m = np.arange(pred_T_ckpt.shape[0], dtype=float)
    pred_T_ckpt = align_profiles_to_depths(pred_T_ckpt, v2_depth_m, depths_m)
    pred_S_ckpt = align_profiles_to_depths(pred_S_ckpt, v2_depth_m, depths_m)

    results["pca_compare"] = {
        "T_rmse_template_pca": results["metrics_template_pca"]["T_rmse"],
        "T_rmse_v2_ckpt_pca_aligned": rmse(pred_T_ckpt[d_mask], true_T[d_mask]),
        "S_rmse_template_pca": results["metrics_template_pca"]["S_rmse"],
        "S_rmse_v2_ckpt_pca_aligned": rmse(pred_S_ckpt[d_mask], true_S[d_mask]),
        "template_depth_levels": int(len(depths_m)),
        "v2_ckpt_depth_levels": int(v2_depth_m.size),
    }

    rmse_tpl_T, bias_tpl_T = depth_rmse_bias(pred_T_tpl - true_T)
    rmse_v2_T, bias_v2_T = depth_rmse_bias(pred_T_v2 - true_T)
    rmse_tpl_S, bias_tpl_S = depth_rmse_bias(pred_S_tpl - true_S)
    rmse_v2_S, bias_v2_S = depth_rmse_bias(pred_S_v2 - true_S)

    results["depth_samples"] = {
        "depths_m_head": depths_m[:8].tolist(),
        "T_rmse_surface": float(rmse_tpl_T[0]),
        "T_rmse_near_500m": float(rmse_tpl_T[np.argmin(np.abs(depths_m - 500))]),
        "T_rmse_near_1500m": float(rmse_tpl_T[np.argmin(np.abs(depths_m - 1500))]),
        "S_rmse_surface": float(rmse_tpl_S[0]),
    }

    lat_val = np.floor(cache["LAT"][val_idx]) + 0.5
    lon_val = np.floor(cache["LON"][val_idx]) + 0.5
    lon_bins = np.arange(np.floor(lon_val.min()) - 0.5, np.ceil(lon_val.max()) + 1.5, 1.0)
    lat_bins = np.arange(np.floor(lat_val.min()) - 0.5, np.ceil(lat_val.max()) + 1.5, 1.0)
    lon_centers = lon_bins + 0.5
    lat_centers = lat_bins + 0.5

    res_tpl_T = pred_T_tpl - true_T
    res_v2_T = pred_T_v2 - true_T
    grid_tpl, nprof = calculate_average_in_bin(
        lon_centers, lat_centers, lon_val, lat_val, res_tpl_T, dpt_range, True
    )
    grid_v2, _ = calculate_average_in_bin(
        lon_centers, lat_centers, lon_val, lat_val, res_v2_T, dpt_range, True
    )
    valid = nprof > 0
    results["maps"] = {
        "grid_shape": list(grid_tpl.shape),
        "max_profiles_per_bin": float(nprof.max()),
        "mean_bin_T_rmse_template": float(np.nanmean(grid_tpl[valid])),
        "mean_bin_T_rmse_v2": float(np.nanmean(grid_v2[valid])),
        "max_abs_bin_T_rmse_diff": float(np.nanmax(np.abs(grid_tpl - grid_v2)[valid])),
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, tpl_r, v2_r, title, xlabel in [
        (axes[0, 0], rmse_tpl_T, rmse_v2_T, "Temperature RMSE", "RMSE [C]"),
        (axes[0, 1], rmse_tpl_S, rmse_v2_S, "Salinity RMSE", "RMSE [PSU]"),
        (axes[1, 0], bias_tpl_T, bias_v2_T, "Temperature bias", "Bias [C]"),
        (axes[1, 1], bias_tpl_S, bias_v2_S, "Salinity bias", "Bias [PSU]"),
    ]:
        ax.plot(tpl_r[d_mask], depths_m[d_mask], label="template port", lw=2)
        ax.plot(v2_r[d_mask], depths_m[d_mask], label="v2 package", lw=2, ls="--")
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Depth [m]")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "depth_rmse_bias.png", dpi=120)
    plt.close()

    if V2_DATASET_PICKLE.exists():
        data = load_dataset_pickle(V2_DATASET_PICKLE)
        full_ds = data["full_dataset"]
        if not hasattr(full_ds, "n_components"):
            full_ds.n_components = full_ds.pca_temp.n_components_
        _, val_ds, _ = split_dataset(full_ds, TRAIN_FRAC, VAL_FRAC, 1 - TRAIN_FRAC - VAL_FRAC)
        v2_val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

        def v2_native_inverse(pcs):
            full_ds.inverse_transform(pcs)

        native_s = time_inference(v2_model, v2_val_loader, DEVICE, N_REPEAT, v2_native_inverse)
        results["v2_native"] = {
            "N_full": len(full_ds),
            "N_val": len(val_ds),
            "inference_ms": native_s * 1e3,
            "us_per_profile": native_s / len(val_ds) * 1e6,
        }

    results["data"] = {"N": n_total, "n_val": n_val, "input_dim": input_dim}
    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"\nWrote {OUT_DIR / 'results.json'} and depth_rmse_bias.png")


if __name__ == "__main__":
    main()
