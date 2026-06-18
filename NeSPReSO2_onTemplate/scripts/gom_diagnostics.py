#!/usr/bin/env python3
"""GoM ML diagnostics + results table (Phase 6)."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notebooks"))
sys.path.insert(0, str(ROOT / "scripts"))

from nb_checkpoints import discover_checkpoint
from nb_configs import make_config_parser
from nb_metrics import (
    align_profiles_to_depth,
    bin_map_scalar_rmse,
    common_depth_mask,
    profile_metrics_from_pcs,
    run_inference,
)
from results_table import build_table, collect_eval_rows, to_markdown


def _plot_gom_metrics(
    metrics: dict,
    lon: np.ndarray,
    lat: np.ndarray,
    plots_dir: Path,
    *,
    label: str,
) -> dict[str, str]:
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, var, color in zip(axes, ("temperature", "salinity"), ("tab:red", "tab:blue")):
        stats = metrics["depth_stats"][var]
        ax.plot(metrics["depth_m_common"], stats["rmse"], color=color)
        ax.set_xlabel("depth (m)")
        ax.set_ylabel("RMSE")
        ax.set_title(f"{label} {var} depth RMSE")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    depth_path = plots_dir / f"{label}_depth_rmse.png"
    fig.savefig(depth_path, dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, var in zip(axes, ("temperature", "salinity")):
        pred_c = metrics["pred_profiles"][var]
        true_c = metrics["true_profiles"][var]
        z = metrics["z_native"]
        pred_a = align_profiles_to_depth(pred_c, z)[common_depth_mask()]
        true_a = align_profiles_to_depth(true_c, z)[common_depth_mask()]
        lon_bins, lat_bins, grid, _nprof = bin_map_scalar_rmse(lon, lat, pred_a, true_a)
        extent = [lon_bins[0], lon_bins[-1], lat_bins[0], lat_bins[-1]]
        im = ax.imshow(grid, origin="lower", extent=extent, aspect="auto", cmap="viridis")
        ax.set_title(f"{label} {var} bin RMSE (1°)")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    bin_path = plots_dir / f"{label}_bin_map_rmse.png"
    fig.savefig(bin_path, dpi=120)
    plt.close(fig)

    return {"depth_rmse": str(depth_path), "bin_map_rmse": str(bin_path)}


def _run_model(key: str, out_dir: Path, device: torch.device) -> dict:
    cfg = make_config_parser(key, template_root=ROOT)
    ckpt = discover_checkpoint(key, cfg, template_root=ROOT)
    if ckpt is None:
        raise FileNotFoundError(f"{key}: checkpoint not found")

    inf = run_inference(cfg, str(ckpt), split="test", device=device)
    metrics = profile_metrics_from_pcs(
        inf["pcs"], inf["indices"], inf["cache"], inf["pca_models"], inf["outputs"]
    )
    idx = np.asarray(inf["indices"], dtype=int)
    lon = np.asarray(inf["cache"]["LON"])[idx]
    lat = np.asarray(inf["cache"]["LAT"])[idx]
    plots = _plot_gom_metrics(metrics, lon, lat, out_dir / key, label=key)

    return {
        "config_key": key,
        "checkpoint": str(ckpt),
        "n_eval": inf["n_samples"],
        "raw_profile_rmse_common": metrics["raw_profile_rmse_common"],
        "raw_profile_rmse_native": metrics["raw_profile_rmse_native"],
        "plots": plots,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="saved/gom_diagnostics")
    parser.add_argument("--keys", nargs="+", default=["isas_patch", "argo_point"])
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report: dict = {"device": str(device), "models": [], "results": None}

    for key in args.keys:
        report["models"].append(_run_model(key, out_dir, device))

    rows = collect_eval_rows(ROOT / "saved")
    results = build_table(rows)
    results_dir = out_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "eval_table.json").write_text(json.dumps(results, indent=2) + "\n")
    (results_dir / "eval_table.md").write_text(to_markdown(results))
    report["results"] = results

    (out_dir / "gom_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"\nWrote {out_dir / 'gom_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
