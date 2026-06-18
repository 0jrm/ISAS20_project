#!/usr/bin/env python3
"""Phase 6 headless pipeline: Tier A global EDA + Tier B GoM ML diagnostics + results table."""

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

from global_eda import run_eda as run_global_eda
from nb_checkpoints import discover_checkpoint
from nb_configs import make_config_parser
from nb_metrics import bin_map_scalar_rmse, profile_metrics_from_pcs, run_inference
from results_table import build_table, collect_eval_rows, to_markdown


def _tier_b_gom(out_dir: Path, device: torch.device) -> dict:
    """Tier B interim: prod patch16 on v2 GoM (global ML blocked on corrupt profiles)."""
    cfg = make_config_parser("isas_patch", template_root=ROOT)
    ckpt = discover_checkpoint("isas_patch", cfg, template_root=ROOT)
    if ckpt is None:
        raise FileNotFoundError("patch16_scales checkpoint not found")

    inf = run_inference(cfg, str(ckpt), split="test", device=device)
    metrics = profile_metrics_from_pcs(
        inf["pcs"], inf["indices"], inf["cache"], inf["pca_models"], inf["outputs"]
    )

    cache = inf["cache"]
    idx = np.asarray(inf["indices"], dtype=int)
    lon = np.asarray(cache["LON"])[idx]
    lat = np.asarray(cache["LAT"])[idx]

    plots_dir = out_dir / "gom_tier_b"
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, var, color in zip(axes, ("temperature", "salinity"), ("tab:red", "tab:blue")):
        stats = metrics["depth_stats"][var]
        ax.plot(metrics["depth_m_common"], stats["rmse"], color=color)
        ax.set_xlabel("depth (m)")
        ax.set_ylabel("RMSE")
        ax.set_title(f"{var} depth RMSE (common grid)")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "depth_rmse.png", dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, var in zip(axes, ("temperature", "salinity")):
        pred_c = metrics["pred_profiles"][var]
        true_c = metrics["true_profiles"][var]
        z = metrics["z_native"]
        from nb_metrics import align_profiles_to_depth, common_depth_mask

        pred_a = align_profiles_to_depth(pred_c, z)[common_depth_mask()]
        true_a = align_profiles_to_depth(true_c, z)[common_depth_mask()]
        lon_bins, lat_bins, grid, nprof = bin_map_scalar_rmse(lon, lat, pred_a, true_a)
        extent = [lon_bins[0], lon_bins[-1], lat_bins[0], lat_bins[-1]]
        im = ax.imshow(grid, origin="lower", extent=extent, aspect="auto", cmap="viridis")
        ax.set_title(f"{var} bin RMSE (1°)")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(plots_dir / "bin_map_rmse.png", dpi=120)
    plt.close(fig)

    return {
        "checkpoint": str(ckpt),
        "n_eval": inf["n_samples"],
        "raw_profile_rmse_common": metrics["raw_profile_rmse_common"],
        "raw_profile_rmse_native": metrics["raw_profile_rmse_native"],
        "plots": {
            "depth_rmse": str(plots_dir / "depth_rmse.png"),
            "bin_map_rmse": str(plots_dir / "bin_map_rmse.png"),
        },
        "note": "v2 GoM interim — global v1 cache blocked (corrupt profiles HDF5)",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", default="config_isas_global_gom.json")
    parser.add_argument("--out-dir", default="saved/phase6")
    parser.add_argument("--skip-eda", action="store_true")
    parser.add_argument("--skip-tier-b", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report: dict = {"device": str(device), "tier_a": None, "tier_b": None, "results": None}

    if not args.skip_eda:
        cfg_path = ROOT / args.config
        eda_out = out_dir / "global_eda"
        report["tier_a"] = run_global_eda(cfg_path, eda_out)

    if not args.skip_tier_b:
        report["tier_b"] = _tier_b_gom(out_dir, device)

    rows = collect_eval_rows(ROOT / "saved")
    results = build_table(rows)
    results_dir = out_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "eval_table.json").write_text(json.dumps(results, indent=2) + "\n")
    (results_dir / "eval_table.md").write_text(to_markdown(results))
    report["results"] = results

    (out_dir / "phase6_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"\nWrote {out_dir / 'phase6_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
