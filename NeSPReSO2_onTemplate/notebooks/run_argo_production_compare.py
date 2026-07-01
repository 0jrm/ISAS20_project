#!/usr/bin/env python3
"""Headless runner for ARGO production point vs L4 patch comparison."""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import torch

warnings.filterwarnings("ignore")

TEMPLATE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TEMPLATE_ROOT))
sys.path.insert(0, str(TEMPLATE_ROOT / "notebooks"))

from eval_run import main as eval_main
from nb_checkpoints import checkpoint_epoch, discover_checkpoint
from nb_configs import (
    PRODUCTION_ARGO_KEYS,
    PRODUCTION_ARGO_SPECS,
    make_production_config_parser,
    production_argo_matrix_row,
)
from nb_metrics import (
    avg_common_rmse,
    plot_depth_rmse_overlay,
    profile_metrics_from_inference,
)
from train import ensure_cache

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = Path(__file__).parent / "compare_outputs"
EVAL_SPLIT = "test"


def main() -> int:
    OUT_DIR.mkdir(exist_ok=True)
    configs = {
        k: make_production_config_parser(k, template_root=TEMPLATE_ROOT)
        for k in PRODUCTION_ARGO_KEYS
    }

    rows = []
    for key, cfg in configs.items():
        ckpt = discover_checkpoint(key, cfg, template_root=TEMPLATE_ROOT)
        if ckpt is None:
            print(f"{key}: no checkpoint")
            continue
        cache_path = ensure_cache(cfg)
        metrics = profile_metrics_from_inference(
            cfg, str(ckpt), split=EVAL_SPLIT, device=DEVICE
        )
        eval_report = eval_main(cfg, str(ckpt), split=EVAL_SPLIT)
        label = PRODUCTION_ARGO_SPECS[key][0]
        row = {
            "key": key,
            "label": label,
            "group": "argo",
            "tag": metrics["inference"]["dataset_tag"],
            "arch": cfg.config["arch"]["type"],
            "matrix": production_argo_matrix_row(key, cfg),
            "checkpoint": str(ckpt),
            "epoch": checkpoint_epoch(ckpt),
            "cache": str(cache_path),
            "n_test": metrics["inference"]["n_samples"],
            "raw_profile_rmse_common": metrics["raw_profile_rmse_common"],
            "raw_profile_rmse_native": metrics["raw_profile_rmse_native"],
            "avg_common_rmse": avg_common_rmse(metrics),
            "loss": eval_report["loss"],
            "metrics": metrics,
        }
        rows.append(row)
        print(
            f"{label:16s} avg_common={row['avg_common_rmse']:.4f} "
            f"T={row['raw_profile_rmse_common']['temperature']:.4f} "
            f"S={row['raw_profile_rmse_common']['salinity']:.4f}"
        )

    if len(rows) == 2:
        pt = next(r for r in rows if r["key"] == "argo_point")
        l4 = next(r for r in rows if r["key"] == "argo_patch_l4")
        delta = l4["avg_common_rmse"] - pt["avg_common_rmse"]
        pct = 100.0 * delta / pt["avg_common_rmse"]
        winner = l4["label"] if delta < 0 else pt["label"]
        print(f"Δ avg_common (L4 − point): {delta:+.4f} ({pct:+.1f}%) → {winner}")

    if rows:
        overlay_path = OUT_DIR / "argo_production_depth_rmse.png"
        plot_depth_rmse_overlay(
            rows,
            colors={"argo_point": "#1f77b4", "argo_patch_l4": "#d62728"},
            out_path=overlay_path,
            show=False,
        )

    out = {
        "device": str(DEVICE),
        "eval_split": EVAL_SPLIT,
        "models": [
            {
                k: v
                for k, v in r.items()
                if k not in ("metrics",)
            }
            for r in rows
        ],
        "depth_overlay_png": str(OUT_DIR / "argo_production_depth_rmse.png") if rows else None,
    }
    out_path = OUT_DIR / "argo_production_results.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
