#!/usr/bin/env python3
"""Headless runner for encoding-compare notebook (nb_metrics contract)."""

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

from nb_checkpoints import discover_compare_checkpoint
from nb_configs import COMPARE_CONFIG_KEYS, COMPARE_CONFIGS, compare_matrix_row, make_compare_config_parser
from nb_metrics import (
    DEPTH_RANGE_M,
    avg_common_rmse,
    plot_bin_maps_best,
    plot_depth_rmse_overlay,
    profile_metrics_from_inference,
    select_best,
    statistics_markdown,
)
from train import ensure_cache

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = Path(__file__).parent / "compare_outputs"
EVAL_SPLIT = "test"


def main() -> int:
    OUT_DIR.mkdir(exist_ok=True)
    configs = {
        k: make_compare_config_parser(k, template_root=TEMPLATE_ROOT) for k in COMPARE_CONFIG_KEYS
    }

    results: dict = {
        "device": str(DEVICE),
        "eval_split": EVAL_SPLIT,
        "common_depth_range_m": list(DEPTH_RANGE_M),
        "statistics_contract": statistics_markdown(),
        "config_matrix": [compare_matrix_row(k, configs[k]) for k in COMPARE_CONFIG_KEYS],
        "surface_models": [],
    }

    summary_rows = []
    for key, cfg in configs.items():
        path = ensure_cache(cfg)
        import pickle

        with open(path, "rb") as f:
            cache = pickle.load(f)
        spec = COMPARE_CONFIGS[key]
        entry = {
            "key": key,
            "label": spec.label,
            "group": spec.group,
            "encoding": spec.encoding,
            "dataset_tag": cache.get("dataset_tag", key),
            "arch": cfg.config["arch"]["type"],
            "cache": path,
            "n_samples": int(cache["inputs"].shape[0]),
            "inference": None,
        }
        ckpt = discover_compare_checkpoint(key, cfg, template_root=TEMPLATE_ROOT)
        if ckpt and ckpt.exists():
            metrics = profile_metrics_from_inference(
                cfg, str(ckpt), split=EVAL_SPLIT, device=DEVICE
            )
            entry["inference"] = {
                "checkpoint": str(ckpt),
                "n_eval": metrics["inference"]["n_samples"],
                "raw_profile_rmse_common": metrics["raw_profile_rmse_common"],
                "raw_profile_rmse_native": metrics["raw_profile_rmse_native"],
                "avg_common_rmse": avg_common_rmse(metrics),
            }
            summary_rows.append(
                {
                    "key": key,
                    "label": spec.label,
                    "group": spec.group,
                    "metrics": metrics,
                    "avg_common_rmse": avg_common_rmse(metrics),
                }
            )
        results["surface_models"].append(entry)

    best_isas = select_best(summary_rows, "isas")
    best_argo = select_best(summary_rows, "argo")
    if summary_rows:
        overlay_path = OUT_DIR / "depth_rmse_overlay.png"
        plot_depth_rmse_overlay(summary_rows, out_path=overlay_path, show=False)
        results["depth_overlay_png"] = str(overlay_path)

    if best_isas or best_argo:
        maps_path = OUT_DIR / "bin_maps_best.png"
        plot_bin_maps_best(best_isas, best_argo, out_path=maps_path, show=False)
        results["bin_maps_png"] = str(maps_path)
        results["best_isas"] = best_isas["label"] if best_isas else None
        results["best_argo"] = best_argo["label"] if best_argo else None

    out_path = OUT_DIR / "results.json"
    # metrics contain numpy arrays via inference — strip for JSON
    serializable = json.loads(
        json.dumps(
            results,
            default=lambda o: o if isinstance(o, (str, int, float, bool, type(None))) else str(o),
        )
    )
    out_path.write_text(json.dumps(serializable, indent=2) + "\n")
    print(json.dumps(serializable, indent=2))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
