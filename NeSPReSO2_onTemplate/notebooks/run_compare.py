#!/usr/bin/env python3
"""Headless runner for compare_v2_vs_template.ipynb (statistics contract in nb_metrics)."""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch

warnings.filterwarnings("ignore")

TEMPLATE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TEMPLATE_ROOT))
sys.path.insert(0, str(TEMPLATE_ROOT / "notebooks"))

from nb_checkpoints import discover_checkpoint
from nb_configs import SURFACE_CONFIG_KEYS, make_config_parser
from nb_metrics import (
    DEPTH_RANGE_M,
    representation_metrics_on_split,
    profile_metrics_from_pcs,
    run_inference,
    statistics_markdown,
)
from train import ensure_cache

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = Path(__file__).parent / "compare_outputs"
EVAL_SPLIT = "test"
AE_EPOCHS = 10


def main() -> int:
    OUT_DIR.mkdir(exist_ok=True)
    results: dict = {
        "device": str(DEVICE),
        "eval_split": EVAL_SPLIT,
        "common_depth_range_m": list(DEPTH_RANGE_M),
        "statistics_contract": statistics_markdown(),
        "representation": [],
        "surface_models": [],
    }

    configs = {k: make_config_parser(k, template_root=TEMPLATE_ROOT) for k in SURFACE_CONFIG_KEYS}

    import pickle

    for key in ("isas_patch", "argo_point"):
        cfg = configs[key]
        path = ensure_cache(cfg)
        with open(path, "rb") as f:
            cache = pickle.load(f)
        for row in representation_metrics_on_split(
            cache,
            EVAL_SPLIT,
            encoding_dim=16,
            ae_epochs=AE_EPOCHS,
            device=DEVICE,
            seed=cfg.config.get("seed", 42),
        ):
            row = {k: v for k, v in row.items() if k != "ae_stats"}
            row["config_key"] = key
            results["representation"].append(row)

    for key, cfg in configs.items():
        path = ensure_cache(cfg)
        with open(path, "rb") as f:
            cache = pickle.load(f)
        tag = cache.get("dataset_tag", key)
        entry = {
            "config_key": key,
            "dataset_tag": tag,
            "arch": cfg.config["arch"]["type"],
            "cache": path,
            "n_samples": int(cache["inputs"].shape[0]),
            "inference": None,
        }
        ckpt_dir = Path(cfg.config["trainer"]["save_dir"])
        ckpt = discover_checkpoint(key, cfg, template_root=TEMPLATE_ROOT)
        if ckpt and ckpt.exists():
            inf = run_inference(cfg, str(ckpt), split=EVAL_SPLIT, device=DEVICE)
            m = profile_metrics_from_pcs(
                inf["pcs"], inf["indices"], inf["cache"], inf["pca_models"], inf["outputs"]
            )
            entry["inference"] = {
                "checkpoint": str(ckpt),
                "n_eval": inf["n_samples"],
                "raw_profile_rmse_common": m["raw_profile_rmse_common"],
                "raw_profile_rmse_native": m["raw_profile_rmse_native"],
            }
        results["surface_models"].append(entry)

    out_path = OUT_DIR / "results.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
