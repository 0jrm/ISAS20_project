#!/usr/bin/env python3
"""Derive ``loss_scales`` from a train-ready cache pickle.

Scales normalize the combined PCA loss so each branch contributes ~1.0 at
zero-initialized predictions:

- ``profile_scales[var]``: profile-space MSE for that variable when pred PCs are
  all zeros (inverse-transform mean vs true reconstruction).
- ``combined_pca_scale``: ``PCALoss`` at zero pred with those profile scales
  (equals the number of output variables when scales match per-var init MSE).
- ``combined_mse_scale``: weighted PC-space MSE at zero pred.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.loss import CombinedPCALoss, PCALoss, output_slices, torch_reconstruct_profile
from parse_config import ConfigParser, validate_config
from train import ensure_cache


def derive_from_cache(cache_path: str | Path, *, device: torch.device | None = None) -> dict:
    device = device or torch.device("cpu")
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)

    targets = torch.tensor(cache["targets"], dtype=torch.float32, device=device)
    zeros = torch.zeros_like(targets)
    pca = PCALoss(cache["pca_models"], cache["outputs"], device=device)

    profile_scales: dict[str, float] = {}
    for name, start, end in output_slices(cache["outputs"]):
        pred = zeros[:, start:end]
        true = targets[:, start:end]
        components = getattr(pca, f"{name}_components")
        mean = getattr(pca, f"{name}_mean")
        pred_profiles = torch_reconstruct_profile(pred, components, mean)
        true_profiles = torch_reconstruct_profile(true, components, mean)
        profile_scales[name] = float(torch.nn.functional.mse_loss(pred_profiles, true_profiles).item())

    combined = CombinedPCALoss(
        cache["pca_models"],
        cache["outputs"],
        cache["weights"],
        device,
        profile_scales=profile_scales,
    )
    combined_pca_scale = float(combined.pca_loss(zeros, targets).item())
    combined_mse_scale = float(combined.weighted_mse_loss(zeros, targets).item())

    return {
        "dataset_tag": cache.get("dataset_tag"),
        "outputs": dict(cache["outputs"]),
        "profile_scales": {k: round(v, 4) for k, v in profile_scales.items()},
        "combined_pca_scale": round(combined_pca_scale, 4),
        "combined_mse_scale": round(combined_mse_scale, 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Derive loss_scales from a config or cache pickle")
    parser.add_argument("-c", "--config", default=None, help="Config JSON (builds/locates cache via ensure_cache)")
    parser.add_argument("--cache", default=None, help="Train-ready pickle path (skip config)")
    parser.add_argument("--update-config", action="store_true", help="Write derived scales into the config JSON")
    parser.add_argument("--round", type=int, default=4, help="Decimal places for printed scales")
    args = parser.parse_args()

    if not args.cache and not args.config:
        parser.error("Provide --cache or -c/--config")

    config_path = None
    if args.config:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = ROOT / config_path
        config_dict = json.loads(config_path.read_text())
        validate_config(config_dict)
        config = ConfigParser(config_dict, run_id="")
        cache_path = ensure_cache(config)
    else:
        cache_path = Path(args.cache)
        if not cache_path.is_absolute():
            cache_path = ROOT / cache_path

    scales = derive_from_cache(cache_path)
    if args.round != 4:
        scales["profile_scales"] = {k: round(v, args.round) for k, v in scales["profile_scales"].items()}
        scales["combined_pca_scale"] = round(scales["combined_pca_scale"], args.round)
        scales["combined_mse_scale"] = round(scales["combined_mse_scale"], args.round)

    loss_scales = {
        "profile_scales": scales["profile_scales"],
        "combined_pca_scale": scales["combined_pca_scale"],
        "combined_mse_scale": scales["combined_mse_scale"],
    }
    print(json.dumps({"cache": str(cache_path), **scales}, indent=2))

    if args.update_config and config_path is not None:
        config_dict = json.loads(config_path.read_text())
        config_dict["loss_scales"] = loss_scales
        config_path.write_text(json.dumps(config_dict, indent=4) + "\n")
        print(f"Updated {config_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
