#!/usr/bin/env python3
"""Auto diagnostics for residual patch model runs (Module G)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import data_loader.data_loaders as module_data
import model.model as module_arch
from eval_run import _resolve_eval_batch_size
from parse_config import ConfigParser
from train import ensure_cache, set_seed


from residual_utils.diagnostics import normalization_report as _normalization_report


def _collect_forward(model, loader, device):
    base_list, delta_list, full_list, gates = [], [], [], []
    with torch.no_grad():
        for data, _target, _idx in loader:
            x = data.to(device)
            base = model.forward_base(x).cpu().numpy()
            delta = model.forward_delta(x).cpu().numpy()
            full = model(x).cpu().numpy()
            base_list.append(base)
            delta_list.append(delta)
            full_list.append(full)
            gate = model.gate.detach().cpu().numpy()
            gates.append(np.atleast_1d(gate).ravel())
    return (
        np.vstack(base_list),
        np.vstack(delta_list),
        np.vstack(full_list),
        np.concatenate(gates),
    )


def main():
    parser = argparse.ArgumentParser(description="Residual model diagnostics")
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-r", "--resume", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

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
    cache_path = ensure_cache(config)

    import pickle

    with open(cache_path, "rb") as f:
        cache = pickle.load(f)

    dl_args = dict(config["data_loader"]["args"])
    dl_args["split"] = args.split
    dl_args["shuffle"] = False
    _resolve_eval_batch_size(dl_args, args.split)
    loader = getattr(module_data, config["data_loader"]["type"])(**dl_args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = config.init_obj("arch", module_arch).to(device)
    ckpt = torch.load(args.resume, map_location=device, weights_only=False)
    state = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
    model.load_state_dict(state)
    model.eval()

    base, delta, full, gate = _collect_forward(model, loader, device)
    delta_norm = np.linalg.norm(delta, axis=1)
    contrib = np.linalg.norm(full - base, axis=1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(delta_norm, bins=40, alpha=0.8)
    ax.set_xlabel("||ΔPC||")
    ax.set_ylabel("count")
    ax.set_title("Residual magnitude")
    fig.tight_layout()
    fig.savefig(out_dir / "residual_magnitude_hist.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(contrib, bins=40, alpha=0.8, color="tab:orange")
    ax.set_xlabel("||output - base||")
    ax.set_ylabel("count")
    ax.set_title("Residual contribution")
    fig.tight_layout()
    fig.savefig(out_dir / "residual_contribution_hist.png", dpi=120)
    plt.close(fig)

    report = {
        "checkpoint": args.resume,
        "split": args.split,
        "gate": gate.tolist(),
        "mean_delta_norm": float(delta_norm.mean()),
        "mean_contribution_norm": float(contrib.mean()),
        "normalization_report": _normalization_report(cache),
        "ablation": {
            "mean_abs_diff_full_vs_base": float(np.mean(np.abs(full - base))),
            "fraction_nonzero_contrib": float(np.mean(contrib > 1e-8)),
        },
    }
    (out_dir / "residual_diagnostics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
