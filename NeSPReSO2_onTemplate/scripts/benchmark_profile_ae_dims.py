#!/usr/bin/env python3
"""Sweep profile AE encoding dimensions (powers of 2) vs PCA baseline."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from parse_config import ConfigParser, validate_config
sys.path.insert(0, str(ROOT / "scripts"))
from train_profile_ae import pca_baseline_rmse, profiles_sample_major, train_variable
from train import ensure_cache

DEFAULT_DIMS = [16, 32, 64, 128, 256]


def print_table(rows: list[dict]) -> None:
    header = f"{'dim':>6} {'variable':<12} {'PCA RMSE':>10} {'AE RMSE':>10} {'AE/PCA':>8} {'train_s':>8}"
    print(header)
    print("-" * len(header))
    for r in rows:
        ratio = r["ae_val_rmse"] / r["pca_rmse"] if r["pca_rmse"] > 0 else float("nan")
        print(
            f"{r['encoding_dim']:>6} {r['variable']:<12} {r['pca_rmse']:>10.6f} "
            f"{r['ae_val_rmse']:>10.6f} {ratio:>8.3f} {r.get('train_sec', 0):>8.1f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep AE encoding dimensions")
    parser.add_argument("-c", "--config", required=True, type=str)
    parser.add_argument("--dims", default=",".join(str(d) for d in DEFAULT_DIMS), type=str)
    parser.add_argument("--arch", default="Autoencoder", choices=["Autoencoder", "KAN_Autoencoder"])
    parser.add_argument("--variable", default="all")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--out", default=None, type=str)
    parser.add_argument("-d", "--device", default=None, type=str)
    args = parser.parse_args()

    if args.device is not None:
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dims = sorted({int(x) for x in args.dims.split(",") if x.strip()})

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config_dict = json.loads(config_path.read_text())
    validate_config(config_dict)
    config = ConfigParser(config_dict, run_id="")

    cache_path = ensure_cache(config)
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)

    outputs = dict(cache["outputs"])
    vars_to_run = list(outputs.keys()) if args.variable == "all" else [args.variable]
    tag = cache.get("dataset_tag", config_dict["io"].get("dataset_tag", "unknown"))
    seed = int(config_dict.get("seed", 42))

    import time

    rows: list[dict] = []
    for dim in dims:
        for name in vars_to_run:
            prof = profiles_sample_major(cache, name)
            mask = np.isnan(prof)
            prof = np.nan_to_num(prof, nan=0.0)
            n_comp = min(dim, prof.shape[0] - 1, prof.shape[1] - 1)
            pca_rmse = pca_baseline_rmse(prof, mask, n_comp)

            t0 = time.perf_counter()
            _, stats = train_variable(
                profiles=prof,
                mask=mask,
                arch=args.arch,
                encoding_dim=dim,
                device=device,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=1e-3,
                val_frac=0.15,
                seed=seed + dim,
            )
            train_sec = time.perf_counter() - t0

            row = {
                "encoding_dim": dim,
                "variable": name,
                "pca_rmse": pca_rmse,
                "ae_val_rmse": stats["val_rmse"],
                "pca_n_components": n_comp,
                "train_sec": train_sec,
                "depth": stats["depth"],
            }
            rows.append(row)
            print(f"dim={dim} {name}: PCA={pca_rmse:.6f} AE={stats['val_rmse']:.6f}")

    print()
    print_table(rows)

    out = {
        "config": str(config_path),
        "cache": cache_path,
        "dataset_tag": tag,
        "arch": args.arch,
        "dims": dims,
        "epochs": args.epochs,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": rows,
    }
    out_path = (
        Path(args.out)
        if args.out
        else ROOT / "saved" / "benchmarks" / f"ae_dims_{tag}_{args.arch}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
