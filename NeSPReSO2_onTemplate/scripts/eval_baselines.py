#!/usr/bin/env python3
"""Evaluate climatology-only and GEM baselines on the test split."""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from base.split_utils import build_split_indices  # noqa: E402
from base.util import read_json  # noqa: E402
from eval_run import raw_profile_rmse, write_report  # noqa: E402
from parse_config import validate_config  # noqa: E402
from preproc.export_v2_cache import build_argo_cache  # noqa: E402


def _gem_predict(train_idx, test_idx, ssh_sla, clim_profiles, profiles, outputs):
    """Per-depth regression anomaly(z) = a(z)·SLA + b(z), fit on train, applied to test."""
    pred = {name: np.array(clim_profiles[name], copy=True) for name in outputs}
    x_train = ssh_sla[train_idx]
    sla_test = ssh_sla[test_idx]
    sla_test_ok = np.isfinite(sla_test)
    for name in outputs:
        anom = np.asarray(profiles[name], dtype=np.float64) - np.asarray(
            clim_profiles[name], dtype=np.float64
        )
        n_z, _ = anom.shape
        for iz in range(n_z):
            y_train = anom[iz, train_idx]
            valid = np.isfinite(x_train) & np.isfinite(y_train)
            if valid.sum() < 5:
                continue
            A = np.column_stack([x_train[valid], np.ones(valid.sum())])
            a, b = np.linalg.lstsq(A, y_train[valid], rcond=None)[0]
            gem_anom = a * np.where(sla_test_ok, sla_test, 0.0) + b
            pred[name][iz, test_idx] = clim_profiles[name][iz, test_idx] + np.where(
                sla_test_ok, gem_anom, 0.0
            )
    return pred


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--saved-dir", default="saved")
    args = parser.parse_args(argv)

    cfg = read_json(args.config)
    validate_config(cfg)
    if not cfg.get("io", {}).get("anomaly_targets"):
        raise ValueError("config must have io.anomaly_targets=true")

    cache_path = build_argo_cache(cfg)
    import pickle

    with open(cache_path, "rb") as f:
        cache = pickle.load(f)

    dl_args = cfg["data_loader"]["args"]
    split_indices = build_split_indices(
        len(cache["JULD"]),
        cache["JULD"],
        dl_args,
        dataset_tag=cache.get("dataset_tag", "argo_v2"),
        v2_src=cfg.get("io", {}).get("v2_src"),
    )
    test_idx = np.asarray(split_indices[args.split], dtype=int)
    train_idx = np.asarray(split_indices["train"], dtype=int)

    outputs = OrderedDict(cache["outputs"])
    profiles = cache["profiles"]

    saved = Path(args.saved_dir)
    if not saved.is_absolute():
        saved = _ROOT / saved
    saved.mkdir(parents=True, exist_ok=True)

    # Climatology-only, on the native depth grid (same grid eval_run.py uses)
    truth = {name: np.asarray(profiles[name], dtype=np.float64)[:, test_idx] for name in outputs}
    clim_rmse = {}
    for name in outputs:
        diff = np.asarray(cache["clim_profiles"][name], dtype=np.float64)[:, test_idx] - truth[name]
        clim_rmse[name] = float(np.sqrt(np.nanmean(diff ** 2)))

    clim_report = {
        "checkpoint": "baseline:climatology",
        "cache": str(cache_path),
        "dataset_tag": cache.get("dataset_tag"),
        "split": args.split,
        "n_samples": int(len(test_idx)),
        "loss": None,
        "raw_profile_rmse": clim_rmse,
    }
    write_report(clim_report, saved / f"eval_clim_{args.split}.json")

    # GEM baseline
    gem_phys = _gem_predict(
        train_idx, test_idx, cache["ssh_obs_sla"], cache["clim_profiles"], profiles, outputs
    )
    gem_rmse = {}
    for name in outputs:
        diff = np.asarray(gem_phys[name], dtype=np.float64)[:, test_idx] - truth[name]
        gem_rmse[name] = float(np.sqrt(np.nanmean(diff ** 2)))
    gem_report = {
        "checkpoint": "baseline:gem_sla",
        "cache": str(cache_path),
        "dataset_tag": cache.get("dataset_tag"),
        "split": args.split,
        "n_samples": int(len(test_idx)),
        "loss": None,
        "raw_profile_rmse": gem_rmse,
    }
    write_report(gem_report, saved / f"eval_gem_{args.split}.json")
    print(json.dumps({"clim": clim_report, "gem": gem_report}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
