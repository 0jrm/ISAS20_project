#!/usr/bin/env python3
"""Stratified evaluation: RMSE/bias by L3 coverage, track distance, and census subsets."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

_ROOT = Path(__file__).resolve().parent
if str(_ROOT / "notebooks") not in sys.path:
    sys.path.insert(0, str(_ROOT / "notebooks"))

from nb_metrics import (  # noqa: E402
    COMMON_DEPTH_M,
    align_profiles_to_depth,
    common_depth_mask,
    depth_meters,
    pcs_to_profiles_depth_major,
    profiles_depth_major,
    run_inference,
)
from parse_config import ConfigParser, validate_config
from base.util import read_json

# ponytail: fixed bins; upgrade path = data-driven quantiles when n is large
COVERAGE_BIN_EDGES = (0.0, 1e-6, 0.01, 0.05, 1.0)
COVERAGE_BIN_LABELS = ("zero", "trace", "low", "mid", "high")

TRACK_KM_BIN_EDGES = (0.0, 25.0, 50.0, 100.0, float("inf"))
TRACK_KM_BIN_LABELS = ("0-25km", "25-50km", "50-100km", "100km+")

EVAL_YEAR_SUBSETS: dict[str, list[int]] = {
    "high_observation_2020": [2020],
    "low_observation_2015_2018": [2015, 2016, 2017, 2018],
}


def _git_commit() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_ROOT.parent,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def bin_coverage(value: float) -> str:
    v = 0.0 if not np.isfinite(value) else float(value)
    for i in range(len(COVERAGE_BIN_LABELS)):
        lo, hi = COVERAGE_BIN_EDGES[i], COVERAGE_BIN_EDGES[i + 1]
        if i == len(COVERAGE_BIN_LABELS) - 1:
            if lo <= v <= hi:
                return COVERAGE_BIN_LABELS[i]
        elif lo <= v < hi:
            return COVERAGE_BIN_LABELS[i]
    return COVERAGE_BIN_LABELS[-1]


def bin_track_km(value: float) -> str:
    if not np.isfinite(value):
        return "missing"
    v = float(value)
    for i in range(len(TRACK_KM_BIN_LABELS)):
        lo, hi = TRACK_KM_BIN_EDGES[i], TRACK_KM_BIN_EDGES[i + 1]
        if lo <= v < hi:
            return TRACK_KM_BIN_LABELS[i]
    return TRACK_KM_BIN_LABELS[-1]


def sample_years(
    cache: Mapping[str, Any],
    indices: Sequence[int],
    *,
    dataset_tag: str,
    v2_src: str | None,
) -> np.ndarray:
    from base.split_utils import sample_dates

    dates = sample_dates(cache["JULD"], dataset_tag=dataset_tag, v2_src=v2_src)
    idx = np.asarray(indices, dtype=int)
    return np.array([int(str(dates[i])[:4]) for i in idx], dtype=int)


def extract_l3_metrics(
    cache: Mapping[str, Any],
    indices: Sequence[int],
) -> dict[str, np.ndarray]:
    """Per evaluated sample (batch order): SSH coverage + nearest track distance."""
    l3 = cache.get("l3_tensors")
    n = len(indices)
    cov = np.zeros(n, dtype=np.float64)
    track = np.full(n, np.nan, dtype=np.float64)
    if not l3:
        return {"ssh_coverage_fraction": cov, "nearest_track_km": track}
    for j, gi in enumerate(indices):
        row = l3[int(gi)]
        if not row:
            continue
        ssh = (row.get("coverage") or {}).get("ssh") or {}
        cov[j] = float(ssh.get("coverage_fraction", 0.0))
        track[j] = float(ssh.get("nearest_track_km", np.nan))
    return {"ssh_coverage_fraction": cov, "nearest_track_km": track}


def profiles_on_common_grid(
    pcs: np.ndarray,
    indices: np.ndarray,
    cache: Mapping[str, Any],
    pca_models: Mapping,
    outputs: OrderedDict,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    """Return pred/true (n_depth_common, n_batch) per output on the common grid."""
    idx = np.asarray(indices, dtype=int)
    pred_all = pcs_to_profiles_depth_major(pcs, pca_models, outputs)
    z_native = depth_meters(cache)
    mask = common_depth_mask()
    z_common = COMMON_DEPTH_M[mask]
    pred: dict[str, np.ndarray] = {}
    true: dict[str, np.ndarray] = {}
    for name in outputs:
        pred_c = align_profiles_to_depth(pred_all[name], z_native, z_common)
        true_c = align_profiles_to_depth(profiles_depth_major(cache, name)[:, idx], z_native, z_common)
        pred[name] = pred_c
        true[name] = true_c
    return pred, true, z_common


def pooled_stratum_metrics(
    pred: Mapping[str, np.ndarray],
    true: Mapping[str, np.ndarray],
    sel: np.ndarray,
) -> dict[str, Any]:
    n = int(np.sum(sel))
    if n == 0:
        return {"n": 0, "rmse": {}, "bias": {}}
    out_rmse: dict[str, float] = {}
    out_bias: dict[str, float] = {}
    for name in pred:
        residual = pred[name][:, sel] - true[name][:, sel]
        out_rmse[name] = float(np.sqrt(np.nanmean(residual**2)))
        out_bias[name] = float(np.nanmean(residual))
    return {"n": n, "rmse": out_rmse, "bias": out_bias}


def build_strata_masks(
    *,
    years: np.ndarray,
    coverage: np.ndarray,
    track_km: np.ndarray,
) -> dict[str, np.ndarray]:
    n = len(years)
    masks: dict[str, np.ndarray] = {"all": np.ones(n, dtype=bool)}

    for label in COVERAGE_BIN_LABELS:
        masks[f"coverage_{label}"] = np.array(
            [bin_coverage(v) == label for v in coverage], dtype=bool
        )

    for label in TRACK_KM_BIN_LABELS + ("missing",):
        masks[f"track_{label}"] = np.array(
            [bin_track_km(v) == label for v in track_km], dtype=bool
        )

    for name, yrs in EVAL_YEAR_SUBSETS.items():
        yr_set = set(yrs)
        masks[f"subset_{name}"] = np.array([y in yr_set for y in years], dtype=bool)

    return masks


def stratified_report(
    config: ConfigParser,
    checkpoint_path: str,
    *,
    split: str = "test",
) -> dict[str, Any]:
    inf = run_inference(config, checkpoint_path, split=split)
    cache = inf["cache"]
    outputs: OrderedDict = inf["outputs"]
    indices = np.asarray(inf["indices"], dtype=int)

    io = config.config.get("io", {})
    dataset_tag = str(inf["dataset_tag"])
    v2_src = io.get("v2_src")

    pred, true, z_common = profiles_on_common_grid(
        inf["pcs"], indices, cache, inf["pca_models"], outputs
    )
    years = sample_years(cache, indices, dataset_tag=dataset_tag, v2_src=v2_src)
    l3 = extract_l3_metrics(cache, indices)
    masks = build_strata_masks(
        years=years,
        coverage=l3["ssh_coverage_fraction"],
        track_km=l3["nearest_track_km"],
    )

    strata: dict[str, Any] = {}
    for name, sel in masks.items():
        strata[name] = pooled_stratum_metrics(pred, true, sel)

    dl_args = dict(config["data_loader"]["args"])
    split_mode = dl_args.get("split_mode", "random")
    split_config = dl_args.get("split_config")

    return {
        "checkpoint": str(checkpoint_path),
        "cache": str(inf["cache_path"]),
        "dataset_tag": dataset_tag,
        "split": split,
        "split_mode": split_mode,
        "split_config": split_config,
        "n_samples": int(inf["n_samples"]),
        "l3_enabled": bool(cache.get("l3_enabled")),
        "depth_grid_m": z_common.tolist(),
        "strata_definitions": {
            "coverage_bins": {
                "edges": list(COVERAGE_BIN_EDGES),
                "labels": list(COVERAGE_BIN_LABELS),
            },
            "track_km_bins": {
                "edges": [e if np.isfinite(e) else "inf" for e in TRACK_KM_BIN_EDGES],
                "labels": list(TRACK_KM_BIN_LABELS) + ["missing"],
            },
            "year_subsets": EVAL_YEAR_SUBSETS,
        },
        "strata": strata,
        "git_commit": _git_commit(),
        "timestamp": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
    }


def to_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Stratified evaluation",
        "",
        f"- **Checkpoint:** `{report['checkpoint']}`",
        f"- **Cache:** `{report['cache']}`",
        f"- **Split:** {report['split']} ({report['split_mode']}, n={report['n_samples']})",
        f"- **L3 enabled:** {report['l3_enabled']}",
        "",
        "| Stratum | n | T RMSE | S RMSE | T bias | S bias |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, stats in report["strata"].items():
        if stats["n"] == 0:
            continue
        rmse = stats.get("rmse", {})
        bias = stats.get("bias", {})
        lines.append(
            f"| `{name}` | {stats['n']} | "
            f"{rmse.get('temperature', float('nan')):.4f} | "
            f"{rmse.get('salinity', float('nan')):.4f} | "
            f"{bias.get('temperature', float('nan')):.4f} | "
            f"{bias.get('salinity', float('nan')):.4f} |"
        )
    return "\n".join(lines) + "\n"


def main(config: ConfigParser, checkpoint_path: str, *, split: str = "test") -> dict[str, Any]:
    return stratified_report(config, checkpoint_path, split=split)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeSPReSO stratified test-split evaluation")
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-r", "--resume", required=True, help="checkpoint .pth")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("-d", "--device", default=None)
    parser.add_argument("--out", default=None, help="write JSON report to this path")
    parser.add_argument("--md-out", default=None, help="write Markdown summary to this path")
    args = parser.parse_args()
    if args.device:
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    cfg = read_json(args.config)
    validate_config(cfg)
    config = ConfigParser(cfg)
    report = main(config, args.resume, split=args.split)
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    if args.md_out:
        Path(args.md_out).write_text(to_markdown(report))
