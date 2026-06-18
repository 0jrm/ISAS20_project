"""Build mask-native L3 processed samples around ARGO profile targets."""

from __future__ import annotations

import pickle
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from base.split_utils import build_split_indices, sample_dates
from preproc.l3_rasterize import (
    FEATURE_NAMES,
    N_FEATURES,
    build_target_from_cache,
    collect_era5_paths,
    collect_ssh_paths,
    l3_config_hash,
    l3_geometry,
    patch_grid,
    rasterize_era5_wind_for_target,
    rasterize_ssh_for_target,
)
from preproc.preproc_isas_sat import config_hash, write_train_cache


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


def _resolve_path(root: Path, p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (root / path).resolve()


def load_argo_cache(config: dict) -> dict:
    io = config["io"]
    cache_dir = _resolve_path(_ROOT, io.get("cache_dir", "../data/cache"))
    chash = config_hash(config)
    cache_path = cache_dir / f"train_ready_{chash}.pkl"
    if not cache_path.is_file():
        from preproc.export_v2_cache import build_argo_cache

        build_argo_cache(config, force=False)
    with open(cache_path, "rb") as f:
        return pickle.load(f)


def select_indices(
    cache: dict,
    config: dict,
    *,
    max_samples: int | None,
    anchor_date: str | None,
) -> list[int]:
    n = len(np.asarray(cache["JULD"]).ravel())
    if max_samples is None or max_samples >= n:
        return list(range(n))

    io = config["io"]
    tag = io.get("dataset_tag", "argo_v2")
    v2_src = io.get("v2_src")
    if anchor_date:
        anchor = date.fromisoformat(anchor_date)
        dates = sample_dates(cache["JULD"], dataset_tag=tag, v2_src=v2_src)
        deltas = [abs((datetime.strptime(str(d)[:10], "%Y-%m-%d").date() - anchor).days) for d in dates]
        order = np.argsort(deltas)
        return [int(i) for i in order[:max_samples]]
    return list(range(max_samples))


def build_l3_sample(
    idx: int,
    cache: dict,
    l3_cfg: dict[str, Any],
    *,
    dataset_tag: str,
    v2_src: str | None,
    split: str,
) -> dict[str, Any]:
    raw_root = _resolve_path(_ROOT, l3_cfg["raw_root"])
    windows = list(l3_cfg["time_windows_hours"])
    half = float(l3_cfg["patch_half_deg"])
    step = float(l3_cfg["grid_step_deg"])
    target = build_target_from_cache(cache, idx, dataset_tag=dataset_tag, v2_src=v2_src)
    grid = patch_grid(target.lat, target.lon, half, step)

    ssh_paths = collect_ssh_paths(raw_root, target.time, windows)
    era5_paths = collect_era5_paths(raw_root, target.time, windows)

    ssh_bundle, ssh_cov = rasterize_ssh_for_target(ssh_paths, target, grid, windows)
    wind_u, wind_u_cov = rasterize_era5_wind_for_target(era5_paths, target, grid, windows, "u")
    wind_v, wind_v_cov = rasterize_era5_wind_for_target(era5_paths, target, grid, windows, "v")

    return {
        "target_idx": int(idx),
        "split": split,
        "target": {
            "lat": target.lat,
            "lon": target.lon,
            "time": target.time.isoformat(),
        },
        "ssh": ssh_bundle,
        "wind_u": wind_u,
        "wind_v": wind_v,
        "coverage": {
            "ssh": ssh_cov,
            "wind_u": wind_u_cov,
            "wind_v": wind_v_cov,
        },
        "sources": {
            "ssh_files": [str(p) for p in ssh_paths],
            "era5_files": [str(p) for p in era5_paths],
        },
    }


def build_l3_processed_batch(
    config: dict,
    *,
    indices: list[int] | None = None,
    max_samples: int | None = None,
    anchor_date: str | None = None,
    force: bool = False,
) -> str:
    io = config["io"]
    l3_cfg = io["l3"]
    processed_root = _resolve_path(_ROOT, l3_cfg["processed_root"])
    processed_root.mkdir(parents=True, exist_ok=True)

    lhash = l3_config_hash(l3_cfg)
    out_path = processed_root / f"l3_samples_{lhash}.pkl"
    if out_path.is_file() and not force:
        return str(out_path)

    cache = load_argo_cache(config)
    tag = io.get("dataset_tag", "argo_v2")
    v2_src = io.get("v2_src")
    if indices is None:
        indices = select_indices(cache, config, max_samples=max_samples, anchor_date=anchor_date)

    dl = dict(config.get("data_loader", {}).get("args", {}))
    dl.setdefault("split_seed", int(config.get("seed", 42)))
    n = len(np.asarray(cache["JULD"]).ravel())
    split_idx = build_split_indices(
        n,
        cache.get("JULD"),
        dl,
        dataset_tag=tag,
        v2_src=v2_src,
    )
    idx_to_split = {}
    for split_name, idxs in split_idx.items():
        for i in idxs:
            idx_to_split[i] = split_name

    spatial_pad, temporal_pad, grid_size = l3_geometry(l3_cfg)
    samples = []
    for idx in indices:
        split = idx_to_split.get(idx, "unassigned")
        samples.append(
            build_l3_sample(
                idx,
                cache,
                l3_cfg,
                dataset_tag=tag,
                v2_src=v2_src,
                split=split,
            )
        )

    payload = {
        "version": 1,
        "l3_hash": lhash,
        "config_hash": config_hash(config),
        "git_commit": _git_commit(),
        "features": list(FEATURE_NAMES),
        "n_features": N_FEATURES,
        "geometry": {
            "spatial_pad": spatial_pad,
            "temporal_pad": temporal_pad,
            "grid_size": grid_size,
            "patch_half_deg": float(l3_cfg["patch_half_deg"]),
            "grid_step_deg": float(l3_cfg["grid_step_deg"]),
            "time_windows_hours": list(l3_cfg["time_windows_hours"]),
        },
        "samples": samples,
    }
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"L3 processed batch saved to {out_path} (n={len(samples)})")
    return str(out_path)


def export_l3_train_cache(config: dict, processed_path: str | Path, force: bool = False) -> str:
    """Attach l3_tensors to ARGO train cache (legacy inputs unchanged)."""
    io = config["io"]
    cache_dir = _resolve_path(_ROOT, io.get("cache_dir", "../data/cache"))
    chash = config_hash(config)
    cache_path = cache_dir / f"train_ready_{chash}.pkl"
    l3_cache_path = cache_dir / f"train_ready_l3_{chash}_{l3_config_hash(io['l3'])}.pkl"
    if l3_cache_path.is_file() and not force:
        return str(l3_cache_path)

    with open(cache_path, "rb") as f:
        base = pickle.load(f)
    with open(processed_path, "rb") as f:
        processed = pickle.load(f)

    l3_by_idx = {s["target_idx"]: s for s in processed["samples"]}
    l3_tensors = []
    for i in range(len(base["inputs"])):
        s = l3_by_idx.get(i)
        if s is None:
            l3_tensors.append(None)
        else:
            l3_tensors.append(
                {
                    "ssh": s["ssh"],
                    "wind_u": s["wind_u"],
                    "wind_v": s["wind_v"],
                    "coverage": s["coverage"],
                }
            )

    payload = dict(base)
    payload["l3_tensors"] = l3_tensors
    payload["l3_geometry"] = processed["geometry"]
    payload["l3_features"] = processed["features"]
    payload["dataset_tag"] = io.get("dataset_tag", "argo_v2") + "_l3"
    payload["l3_processed_path"] = str(processed_path)
    return write_train_cache(payload, str(cache_dir), f"l3_{chash}_{l3_config_hash(io['l3'])}")
