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
from preproc.l3_input import build_l3_input_rows, l3_channel_metadata
from preproc.l4_augment import augment_config_summary, apply_l4_to_sample
from preproc.preproc_isas_sat import config_hash, count_encoding_dims, write_train_cache


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
    l4_cfg: dict[str, Any] | None = None,
    augment_seed: int = 0,
) -> dict[str, Any]:
    raw_root = _resolve_path(_ROOT, l3_cfg["raw_root"])
    windows = list(l3_cfg["time_windows_hours"])
    half = float(l3_cfg["patch_half_deg"])
    step = float(l3_cfg["grid_step_deg"])
    target = build_target_from_cache(cache, idx, dataset_tag=dataset_tag, v2_src=v2_src)
    grid = patch_grid(target.lat, target.lon, half, step)

    if not raw_root.is_dir():
        from preproc.l3_rasterize import coverage_metrics, empty_bundle

        t, h, w = len(windows), grid.height, grid.width
        empty = empty_bundle(t, h, w)
        cov = coverage_metrics(empty, target=target)
        sample = {
            "target_idx": int(idx),
            "split": split,
            "target": {"lat": target.lat, "lon": target.lon, "time": target.time.isoformat()},
            "ssh": empty,
            "wind_u": empty.copy(),
            "wind_v": empty.copy(),
            "coverage": {"ssh": cov, "wind_u": cov, "wind_v": cov},
            "sources": {"ssh_files": [], "era5_files": []},
        }
    else:
        ssh_paths = collect_ssh_paths(raw_root, target.time, windows)
        era5_paths = collect_era5_paths(raw_root, target.time, windows)

        ssh_bundle, ssh_cov = rasterize_ssh_for_target(ssh_paths, target, grid, windows)
        wind_u, wind_u_cov = rasterize_era5_wind_for_target(era5_paths, target, grid, windows, "u")
        wind_v, wind_v_cov = rasterize_era5_wind_for_target(era5_paths, target, grid, windows, "v")

        sample = {
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

    if l4_cfg and l4_cfg.get("enabled"):
        rng = np.random.default_rng(int(augment_seed))
        sample = apply_l4_to_sample(
            sample,
            target=target,
            grid=grid,
            raw_root=raw_root,
            windows_hours=windows,
            l4_cfg=l4_cfg,
            rng=rng,
        )
    return sample


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
    l4_cfg = io.get("l4") or {}
    processed_root = _resolve_path(_ROOT, l3_cfg["processed_root"])
    processed_root.mkdir(parents=True, exist_ok=True)

    lhash = l3_config_hash(l3_cfg, l4_cfg if l4_cfg.get("enabled") else None)
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
    seed_base = int(config.get("seed", 42))
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
                l4_cfg=l4_cfg if l4_cfg.get("enabled") else None,
                augment_seed=seed_base + int(idx),
            )
        )

    payload = {
        "version": 1,
        "l3_hash": lhash,
        "config_hash": config_hash(config),
        "git_commit": _git_commit(),
        "features": list(FEATURE_NAMES),
        "n_features": N_FEATURES,
        "l4_augment": augment_config_summary(l4_cfg) if l4_cfg.get("enabled") else None,
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
    """Build train-ready cache with L3-flattened ``inputs`` and ``l3_tensors``."""
    io = config["io"]
    l3_cfg = io["l3"]
    l4_cfg = io.get("l4") or {}
    cache_dir = _resolve_path(_ROOT, io.get("cache_dir", "../data/cache"))
    chash = config_hash(config)
    lhash = l3_config_hash(l3_cfg, l4_cfg if l4_cfg.get("enabled") else None)
    cache_tag = f"l3_{chash}_{lhash}"
    l3_cache_path = cache_dir / f"train_ready_{cache_tag}.pkl"
    if l3_cache_path.is_file() and not force:
        return str(l3_cache_path)

    with open(cache_dir / f"train_ready_{chash}.pkl", "rb") as f:
        base = pickle.load(f)
    with open(processed_path, "rb") as f:
        processed = pickle.load(f)

    l3_by_idx = {s["target_idx"]: s for s in processed["samples"]}
    n = len(base["inputs"])
    l3_tensors = []
    for i in range(n):
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

    n_enc = count_encoding_dims(config.get("input_params", {}))
    inputs = build_l3_input_rows(base["inputs"], l3_tensors, l3_cfg, n_enc=n_enc)
    spatial_pad, temporal_pad, grid_size = l3_geometry(l3_cfg)
    from preproc.l3_input import l3_sat_patch_shape

    c, t, h, w = l3_sat_patch_shape(l3_cfg)
    payload = dict(base)
    payload["inputs"] = inputs
    payload["l3_tensors"] = l3_tensors
    payload["l3_geometry"] = processed.get("geometry") or {
        "spatial_pad": spatial_pad,
        "temporal_pad": temporal_pad,
        "grid_size": grid_size,
    }
    payload["l3_features"] = processed.get("features", list(FEATURE_NAMES))
    payload["l3_channel_metadata"] = l3_channel_metadata(l3_cfg)
    payload["spatial_pad"] = spatial_pad
    payload["temporal_pad"] = temporal_pad
    payload["sat_patch_shape"] = [c, t, h, w]
    payload["l3_enabled"] = True
    payload["l4_enabled"] = bool(l4_cfg.get("enabled"))
    if l4_cfg.get("enabled"):
        payload["l4_augment"] = augment_config_summary(l4_cfg)
    payload["dataset_tag"] = io.get("dataset_tag", "argo_v2") + "_l3"
    payload["l3_processed_path"] = str(processed_path)
    return write_train_cache(payload, str(cache_dir), cache_tag)


def build_argo_l3_train_cache(config: dict, force: bool = False, max_samples: int | None = None) -> str:
    """Rasterize L3 (or empty bundles), flatten inputs, write ``train_ready_l3_*.pkl``."""
    processed_path = build_l3_processed_batch(
        config,
        max_samples=max_samples,
        anchor_date=None if max_samples is None else "2020-01-15",
        force=force,
    )
    return export_l3_train_cache(config, processed_path, force=force)
