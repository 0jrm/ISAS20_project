#!/usr/bin/env python3
"""Export train-ready cache from Zarr cube + feature operators (Component B)."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model.loss import get_pca_weights, true_profiles_numpy  # noqa: E402
from preproc.climatology import _profiles_depth_major  # noqa: E402
from preproc.cube.cube_schema import cube_hash_metadata, default_cube_path  # noqa: E402
from preproc.features.operators import list_operators  # noqa: E402
from preproc.features.sampler import CubeProvider, expand_feature_names  # noqa: E402
from preproc.preproc_isas_sat import write_train_cache  # noqa: E402
from base.split_utils import build_split_indices  # noqa: E402


HARMONIC_NAMES = frozenset({"timecos", "timesin", "latcos", "latsin", "loncos", "lonsin"})

POINT_CUBE_FEATURES: Dict[str, Any] = {
    "spec_version": 1,
    "scalars": ["timecos", "timesin", "latcos", "latsin", "loncos", "lonsin"],
    "point_backbone_inputs": ["sss.value@local", "sst.value@local", "ssh.value@local"],
    "operators": [
        {"op": "value", "channels": ["sst", "sss", "ssh"], "scales": ["local"]},
    ],
}

DEFAULT_PCA_CKPT = "saved/models/NeSPReSO2_ARGO_GoM/argo16_scales/model_best.pth"


def _cube_meta_or_missing(cube_path: Path) -> dict[str, Any]:
    if cube_path.exists():
        return cube_hash_metadata(cube_path)
    return {
        "cube_schema_version": 0,
        "data_revision": 0,
        "data_through": "missing",
    }


def _hash_payload(config: Dict, *, cache_kind: str | None = None, features: Dict | None = None) -> dict[str, Any]:
    io_cfg = config.get("io", {})
    features = features if features is not None else config.get("features", {})
    cube_path = Path(io_cfg.get("cube_path", default_cube_path(_ROOT)))
    cube_meta = _cube_meta_or_missing(cube_path)
    dl = config.get("data_loader", {}).get("args", {})
    payload: dict[str, Any] = {
        "features": features,
        "operator_versions": list_operators(),
        "cube_schema_version": cube_meta["cube_schema_version"],
        "data_revision": cube_meta.get("data_revision", 0),
        "cube_data_through": cube_meta["data_through"],
        "io": {
            "dataset_tag": io_cfg.get("dataset_tag"),
            "cube_path": str(cube_path),
            "v2_pickle": io_cfg.get("v2_pickle"),
        },
        "outputs": config.get("outputs"),
        "split_def": {
            "split_mode": dl.get("split_mode"),
            "train_frac": dl.get("train_frac"),
            "val_frac": dl.get("val_frac"),
            "test_frac": dl.get("test_frac"),
        },
    }
    if cache_kind is not None:
        payload["cache_kind"] = cache_kind
    return payload


def cube_feature_hash(config: Dict) -> str:
    payload = _hash_payload(config)
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def point_cube_cache_hash(config: Dict) -> str:
    features = config.get("features") or POINT_CUBE_FEATURES
    payload = _hash_payload(config, cache_kind="point_cube", features=features)
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def _resolve_pca_ckpt(config: Dict) -> str:
    io_cfg = config.get("io", {})
    arch_args = config.get("arch", {}).get("args", {})
    for key in ("pca_ckpt", "warmstart_ckpt"):
        path = io_cfg.get(key) or arch_args.get(key)
        if path and Path(path).is_file():
            return str(path)
    if Path(DEFAULT_PCA_CKPT).is_file():
        return DEFAULT_PCA_CKPT
    raise FileNotFoundError(
        "PCA checkpoint required for cube cache export "
        f"(io.pca_ckpt, io.warmstart_ckpt, arch.args.warmstart_ckpt, or {DEFAULT_PCA_CKPT})"
    )


def _load_v2_profiles(config: Dict, outputs: OrderedDict) -> tuple[dict, Any, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    io_cfg = config["io"]
    pickle_path = io_cfg.get("v2_pickle")
    v2_src = io_cfg.get("v2_src")
    if v2_src:
        sys.path.insert(0, str(v2_src))
    from astropy.time import Time
    from nespreso.data.pickle_compat import load_dataset_pickle
    from nespreso.utils.time import datenum_to_datetime

    data = load_dataset_pickle(pickle_path)
    ds = data["full_dataset"]
    n = len(ds)
    max_samples = io_cfg.get("max_samples")
    if max_samples is not None:
        n = min(n, int(max_samples))

    lat = np.asarray(ds.LAT[:n], dtype=np.float64)
    lon = np.asarray(ds.LON[:n], dtype=np.float64)
    juld = np.asarray(ds.TIME[:n], dtype=np.float64)
    dates_jd = np.array([Time(datenum_to_datetime(float(t))).jd for t in juld], dtype=np.float64)

    pres = np.arange(ds.min_depth, ds.max_depth + 1, dtype=np.float32)
    raw_profiles = {}
    for name in outputs:
        if name == "temperature":
            raw_profiles[name] = np.asarray(ds.TEMP, dtype=np.float32)
        elif name == "salinity":
            raw_profiles[name] = np.asarray(ds.SAL, dtype=np.float32)
        else:
            raise KeyError(name)
    full_n = len(ds)
    if max_samples is not None:
        sliced = {}
        for name, arr in raw_profiles.items():
            if arr.shape[0] == full_n:
                sliced[name] = arr[:n]
            elif arr.shape[1] == full_n:
                sliced[name] = arr[:, :n]
            else:
                sliced[name] = arr
        raw_profiles = sliced
    profiles = _profiles_depth_major(raw_profiles, n, pres)
    return io_cfg, ds, n, lat, lon, juld, dates_jd, profiles, pres


def _fill_harmonics(values: np.ndarray, names: list[str], juld: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> None:
    name_to_j = {n: i for i, n in enumerate(names)}
    day = juld % 365.0
    for i in range(len(juld)):
        if "timecos" in name_to_j:
            values[i, name_to_j["timecos"]] = np.cos(2 * np.pi * day[i] / 365.0)
        if "timesin" in name_to_j:
            values[i, name_to_j["timesin"]] = np.sin(2 * np.pi * day[i] / 365.0)
        if "latcos" in name_to_j:
            values[i, name_to_j["latcos"]] = np.cos(np.pi * lat[i] / 180.0)
        if "latsin" in name_to_j:
            values[i, name_to_j["latsin"]] = np.sin(np.pi * lat[i] / 180.0)
        if "loncos" in name_to_j:
            values[i, name_to_j["loncos"]] = np.cos(np.pi * lon[i] / 180.0)
        if "lonsin" in name_to_j:
            values[i, name_to_j["lonsin"]] = np.sin(np.pi * lon[i] / 180.0)


def _load_point_pca_from_ckpt(ckpt_path: str, outputs: OrderedDict) -> tuple[dict, dict]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    pca_models = ckpt["pca_models"]
    pcs_by_name = {}
    for name in outputs:
        pca = pca_models[name]
        # Reconstruct pcs from v2 not in ckpt — caller must supply profile pcs
        pcs_by_name[name] = None
    return pca_models, pcs_by_name


def build_feature_cache(config: Dict, force: bool = False) -> str:
    io_cfg = config["io"]
    if io_cfg.get("refit_pca"):
        raise ValueError("argo_cube residual configs must set io.refit_pca=false (reuse point PCA)")

    cache_dir = io_cfg.get("cache_dir", "data/cache")
    chash = cube_feature_hash(config)
    cache_path = Path(cache_dir) / f"train_ready_{chash}.pkl"
    if cache_path.exists() and not force:
        return str(cache_path)

    features_cfg = config.get("features")
    if not features_cfg:
        raise ValueError("config must include 'features' block for argo_cube export")

    pickle_path = io_cfg.get("v2_pickle")
    v2_src = io_cfg.get("v2_src")
    if v2_src:
        sys.path.insert(0, str(v2_src))
    from astropy.time import Time
    from nespreso.data.pickle_compat import load_dataset_pickle
    from nespreso.utils.time import datenum_to_datetime

    outputs = OrderedDict(config["outputs"])
    data = load_dataset_pickle(pickle_path)
    ds = data["full_dataset"]
    n = len(ds)
    max_samples = io_cfg.get("max_samples")
    if max_samples is not None:
        n = min(n, int(max_samples))

    lat = np.asarray(ds.LAT[:n], dtype=np.float64)
    lon = np.asarray(ds.LON[:n], dtype=np.float64)
    juld = np.asarray(ds.TIME[:n], dtype=np.float64)
    dates_jd = np.array([Time(datenum_to_datetime(float(t))).jd for t in juld], dtype=np.float64)

    cube_path = io_cfg.get("cube_path")
    provider = CubeProvider(cube_path)
    print(f"[export] sampling {n} profiles ({len(features_cfg.get('operators', []))} operator groups)...", file=sys.stderr, flush=True)
    table = provider.sample(features_cfg, lat, lon, dates_jd)
    print(f"[export] done: {len(table.names)} features", file=sys.stderr, flush=True)
    _fill_harmonics(table.values, table.names, juld, lat, lon)

    # Point backbone raw inputs (C-I2)
    backbone_names = features_cfg.get(
        "point_backbone_inputs",
        ["sss.value@local", "sst.value@local", "ssh.value@local"],
    )
    name_to_j = {n: i for i, n in enumerate(table.names)}
    harmonic_block = np.stack(
        [
            np.cos(2 * np.pi * (juld % 365.0) / 365.0),
            np.sin(2 * np.pi * (juld % 365.0) / 365.0),
            np.cos(np.pi * lat / 180.0),
            np.sin(np.pi * lat / 180.0),
            np.cos(np.pi * lon / 180.0),
            np.sin(np.pi * lon / 180.0),
        ],
        axis=1,
    ).astype(np.float32)
    sat_block = np.stack([table.values[:, name_to_j[n]] for n in backbone_names], axis=1).astype(np.float32)
    point_raw = np.concatenate([harmonic_block, sat_block], axis=1)

    # Z-score residual features on train split only; harmonics pass through
    feat_raw = table.values.astype(np.float32)
    valid_mask = table.valid_mask.copy()
    dl_cfg = config.get("data_loader", {}).get("args", {})
    split_indices = build_split_indices(
        n,
        juld,
        dl_cfg,
        dataset_tag=io_cfg.get("dataset_tag", "argo_cube"),
        v2_src=v2_src,
    )
    tr = np.asarray(split_indices["train"], dtype=int)
    mu = np.zeros(feat_raw.shape[1], dtype=np.float32)
    sigma = np.ones(feat_raw.shape[1], dtype=np.float32)
    for j, nm in enumerate(table.names):
        if nm in HARMONIC_NAMES:
            continue
        col = feat_raw[tr, j]
        m = col[np.isfinite(col)]
        if m.size == 0:
            raise ValueError(f"no finite train values for feature {nm}")
        mu[j] = float(np.mean(m))
        sigma[j] = max(float(np.std(m)), 1e-6)
    feat_z = ((feat_raw - mu) / sigma).astype(np.float32)
    for j, nm in enumerate(table.names):
        if nm in HARMONIC_NAMES:
            feat_z[:, j] = feat_raw[:, j]
        impute = mu[j]
        bad = ~valid_mask[:, j]
        feat_z[bad, j] = 0.0 if nm in HARMONIC_NAMES else 0.0  # already z-scored mean=0

    inputs = np.concatenate([point_raw, feat_z], axis=1).astype(np.float32)

    pres = np.arange(ds.min_depth, ds.max_depth + 1, dtype=np.float32)
    raw_profiles = {}
    for name in outputs:
        if name == "temperature":
            raw_profiles[name] = np.asarray(ds.TEMP, dtype=np.float32)
        elif name == "salinity":
            raw_profiles[name] = np.asarray(ds.SAL, dtype=np.float32)
        else:
            raise KeyError(name)
    full_n = len(ds)
    if max_samples is not None:
        sliced = {}
        for name, arr in raw_profiles.items():
            if arr.shape[0] == full_n:
                sliced[name] = arr[:n]
            elif arr.shape[1] == full_n:
                sliced[name] = arr[:, :n]
            else:
                sliced[name] = arr
        raw_profiles = sliced
    profiles = _profiles_depth_major(raw_profiles, n, pres)

    pca_ckpt = _resolve_pca_ckpt(config)
    ckpt = torch.load(pca_ckpt, map_location="cpu", weights_only=False)
    pca_models = ckpt["pca_models"]

    target_cols = []
    pcs_by_name = {}
    for name, n_comp in outputs.items():
        pca = pca_models[name]
        prof = profiles[name]  # depth-major (n_depth, n)
        pcs = pca.transform(prof.T.astype(np.float64))[:, :n_comp].T.astype(np.float32)
        pcs_by_name[name] = pcs
        target_cols.append(pcs.T.astype(np.float32))

    targets = np.hstack(target_cols).astype(np.float32)
    weights = get_pca_weights(pca_models, pcs_by_name, list(outputs.keys()))

    payload = {
        "inputs": inputs,
        "targets": targets,
        "pca_models": pca_models,
        "pcs_by_name": pcs_by_name,
        "outputs": outputs,
        "weights": weights,
        "profiles": profiles,
        "true_profiles": true_profiles_numpy(targets, pca_models, outputs),
        "LAT": lat.astype(np.float32),
        "LON": lon.astype(np.float32),
        "PRES": pres,
        "JULD": juld.astype(np.float32),
        "station_indices": np.arange(n, dtype=np.int64),
        "input_params": dict(config.get("input_params", {})),
        "spatial_pad": int(io_cfg.get("spatial_pad", 2)),
        "temporal_pad": int(io_cfg.get("temporal_pad", 6)),
        "sat_patch_shape": None,
        "config_hash": chash,
        "dataset_tag": io_cfg.get("dataset_tag", "argo_cube"),
        "min_depth": int(ds.min_depth),
        "max_depth": int(ds.max_depth),
        "feature_names": table.names,
        "residual_feature_names": table.names,
        "feature_units": table.units,
        "valid_mask": valid_mask,
        "input_standardization": {
            "mean": mu,
            "std": sigma,
            "feature_names": table.names,
            "normalization_version": "train_zscore_v1_harmonic_passthrough",
            "train_indices": tr,
        },
        "base_dim": point_raw.shape[1],
        "feat_dim": feat_z.shape[1],
        "point_backbone_inputs": backbone_names,
        "provenance": {
            "cube_path": str(provider.cube_path),
            "feature_spec_version": features_cfg.get("spec_version", 1),
            "operator_versions": list_operators(),
        },
    }
    return write_train_cache(payload, cache_dir, chash)


def build_point_cube_cache(config: Dict, force: bool = False) -> str:
    io_cfg = config["io"]
    if io_cfg.get("refit_pca"):
        raise ValueError("argo_cube point cache must set io.refit_pca=false (reuse point PCA)")

    cache_dir = io_cfg.get("cache_dir", "data/cache")
    features_cfg = config.get("features") or POINT_CUBE_FEATURES
    chash = point_cube_cache_hash({**config, "features": features_cfg})
    cache_path = Path(cache_dir) / f"train_ready_{chash}.pkl"
    if cache_path.exists() and not force:
        return str(cache_path)

    outputs = OrderedDict(config["outputs"])
    io_cfg, ds, n, lat, lon, juld, dates_jd, profiles, pres = _load_v2_profiles(config, outputs)
    v2_src = io_cfg.get("v2_src")

    cube_path = io_cfg.get("cube_path")
    provider = CubeProvider(cube_path)
    table = provider.sample(features_cfg, lat, lon, dates_jd)
    _fill_harmonics(table.values, table.names, juld, lat, lon)

    backbone_names = features_cfg.get(
        "point_backbone_inputs",
        ["sss.value@local", "sst.value@local", "ssh.value@local"],
    )
    name_to_j = {n: i for i, n in enumerate(table.names)}
    harmonic_block = np.stack(
        [
            np.cos(2 * np.pi * (juld % 365.0) / 365.0),
            np.sin(2 * np.pi * (juld % 365.0) / 365.0),
            np.cos(np.pi * lat / 180.0),
            np.sin(np.pi * lat / 180.0),
            np.cos(np.pi * lon / 180.0),
            np.sin(np.pi * lon / 180.0),
        ],
        axis=1,
    ).astype(np.float32)
    sat_block = np.stack([table.values[:, name_to_j[n]] for n in backbone_names], axis=1).astype(np.float32)
    point_raw = np.concatenate([harmonic_block, sat_block], axis=1)

    dl_cfg = config.get("data_loader", {}).get("args", {})
    split_indices = build_split_indices(
        n,
        juld,
        dl_cfg,
        dataset_tag=io_cfg.get("dataset_tag", "argo_cube"),
        v2_src=v2_src,
    )
    tr = np.asarray(split_indices["train"], dtype=int)
    sat_cols = np.array([6, 7, 8], dtype=int)
    mu_point = np.zeros(point_raw.shape[1], dtype=np.float32)
    sigma_point = np.ones(point_raw.shape[1], dtype=np.float32)
    for j in sat_cols:
        col = point_raw[tr, j]
        m = col[np.isfinite(col)]
        if m.size == 0:
            raise ValueError(f"no finite train values for point column {j}")
        mu_point[j] = float(np.mean(m))
        sigma_point[j] = max(float(np.std(m)), 1e-6)
    inputs = point_raw.copy()
    for j in sat_cols:
        inputs[:, j] = (point_raw[:, j] - mu_point[j]) / sigma_point[j]

    pca_ckpt = _resolve_pca_ckpt(config)
    ckpt = torch.load(pca_ckpt, map_location="cpu", weights_only=False)
    pca_models = ckpt["pca_models"]

    target_cols = []
    pcs_by_name = {}
    for name, n_comp in outputs.items():
        pca = pca_models[name]
        prof = profiles[name]
        pcs = pca.transform(prof.T.astype(np.float64))[:, :n_comp].T.astype(np.float32)
        pcs_by_name[name] = pcs
        target_cols.append(pcs.T.astype(np.float32))

    targets = np.hstack(target_cols).astype(np.float32)
    weights = get_pca_weights(pca_models, pcs_by_name, list(outputs.keys()))

    payload = {
        "inputs": inputs.astype(np.float32),
        "targets": targets,
        "pca_models": pca_models,
        "pcs_by_name": pcs_by_name,
        "outputs": outputs,
        "weights": weights,
        "profiles": profiles,
        "true_profiles": true_profiles_numpy(targets, pca_models, outputs),
        "LAT": lat.astype(np.float32),
        "LON": lon.astype(np.float32),
        "PRES": pres,
        "JULD": juld.astype(np.float32),
        "station_indices": np.arange(n, dtype=np.int64),
        "spatial_pad": int(io_cfg.get("spatial_pad", 0)),
        "temporal_pad": int(io_cfg.get("temporal_pad", 0)),
        "sat_patch_shape": None,
        "config_hash": chash,
        "dataset_tag": io_cfg.get("dataset_tag", "argo_cube"),
        "cache_kind": "point_cube",
        "min_depth": int(ds.min_depth),
        "max_depth": int(ds.max_depth),
        "feature_names": backbone_names,
        "point_backbone_inputs": backbone_names,
        "input_standardization": {
            "mean": mu_point,
            "std": sigma_point,
            "feature_names": [
                "timecos", "timesin", "latcos", "latsin", "loncos", "lonsin",
                *backbone_names,
            ],
            "zscore_columns": sat_cols.tolist(),
            "normalization_version": "train_zscore_point_sat_only_v1",
            "train_indices": tr,
        },
        "base_dim": inputs.shape[1],
        "provenance": {
            "cube_path": str(provider.cube_path),
            "feature_spec_version": features_cfg.get("spec_version", 1),
            "operator_versions": list_operators(),
            "pca_ckpt": pca_ckpt,
            "cache_kind": "point_cube",
        },
    }
    return write_train_cache(payload, cache_dir, chash)


def main(argv: list[str] | None = None) -> int:
    from base.util import read_json
    from parse_config import validate_config

    parser = argparse.ArgumentParser(description="Export cube feature cache")
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--point-only", action="store_true", help="export 9-D point-cube cache")
    args = parser.parse_args(argv)
    cfg = read_json(args.config)
    validate_config(cfg)
    if args.point_only:
        path = build_point_cube_cache(cfg, force=args.force)
    else:
        path = build_feature_cache(cfg, force=args.force)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
