#!/usr/bin/env python3
"""Build train-ready cache for point-anchored residual patch model."""

from __future__ import annotations

import argparse
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import h5py
import numpy as np
from sklearn.decomposition import PCA

from model.loss import get_pca_weights, true_profiles_numpy
from preproc.basin_stats import compute_basin_daily_means
from preproc.export_argo_l4_cache import _bottom_depth_from_bathy, _load_v2_dataset
from preproc.export_v2_cache import _refit_pca_from_profiles
from preproc.preproc_isas_sat import (
    apply_train_standardization,
    build_argo_residual_input_matrix,
    build_residual_feature_names,
    compute_argo_residual_input_dim,
    config_hash,
    extract_sat_values,
    resolve_use_mask_channels,
    residual_sat_patch_shape,
    write_train_cache,
)


def build_argo_residual_cache(config: Dict, force: bool = False, max_samples: int | None = None) -> str:
    io_cfg = config["io"]
    cache_dir = io_cfg.get("cache_dir", "../data/cache")
    chash = config_hash(config)
    cache_path = Path(cache_dir) / f"train_ready_{chash}.pkl"
    if cache_path.exists() and not force:
        return str(cache_path)

    spatial_pad = int(io_cfg.get("spatial_pad", 2))
    temporal_pad = int(io_cfg.get("temporal_pad", 6))
    input_params = dict(config["input_params"])
    outputs = OrderedDict(config["outputs"])
    center_relative = bool(io_cfg.get("center_relative_patches", True))
    include_bathy_scalar = bool(io_cfg.get("include_bathy_scalar", False))
    standardize = bool(io_cfg.get("standardize_inputs", True))
    groups = io_cfg.get(
        "groups",
        {
            "sss": {"h5_group": "sss", "h5_var": "sos"},
            "sst": {"h5_group": "ostia", "h5_var": "analysed_sst"},
            "ssh": {"h5_group": "ssh", "h5_var": "adt"},
        },
    )

    sat_path = os.path.join(io_cfg["data_path"], io_cfg["sat_file"])
    if not os.path.isfile(sat_path):
        raise FileNotFoundError(
            f"Missing satellite HDF5 {sat_path}; run utils/generate_argo_satellite_data.py first"
        )

    ds = _load_v2_dataset(io_cfg)
    n = len(ds)
    max_samples = max_samples if max_samples is not None else io_cfg.get("max_samples")
    if max_samples is not None:
        n = min(n, int(max_samples))

    juld = np.asarray(ds.TIME[:n], dtype=np.float32)
    lat = np.asarray(ds.LAT[:n], dtype=np.float32)
    lon = np.asarray(ds.LON[:n], dtype=np.float32)

    sat_vars: dict[str, np.ndarray] = {}
    bottom_depth = None
    with h5py.File(sat_path, "r") as sf:
        n_h5 = sf["stations"]["latitude"].shape[0]
        if n_h5 < n:
            raise ValueError(f"satellite HDF5 has {n_h5} stations, need {n}")
        indices = np.arange(n, dtype=np.int64)
        for key, spec in groups.items():
            if not input_params.get(key):
                continue
            d = sf[spec["h5_group"]][spec["h5_var"]][indices]
            sat_vars[key] = extract_sat_values(d, spatial_pad, temporal_pad)
        if "bathymetry" in sf and "elevation" in sf["bathymetry"]:
            bathy = sf["bathymetry"]["elevation"][indices]
            bottom_depth = _bottom_depth_from_bathy(bathy, spatial_pad)

    if bottom_depth is None:
        bottom_depth = np.full(n, np.nan, dtype=np.float32)

    use_mask_channels = resolve_use_mask_channels(config)
    inputs, layout = build_argo_residual_input_matrix(
        juld,
        lat,
        lon,
        sat_vars,
        input_params,
        spatial_pad=spatial_pad,
        temporal_pad=temporal_pad,
        bathy_depth=bottom_depth,
        center_relative=center_relative,
        include_bathy_scalar=include_bathy_scalar,
    )
    expected = compute_argo_residual_input_dim(
        input_params,
        spatial_pad,
        temporal_pad,
        use_mask_channels=use_mask_channels,
        include_bathy_scalar=include_bathy_scalar,
    )
    if inputs.shape[1] != expected:
        raise ValueError(f"input dim {inputs.shape[1]} != expected {expected}")

    feature_names = build_residual_feature_names(
        input_params,
        spatial_pad,
        temporal_pad,
        include_bathy_scalar=include_bathy_scalar,
        center_relative=center_relative,
    )
    input_standardization = None
    if standardize:
        inputs, input_standardization = apply_train_standardization(
            inputs,
            juld,
            config,
            feature_names,
            dataset_tag=io_cfg.get("dataset_tag", "argo_residual"),
            v2_src=io_cfg.get("v2_src"),
        )

    refit_pca = bool(io_cfg.get("refit_pca", True))
    use_anom = bool(io_cfg.get("anomaly_targets"))
    target_cols = []
    pcs_by_name = {}
    pca_models = {}
    profiles = {}
    for name, n_comp in outputs.items():
        if name == "temperature":
            pca, pcs = ds.pca_temp, ds.temp_pcs
            prof = np.asarray(ds.TEMP, dtype=np.float32)
        elif name == "salinity":
            pca, pcs = ds.pca_sal, ds.sal_pcs
            prof = np.asarray(ds.SAL, dtype=np.float32)
        else:
            raise KeyError(f"unknown output {name}")
        if prof.ndim == 2:
            prof = prof[:, :n]
        else:
            prof = prof[..., :n]
        if not use_anom:
            if n_comp != pca.n_components_:
                if not refit_pca:
                    raise ValueError(f"{name}: need refit_pca for {n_comp} components")
                pca, pcs = _refit_pca_from_profiles(prof, n_comp)
            else:
                pcs = pcs[:, :n]
            pca_models[name] = pca
            pcs_by_name[name] = pcs
            target_cols.append(pcs.T.astype(np.float32))
        profiles[name] = np.ascontiguousarray(prof[:, :n].astype(np.float32) if prof.shape[0] != n else prof.astype(np.float32))

    if use_anom:
        targets = None
        weights = None
    else:
        targets = np.hstack(target_cols).astype(np.float32)
        weights = get_pca_weights(pca_models, pcs_by_name, list(outputs.keys()))
    pres = np.arange(int(ds.min_depth), int(ds.max_depth) + 1, dtype=np.float32)
    patch_shape = residual_sat_patch_shape(
        input_params,
        spatial_pad,
        temporal_pad,
        use_mask_channels=use_mask_channels,
    )

    profiles_dm = {}
    for name in outputs:
        p = np.asarray(profiles[name], dtype=np.float32)
        profiles_dm[name] = np.ascontiguousarray(p if p.shape[1] == n else p.T)

    anom = None
    if io_cfg.get("anomaly_targets"):
        from preproc.climatology import build_anomaly_targets_block

        anom = build_anomaly_targets_block(
            profiles_dm,
            lat,
            lon,
            juld,
            pres,
            outputs,
            config,
        )
        targets = anom["targets"]
        pca_models = anom["pca_models"]
        pcs_by_name = anom["pcs_by_name"]
        weights = anom["weights"]

    payload: dict[str, Any] = {
        "inputs": inputs,
        "targets": targets,
        "pca_models": pca_models,
        "pcs_by_name": pcs_by_name,
        "outputs": outputs,
        "weights": weights,
        "profiles": profiles_dm,
        "true_profiles": true_profiles_numpy(targets, pca_models, outputs),
        "LAT": lat,
        "LON": lon,
        "PRES": pres,
        "JULD": juld,
        "bottom_depth": bottom_depth,
        "station_indices": np.arange(n, dtype=np.int64),
        "input_params": input_params,
        "spatial_pad": spatial_pad,
        "temporal_pad": temporal_pad,
        "sat_patch_shape": patch_shape,
        "patch_order": "time_major_row_major_lat_lon",
        "config_hash": chash,
        "dataset_tag": io_cfg.get("dataset_tag", "argo_residual"),
        "min_depth": int(ds.min_depth),
        "max_depth": int(ds.max_depth),
        "bathy_truncation": True,
        "use_mask_channels": use_mask_channels,
        "base_dim": layout["base_dim"],
        "patch_offset": layout["patch_offset"],
        "center_relative_patches": center_relative,
        "feature_names": feature_names,
    }
    if input_standardization is not None:
        payload["input_standardization"] = input_standardization
    if anom is not None:
        payload.update(
            {
                k: anom[k]
                for k in (
                    "climatology",
                    "clim_profiles",
                    "anomaly_targets",
                    "ssh_obs_adt",
                    "ssh_obs_sla",
                    "clim_steric",
                    "steric_calibration",
                )
                if k in anom
            }
        )
    return write_train_cache(payload, cache_dir, chash)


def main(argv: list[str] | None = None) -> int:
    from parse_config import validate_config
    from base.util import read_json

    parser = argparse.ArgumentParser(description="Export ARGO residual patch train cache")
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args(argv)
    cfg = read_json(args.config)
    validate_config(cfg)
    if cfg.get("io", {}).get("dataset_tag") != "argo_residual":
        raise ValueError("io.dataset_tag must be 'argo_residual'")
    path = build_argo_residual_cache(cfg, force=args.force, max_samples=args.max_samples)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
