#!/usr/bin/env python3
"""Export v2 ARGO+COAPS dataset pickle into the template train-ready cache schema."""

from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
from sklearn.decomposition import PCA

from model.loss import get_pca_weights, true_profiles_numpy
from preproc.climatology import _profiles_depth_major
from preproc.preproc_isas_sat import config_hash, write_train_cache


def _refit_pca_from_profiles(prof: np.ndarray, n_comp: int) -> tuple[PCA, np.ndarray]:
    """Fit PCA on depth profiles; returns model and pcs (n_comp, n_stations)."""
    arr = np.asarray(prof, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"profile array must be 2-D, got shape {arr.shape}")
    # depth-major (n_z, N) from v2 caches
    fit_x = np.nan_to_num(arr.T, nan=0.0)
    pca = PCA(n_components=n_comp).fit(fit_x)
    pcs = pca.transform(fit_x).astype(np.float32).T
    return pca, pcs


def build_argo_cache(config: Dict, force: bool = False) -> str:
    """Load v2 ``config_dataset_full.pkl`` and write ``train_ready_<hash>.pkl``."""
    io_cfg = config["io"]
    cache_dir = io_cfg.get("cache_dir", "data/cache")
    chash = config_hash(config)
    cache_path = Path(cache_dir) / f"train_ready_{chash}.pkl"
    if cache_path.exists() and not force:
        return str(cache_path)

    pickle_path = io_cfg.get("v2_pickle")
    if not pickle_path or not Path(pickle_path).is_file():
        raise FileNotFoundError(f"io.v2_pickle missing or not found: {pickle_path}")

    v2_src = io_cfg.get("v2_src")
    if v2_src:
        sys.path.insert(0, str(v2_src))
    from nespreso.data.pickle_compat import load_dataset_pickle

    outputs = OrderedDict(config["outputs"])
    input_params = dict(config["input_params"])
    data = load_dataset_pickle(pickle_path)
    ds = data["full_dataset"]
    if not hasattr(ds, "n_components"):
        ds.n_components = list(outputs.values())[0]
    ds.input_params = input_params
    if io_cfg.get("v2_reload", False):
        ds.reload()

    n = len(ds)
    max_samples = io_cfg.get("max_samples")
    if max_samples is not None:
        n = min(n, int(max_samples))
    rows = []
    for i in range(n):
        x, _ = ds[i]
        rows.append(x.numpy() if torch.is_tensor(x) else np.asarray(x, dtype=np.float32))
    inputs = np.stack(rows).astype(np.float32)

    pres = np.arange(ds.min_depth, ds.max_depth + 1, dtype=np.float32)
    dataset_tag = io_cfg.get("dataset_tag", "argo_v2")

    raw_profiles = {}
    for name in outputs:
        if name == "temperature":
            raw_profiles[name] = np.asarray(ds.TEMP, dtype=np.float32)
        elif name == "salinity":
            raw_profiles[name] = np.asarray(ds.SAL, dtype=np.float32)
        else:
            raise KeyError(f"ponytail: v2 export only knows temperature/salinity, got {name}")

    full_n = len(ds)
    if max_samples is not None:
        sliced_raw = {}
        for name, arr in raw_profiles.items():
            if arr.shape[0] == full_n:
                sliced_raw[name] = arr[:n]
            elif arr.shape[1] == full_n:
                sliced_raw[name] = arr[:, :n]
            else:
                sliced_raw[name] = arr
        raw_profiles = sliced_raw

    profiles = _profiles_depth_major(raw_profiles, n, pres)
    profiles = {name: np.ascontiguousarray(arr) for name, arr in profiles.items()}

    target_cols = []
    pcs_by_name = {}
    pca_models = {}
    refit_pca = bool(io_cfg.get("refit_pca", True))
    for name, n_comp in outputs.items():
        prof = profiles[name]
        legacy_pca = ds.pca_temp if name == "temperature" else ds.pca_sal
        if refit_pca or n_comp != legacy_pca.n_components_:
            pca, pcs = _refit_pca_from_profiles(prof, n_comp)
        else:
            pca = legacy_pca
            pcs = ds.temp_pcs if name == "temperature" else ds.sal_pcs
            if pcs.shape[1] > n:
                pcs = pcs[:, :n]
        pca_models[name] = pca
        pcs_by_name[name] = pcs
        target_cols.append(pcs.T.astype(np.float32))

    targets = np.hstack(target_cols).astype(np.float32)
    weights = get_pca_weights(pca_models, pcs_by_name, list(outputs.keys()))

    if io_cfg.get("anomaly_targets"):
        from preproc.climatology import build_anomaly_targets_block

        anom = build_anomaly_targets_block(
            profiles,
            ds.LAT.astype(np.float32)[:n],
            ds.LON.astype(np.float32)[:n],
            ds.TIME.astype(np.float32)[:n],
            pres,
            outputs,
            config,
        )
        targets = anom["targets"]
        pca_models = anom["pca_models"]
        pcs_by_name = anom["pcs_by_name"]
        weights = anom["weights"]

    payload = {
        "inputs": inputs,
        "targets": targets,
        "pca_models": pca_models,
        "pcs_by_name": pcs_by_name,
        "outputs": outputs,
        "weights": weights,
        "profiles": profiles,
        "true_profiles": true_profiles_numpy(targets, pca_models, outputs),
        "LAT": ds.LAT.astype(np.float32)[:n],
        "LON": ds.LON.astype(np.float32)[:n],
        "PRES": pres,
        "JULD": ds.TIME.astype(np.float32)[:n],
        "station_indices": np.arange(n, dtype=np.int64),
        "input_params": input_params,
        "spatial_pad": int(io_cfg.get("spatial_pad", 0)),
        "temporal_pad": int(io_cfg.get("temporal_pad", 0)),
        "sat_patch_shape": None,
        "config_hash": chash,
        "dataset_tag": dataset_tag,
        "min_depth": int(ds.min_depth),
        "max_depth": int(ds.max_depth),
    }
    if io_cfg.get("anomaly_targets"):
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
    from base.util import read_json
    from parse_config import validate_config

    parser = argparse.ArgumentParser(description="Export v2 dataset pickle to train-ready cache")
    parser.add_argument("-c", "--config", required=True, help="config JSON (e.g. config/argo/config_argo.json)")
    parser.add_argument("--force", action="store_true", help="rebuild even if cache exists")
    args = parser.parse_args(argv)
    cfg = read_json(args.config)
    validate_config(cfg)
    if cfg.get("io", {}).get("dataset_tag", "isas20") != "argo_v2":
        raise ValueError("config io.dataset_tag must be 'argo_v2' for this exporter")
    path = build_argo_cache(cfg, force=args.force)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
