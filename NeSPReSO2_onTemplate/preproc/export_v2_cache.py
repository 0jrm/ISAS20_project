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

from model.density_spice import make_control_grid, normalized_dz
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


def _station_major(prof: np.ndarray, n: int) -> np.ndarray:
    """Return (n, n_z) regardless of depth-major vs station-major storage."""
    arr = np.asarray(prof, dtype=np.float32)
    if arr.shape[0] == n:
        return arr
    if arr.shape[1] == n:
        return arr.T
    raise ValueError(f"cannot align profile shape {arr.shape} to n={n}")


def _build_density_spice_targets(
    profiles: dict[str, np.ndarray],
    lat: np.ndarray,
    lon: np.ndarray,
    depth: np.ndarray,
    outputs: OrderedDict,
    train_idx: np.ndarray,
) -> dict[str, Any]:
    """Phase 3.1–3.3: σ₀ control-grid + spice PCA targets (train-fit stats only)."""
    from evalphys.inversion import sigma0_spice_from_ts
    from evalphys.gsw_backend import get_gsw

    if set(outputs) != {"density_ctrl", "spice"}:
        raise ValueError(f"density_spice outputs must be density_ctrl+spice, got {dict(outputs)}")
    k = int(outputs["density_ctrl"])
    n_spice = int(outputs["spice"])
    T = _station_major(profiles["temperature"], lat.shape[0])
    S = _station_major(profiles["salinity"], lat.shape[0])
    n, n_z = T.shape
    gsw = get_gsw()
    p = gsw.p_from_z(-np.broadcast_to(depth, (n, n_z)), lat[:, None])
    sig, tau = sigma0_spice_from_ts(T, S, p, lon[:, None], lat[:, None])

    z_ctrl = make_control_grid(depth, K=k)
    dz_tilde = normalized_dz(z_ctrl)
    sig_ctrl = np.empty((n, k), dtype=np.float64)
    for i in range(n):
        ok = np.isfinite(sig[i]) & np.isfinite(depth)
        if ok.sum() < 2:
            sig_ctrl[i] = np.nan
            continue
        sig_ctrl[i] = np.interp(z_ctrl, depth[ok], sig[i, ok])

    tr = np.asarray(train_idx, dtype=int)
    mu_s = np.nanmean(sig_ctrl[tr], axis=0)
    sd_s = np.maximum(np.nanstd(sig_ctrl[tr], axis=0), 1e-6)
    sig_ctrl_z = ((sig_ctrl - mu_s) / sd_s).astype(np.float32)

    mu_t = np.nanmean(tau[tr], axis=0)
    sd_t = np.maximum(np.nanstd(tau[tr], axis=0), 1e-6)
    tau_z = ((tau - mu_t) / sd_t).astype(np.float64)
    tau_z = np.nan_to_num(tau_z, nan=0.0)
    pca_spice = PCA(n_components=n_spice).fit(tau_z[tr])
    spice_pcs = pca_spice.transform(tau_z).astype(np.float32)  # (n, n_spice)

    targets = np.hstack([sig_ctrl_z, spice_pcs]).astype(np.float32)
    pcs_by_name = {
        "density_ctrl": sig_ctrl_z.T,  # (K, n) for weight helper shape compat
        "spice": spice_pcs.T,
    }
    # Weights: uniform on density ctrl; spice explained variance via get_pca_weights
    w_rho = np.ones(k, dtype=np.float32)
    w_spice = get_pca_weights({"spice": pca_spice}, {"spice": spice_pcs.T}, ["spice"])
    weights = np.concatenate([w_rho, w_spice]).astype(np.float32)

    return {
        "targets": targets,
        "pca_models": {"spice": pca_spice},
        "pcs_by_name": pcs_by_name,
        "weights": weights,
        "targets_sigma0": sig.astype(np.float32),
        "targets_spice": tau.astype(np.float32),
        "density_spice_meta": {
            "z_ctrl": z_ctrl.astype(np.float32),
            "dz_tilde": dz_tilde.astype(np.float32),
            "sigma0_ctrl_mean": mu_s.astype(np.float32),
            "sigma0_ctrl_std": sd_s.astype(np.float32),
            "spice_mean": mu_t.astype(np.float32),
            "spice_std": sd_t.astype(np.float32),
            "K": k,
            "n_spice": n_spice,
        },
    }


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
    representation = io_cfg.get("representation", "ts_pca")
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
    lat = ds.LAT.astype(np.float32)[:n]
    lon = ds.LON.astype(np.float32)[:n]
    juld = ds.TIME.astype(np.float32)[:n]

    # Always keep physical T/S for eval / density_spice construction
    raw_profiles = {
        "temperature": np.asarray(ds.TEMP, dtype=np.float32),
        "salinity": np.asarray(ds.SAL, dtype=np.float32),
    }
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

    profiles_ts = _profiles_depth_major(raw_profiles, n, pres)
    profiles_ts = {name: np.ascontiguousarray(arr) for name, arr in profiles_ts.items()}

    if representation == "density_spice":
        from base.split_utils import build_split_indices

        dl_cfg = dict((config.get("data_loader") or {}).get("args") or {})
        splits = build_split_indices(
            n, juld, dl_cfg, dataset_tag=dataset_tag, v2_src=v2_src
        )
        block = _build_density_spice_targets(
            profiles_ts, lat, lon, pres, outputs, splits["train"]
        )
        targets = block["targets"]
        pca_models = block["pca_models"]
        pcs_by_name = block["pcs_by_name"]
        weights = block["weights"]
        profiles = profiles_ts
        true_profiles = {
            "temperature": _station_major(profiles_ts["temperature"], n),
            "salinity": _station_major(profiles_ts["salinity"], n),
        }
        payload_extra = {
            "targets_sigma0": block["targets_sigma0"],
            "targets_spice": block["targets_spice"],
            "density_spice_meta": block["density_spice_meta"],
            "representation": "density_spice",
        }
    else:
        target_cols = []
        pcs_by_name = {}
        pca_models = {}
        refit_pca = bool(io_cfg.get("refit_pca", True))
        for name, n_comp in outputs.items():
            if name not in ("temperature", "salinity"):
                raise KeyError(f"ponytail: v2 export only knows temperature/salinity, got {name}")
            prof = profiles_ts[name]
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
        profiles = profiles_ts
        true_profiles = true_profiles_numpy(targets, pca_models, outputs)
        payload_extra = {}

        if io_cfg.get("anomaly_targets"):
            from preproc.climatology import build_anomaly_targets_block

            anom = build_anomaly_targets_block(
                profiles,
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
            payload_extra.update(
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

    payload = {
        "inputs": inputs,
        "targets": targets,
        "pca_models": pca_models,
        "pcs_by_name": pcs_by_name,
        "outputs": outputs,
        "weights": weights,
        "profiles": profiles,
        "true_profiles": true_profiles,
        "LAT": lat,
        "LON": lon,
        "PRES": pres,
        "JULD": juld,
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
    payload.update(payload_extra)
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
