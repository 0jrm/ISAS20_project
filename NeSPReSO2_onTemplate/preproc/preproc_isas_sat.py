import os
import sys
import hashlib
import json
import pickle
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import h5py
import numpy as np
from sklearn.decomposition import PCA

from model.loss import OUTPUT_H5_VARS, get_pca_weights

# ponytail: patch flatten order = time (oldest..newest within window), then row-major (lat, lon) per slice
PATCH_ORDER_DOC = "time_major_row_major_lat_lon"


def compute_input_dim(input_params: Mapping[str, bool]) -> int:
    """Match v2: ``sum(flags) - 1`` when ``sat`` groups sss/sst/ssh."""
    return sum(v for v in input_params.values() if v) - (1 if input_params.get("sat") else 0)


def config_hash(config: Dict) -> str:
    """Stable hash for cache filenames (includes ``io.dataset_tag``)."""
    payload = {
        "input_params": config.get("input_params"),
        "io": config.get("io"),
        "outputs": config.get("outputs"),
    }
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def write_train_cache(payload: Dict, cache_dir: str, chash: str) -> str:
    """Pickle train-ready cache; returns path."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"train_ready_{chash}.pkl")
    with open(cache_path, "wb") as f:
        pickle.dump(payload, f)
    n = payload["inputs"].shape[0]
    d = payload["inputs"].shape[1]
    tag = payload.get("dataset_tag", "?")
    print(f"Train-ready cache saved to {cache_path} (tag={tag}, N={n}, D={d})")
    return cache_path


def _center_indices(n: int) -> int:
    return n // 2


def extract_sat_values(
    arr: np.ndarray,
    spatial_pad: int,
    temporal_pad: int,
) -> np.ndarray:
    """
    Extract scalar (offset=0) or flattened patch features per station.

    ``arr`` shape: (N,), (N, H, W), (N, T), or (N, T, H, W).
    """
    if arr.ndim == 1:
        return arr.astype(np.float32)

    if arr.ndim == 2:
        if spatial_pad == 0:
            return arr[:, _center_indices(arr.shape[1])].astype(np.float32)
        return arr.astype(np.float32).reshape(arr.shape[0], -1)

    if arr.ndim == 3:
        cy, cx = _center_indices(arr.shape[1]), _center_indices(arr.shape[2])
        if spatial_pad == 0:
            return arr[:, cy, cx].astype(np.float32)
        patch = arr[:, cy - spatial_pad : cy + spatial_pad + 1, cx - spatial_pad : cx + spatial_pad + 1]
        return patch.reshape(arr.shape[0], -1).astype(np.float32)

    # (N, T, H, W)
    t_win = temporal_pad + 1
    t_slice = arr[:, -t_win:] if temporal_pad > 0 else arr[:, -1:]
    cy, cx = _center_indices(arr.shape[2]), _center_indices(arr.shape[3])
    if spatial_pad == 0:
        vals = t_slice[:, :, cy, cx]
        if temporal_pad == 0:
            return vals[:, -1].astype(np.float32)
        return vals.astype(np.float32).reshape(arr.shape[0], -1)

    patches = t_slice[
        :,
        :,
        cy - spatial_pad : cy + spatial_pad + 1,
        cx - spatial_pad : cx + spatial_pad + 1,
    ]
    return patches.reshape(arr.shape[0], -1).astype(np.float32)


def build_input_matrix(
    juld: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    sat_vars: Dict[str, np.ndarray],
    input_params: Mapping[str, bool],
) -> np.ndarray:
    """Vectorized v2 ``prepare_inputs`` (offset already applied in ``sat_vars``)."""
    cols = []
    day = juld % 365

    if input_params.get("timecos"):
        cols.append(np.cos(2 * np.pi * day / 365))
    if input_params.get("timesin"):
        cols.append(np.sin(2 * np.pi * day / 365))
    if input_params.get("latcos"):
        cols.append(np.cos(2 * np.pi * lat / 180))
    if input_params.get("latsin"):
        cols.append(np.sin(2 * np.pi * lat / 180))
    if input_params.get("loncos"):
        cols.append(np.cos(2 * np.pi * lon / 360))
    if input_params.get("lonsin"):
        cols.append(np.sin(2 * np.pi * lon / 360))

    for key in ("sss", "sst", "ssh"):
        if not input_params.get(key):
            continue
        if key not in sat_vars:
            raise KeyError(f"Missing satellite variable '{key}' in groups/io config")
        val = sat_vars[key]
        if key == "sst":
            val = val - 273.15
        if val.ndim == 1:
            cols.append(val)
        else:
            for j in range(val.shape[1]):
                cols.append(val[:, j])

    if not cols:
        raise ValueError("No input features selected in input_params")

    stacked = np.column_stack(cols).astype(np.float32)
    return np.nan_to_num(stacked, nan=0.0)


def _resolve_h5_paths(io_cfg: Dict) -> tuple[str, str]:
    data_path = io_cfg["data_path"]
    sat_file = io_cfg.get("sat_file", "satellite_*.h5")
    prof_file = io_cfg.get("prof_file", "profiles_*.h5")
    if "*" in sat_file:
        import glob

        matches = sorted(glob.glob(os.path.join(data_path, sat_file)))
        if not matches:
            raise FileNotFoundError(f"No satellite file matching {sat_file} in {data_path}")
        sat_path = matches[0]
    else:
        sat_path = os.path.join(data_path, sat_file)
    prof_path = os.path.join(data_path, prof_file)
    return sat_path, prof_path


def build_train_cache(config: Dict, force: bool = False) -> str:
    """
    Read HDF5 once, build inputs + per-output PCA, pickle train-ready cache.

    Returns path to the pickle file.
    """
    io_cfg = config["io"]
    spatial_pad = int(io_cfg.get("spatial_pad", 0))
    temporal_pad = int(io_cfg.get("temporal_pad", 0))
    outputs = OrderedDict(config["outputs"])
    input_params = config["input_params"]
    groups = io_cfg.get("groups", {
        "sss": {"h5_group": "sss", "h5_var": "sos"},
        "sst": {"h5_group": "ostia", "h5_var": "analysed_sst"},
        "ssh": {"h5_group": "ssh", "h5_var": "adt"},
    })

    cache_dir = io_cfg.get("cache_dir", "data/cache")
    os.makedirs(cache_dir, exist_ok=True)
    chash = config_hash(config)
    cache_path = os.path.join(cache_dir, f"train_ready_{chash}.pkl")
    if os.path.exists(cache_path) and not force:
        return cache_path

    sat_path, prof_path = _resolve_h5_paths(io_cfg)  # trust boundary: paths from config only
    bbox = io_cfg.get("BBox")

    with h5py.File(prof_path, "r") as pf:
        model = pf["model"]
        lat = model["LATITUDE"][:]
        lon = model["LONGITUDE"][:]
        juld = model["JULD"][:]
        if bbox is not None:
            min_lat, max_lat, min_lon, max_lon = bbox
            mask = (lat >= min_lat) & (lat <= max_lat) & (lon >= min_lon) & (lon <= max_lon)
            station_indices = np.where(mask)[0]
        else:
            station_indices = np.arange(len(lat))

        lat = lat[station_indices]
        lon = lon[station_indices]
        juld = juld[station_indices]

        profile_data = {}
        for name in outputs:
            h5_var = OUTPUT_H5_VARS.get(name, name.upper())
            profile_data[name] = model[h5_var][station_indices].astype(np.float64)

        pres = model.get("DEPH")
        if pres is not None:
            pres = pres[0].astype(np.float32)
        else:
            pres = np.arange(profile_data[list(outputs.keys())[0]].shape[1], dtype=np.float32)

    sat_vars = {}
    with h5py.File(sat_path, "r") as sf:
        for key, spec in groups.items():
            if not input_params.get(key):
                continue
            d = sf[spec["h5_group"]][spec["h5_var"]][station_indices]
            sat_vars[key] = extract_sat_values(d, spatial_pad, temporal_pad)

    inputs = build_input_matrix(juld, lat, lon, sat_vars, input_params)

    pca_models = {}
    pcs_by_name = {}
    target_cols = []
    for name, n_comp in outputs.items():
        profiles = profile_data[name]
        profiles = np.nan_to_num(profiles, nan=0.0)
        pca = PCA(n_components=n_comp)
        pcs = pca.fit_transform(profiles)
        pca_models[name] = pca
        pcs_by_name[name] = pcs.T
        target_cols.append(pcs.astype(np.float32))

    targets = np.hstack(target_cols).astype(np.float32)
    weights = get_pca_weights(pca_models, pcs_by_name, list(outputs.keys()))
    # ponytail: depth-major (n_z, N) matches v2 ``TEMP``/``SAL`` layout for eval scripts.
    # Keep NaN for missing/deep levels so eval RMSE can mask them; PCA fit above uses its
    # own nan_to_num. NEVER nan_to_num here — 0.0 fill contaminates profile-space RMSE.
    profiles = {
        name: np.ascontiguousarray(profile_data[name].T.astype(np.float32))
        for name in outputs
    }
    dataset_tag = io_cfg.get("dataset_tag", "isas20")

    payload = {
        "inputs": inputs,
        "targets": targets,
        "pca_models": pca_models,
        "pcs_by_name": pcs_by_name,
        "outputs": outputs,
        "weights": weights,
        "profiles": profiles,
        "LAT": lat.astype(np.float32),
        "LON": lon.astype(np.float32),
        "PRES": pres,
        "JULD": juld.astype(np.float32),
        "station_indices": station_indices,
        "input_params": dict(input_params),
        "patch_order": PATCH_ORDER_DOC,
        "config_hash": chash,
        "dataset_tag": dataset_tag,
        "min_depth": 0,
        "max_depth": int(profile_data[list(outputs.keys())[0]].shape[1] - 1),
    }

    return write_train_cache(payload, cache_dir, chash)


# --- legacy helpers (unchanged API) ---

def load_config(config_path: Optional[str] = None) -> Dict:
    if config_path is None:
        return {
            "data_path": "/unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project/data/NeSPReSO_v2_GoM_sat",
            "BBox": None,
            "input_params": {
                "bathymetry": ["elevation"],
                "ssh": ["adt", "sla", "ugos", "vgos"],
                "ostia": ["analysed_sst"],
                "sss": ["sos"],
                "wind": ["windspeed", "u_wind", "v_wind"],
            },
            "input_processing": "norm",
            "spatial_step": 1,
            "spatial_pad": 2,
            "temporal_pad": 3,
            "output_params": {"temperature": ["TEMP"], "salinity": ["PSAL"]},
            "output_processing": "norm",
        }

    with open(config_path, "r") as f:
        return json.load(f)


def normalize_ignore_nan(arr, axis=None):
    if axis is not None:
        n_valid = np.sum(~np.isnan(arr), axis=axis, keepdims=True)
    else:
        n_valid = np.sum(~np.isnan(arr))

    mean = np.nanmean(arr, axis=axis, keepdims=True)
    mean = np.where(n_valid > 0, mean, 0)
    std = np.nanstd(arr, axis=axis, keepdims=True)
    std = np.where(n_valid > 1, std, 1)
    normed = (arr - mean) / std
    return np.where(np.isnan(normed), 0, normed), mean, std


def get_bbox_mask(lat, lon, bbox):
    if bbox is None:
        return np.ones_like(lat, dtype=bool)
    min_lat, max_lat, min_lon, max_lon = bbox
    return (lat >= min_lat) & (lat <= max_lat) & (lon >= min_lon) & (lon <= max_lon)


def reshape_time(data, temporal_pad):
    N, T = data.shape[:2]
    return data[:, : temporal_pad + 1, ...]


def preprocess_data(config: Dict) -> Dict:
    """Legacy multi-dict preprocess (kept for playground notebooks)."""
    data_path = config["data_path"]
    sat_file = os.path.join(data_path, "satellite_NeSPReSO_v1_global.h5")
    prof_file = os.path.join(data_path, "profiles_NeSPReSO_v1_global.h5")

    with h5py.File(prof_file, "r") as pf:
        station_lat = pf["model"]["LATITUDE"][:]
        station_lon = pf["model"]["LONGITUDE"][:]
        bbox_mask = get_bbox_mask(station_lat, station_lon, config["BBox"])
        station_indices = np.where(bbox_mask)[0]

    processed = {"inputs": {}, "outputs": {}, "masks": {}, "station_indices": station_indices, "stats": {}}

    with h5py.File(sat_file, "r") as sf:
        for group, vars in config["input_params"].items():
            g = sf[group]
            for var in vars:
                d = g[var][:]
                d = d[station_indices]
                mask = np.isnan(d)
                if config["input_processing"] == "norm":
                    d, input_mean, input_std = normalize_ignore_nan(d)
                    processed["stats"][f"{group}_{var}_mean"] = input_mean
                    processed["stats"][f"{group}_{var}_std"] = input_std
                if d.ndim >= 3 and d.shape[1] >= config["temporal_pad"]:
                    d = reshape_time(d, config["temporal_pad"])
                    mask = reshape_time(mask, config["temporal_pad"])
                processed["inputs"][f"{group}_{var}"] = d
                processed["masks"][f"{group}_{var}"] = mask

    with h5py.File(prof_file, "r") as pf:
        for group, vars in config["output_params"].items():
            g = pf["model"]
            for var in vars:
                d = g[var][:]
                d = d[station_indices]
                mask = np.isnan(d)
                if config["output_processing"] == "norm":
                    d, output_mean, output_std = normalize_ignore_nan(d, axis=0)
                    processed["stats"][f"{group}_{var}_mean"] = output_mean
                    processed["stats"][f"{group}_{var}_std"] = output_std
                processed["outputs"][f"{group}_{var}"] = d
                processed["masks"][f"{group}_{var}"] = mask

    return processed


def save_processed_data(processed: Dict, save_path: str) -> None:
    import torch

    tensor_processed = {}
    for key, value in processed.items():
        if key == "station_indices":
            tensor_processed[key] = torch.tensor(value)
        elif isinstance(value, dict):
            tensor_processed[key] = {
                k: torch.tensor(v) if isinstance(v, np.ndarray) else v for k, v in value.items()
            }
        elif isinstance(value, np.ndarray):
            tensor_processed[key] = torch.tensor(value)
        else:
            tensor_processed[key] = value

    with open(save_path, "wb") as f:
        pickle.dump(tensor_processed, f)
    print(f"Preprocessed data saved to {save_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "cache":
        from parse_config import read_json, validate_config

        cfg_path = sys.argv[2] if len(sys.argv) > 2 else "config.json"
        cfg = read_json(cfg_path)
        validate_config(cfg)
        build_train_cache(cfg, force="--force" in sys.argv)
    else:
        config_path = sys.argv[1] if len(sys.argv) > 1 else None
        config = load_config(config_path)
        save_path = os.path.join(config["data_path"], "preprocessed_satellite_data.pkl")
        save_processed_data(preprocess_data(config), save_path)
