"""Build field-to-field train cache (Phase C)."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import xarray as xr

_ROOT = Path(__file__).resolve().parents[1]
_UTILS = _ROOT.parent / "utils"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_UTILS) not in sys.path:
    sys.path.insert(0, str(_UTILS))

from retrieve_sat import ROOT_DIRS, get_coord_names, get_variable_name, select_candidate_file  # noqa: E402

from base.split_utils import sample_dates  # noqa: E402
from preproc.preproc_isas_sat import config_hash, write_train_cache  # noqa: E402


def _gom_grid(basin: dict, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    lats = np.linspace(basin["min_lat"], basin["max_lat"], shape[0], dtype=np.float32)
    lons = np.linspace(basin["min_lon"], basin["max_lon"], shape[1], dtype=np.float32)
    return lats, lons


def _latlon_to_rc(lat: float, lon: float, lats: np.ndarray, lons: np.ndarray) -> tuple[int, int]:
    r = int(np.argmin(np.abs(lats - lat)))
    c = int(np.argmin(np.abs(lons - lon)))
    return r, c


def _resolve_source_cache_path(io_cfg: dict) -> Path:
    field_cfg = io_cfg.get("field") or {}
    src = field_cfg.get("source_cache")
    if src:
        path = Path(src)
        if not path.is_absolute():
            path = (_ROOT / path).resolve()
        if path.is_file():
            return path
        raise FileNotFoundError(f"source anomaly cache not found: {path}")

    source_config = field_cfg.get("source_config")
    if source_config:
        from base.util import read_json

        cfg_path = Path(source_config)
        if not cfg_path.is_absolute():
            cfg_path = (_ROOT / cfg_path).resolve()
        src_cfg = read_json(cfg_path)
        tag = src_cfg.get("io", {}).get("dataset_tag", "argo_v2")
        if tag == "argo_l4":
            from preproc.export_argo_l4_cache import build_argo_l4_cache

            built = build_argo_l4_cache(src_cfg)
        else:
            from preproc.export_v2_cache import build_argo_cache

            built = build_argo_cache(src_cfg)
        return Path(built)

    raise ValueError("io.field.source_cache or io.field.source_config required for argo_field cache")


def _load_source_cache(io_cfg: dict) -> dict:
    path = _resolve_source_cache_path(io_cfg)
    with open(path, "rb") as f:
        return pickle.load(f)


def _sla_grid_for_date(dt, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    fp = select_candidate_file("ssh", dt)
    grid = np.full((len(lats), len(lons)), np.nan, dtype=np.float32)
    if fp is None:
        return grid
    with xr.open_dataset(fp, decode_times=True) as ds:
        lat_name, lon_name, time_name = get_coord_names(ds)
        var = get_variable_name("ssh", "sla", ds)
        if var is None:
            return grid
        da = ds[var]
        if time_name and time_name in da.dims:
            da = da.sel({time_name: dt}, method="nearest")
        interp = da.interp({lat_name: lats, lon_name: lons}, method="linear")
        grid = np.asarray(interp.values, dtype=np.float32)
    return grid


def _static_bathy_mask(lats: np.ndarray, lons: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fp = select_candidate_file("bathymetry", None)
    log_bathy = np.zeros((len(lats), len(lons)), dtype=np.float32)
    land_mask = np.zeros((len(lats), len(lons)), dtype=np.float32)
    if fp is None:
        land_mask[:] = 1.0
        return log_bathy, land_mask
    with xr.open_dataset(fp, decode_times=True) as ds:
        lat_name, lon_name, _ = get_coord_names(ds)
        var = get_variable_name("bathymetry", "elevation", ds)
        elev = ds[var].interp({lat_name: lats, lon_name: lons}, method="nearest")
        arr = np.asarray(elev.values, dtype=np.float32)
        ocean = arr < 0
        land_mask = ocean.astype(np.float32)
        depth = np.maximum(0.0, -arr)
        log_bathy = np.log1p(depth).astype(np.float32)
        log_bathy[~ocean] = 0.0
    return log_bathy, land_mask


def build_field_cache(config: Dict, force: bool = False) -> str:
    io_cfg = config["io"]
    cache_dir = io_cfg.get("cache_dir", "../data/cache")
    chash = config_hash(config)
    cache_path = Path(cache_dir) / f"train_ready_{chash}.pkl"
    if cache_path.exists() and not force:
        return str(cache_path)

    src = _load_source_cache(io_cfg)
    if not src.get("anomaly_targets"):
        raise ValueError("source cache must be anomaly-target cache (io.anomaly_targets=true)")

    basin = io_cfg.get("basin") or {"min_lat": 18.0, "max_lat": 31.0, "min_lon": -98.0, "max_lon": -81.0}
    grid_shape = tuple(io_cfg.get("field", {}).get("grid_shape", [52, 68]))
    lats, lons = _gom_grid(basin, grid_shape)
    log_bathy, land_mask = _static_bathy_mask(lats, lons)

    juld = np.asarray(src["JULD"], dtype=np.float32)
    lat = np.asarray(src["LAT"], dtype=np.float32)
    lon = np.asarray(src["LON"], dtype=np.float32)
    dates = sample_dates(juld, dataset_tag=src.get("dataset_tag", "argo_v2"), v2_src=io_cfg.get("v2_src"))
    unique_dates = np.unique(dates)
    max_dates = io_cfg.get("field", {}).get("max_dates")
    if max_dates is not None:
        unique_dates = unique_dates[: int(max_dates)]

    n_dates = len(unique_dates)
    fields = np.zeros((n_dates, 3, grid_shape[0], grid_shape[1]), dtype=np.float32)
    date_to_idx = {str(d): i for i, d in enumerate(unique_dates)}

    for i, d in enumerate(unique_dates):
        from datetime import date

        day = date.fromisoformat(str(d)[:10])
        dt = __import__("datetime").datetime.combine(day, __import__("datetime").time())
        # zero-fill land/missing SLA: NaN in any input pixel poisons every conv output
        fields[i, 0] = np.nan_to_num(_sla_grid_for_date(dt, lats, lons), nan=0.0)
        fields[i, 1] = log_bathy
        fields[i, 2] = land_mask

    n = len(juld)
    # -1 = date not kept (max_dates truncation); FieldDataset masks by date index so
    # these samples are never served
    sample_date_idx = np.full(n, -1, dtype=np.int64)
    sample_pixel_rc = np.zeros((n, 2), dtype=np.int64)
    for i in range(n):
        sample_date_idx[i] = date_to_idx.get(str(dates[i]), -1)
        sample_pixel_rc[i] = _latlon_to_rc(float(lat[i]), float(lon[i]), lats, lons)

    payload = {
        "fields": fields,
        "dates": unique_dates,
        "sample_date_idx": sample_date_idx,
        "sample_pixel_rc": sample_pixel_rc,
        "grid_lats": lats,
        "grid_lons": lons,
        "inputs": src["inputs"],
        "targets": src["targets"],
        "pca_models": src["pca_models"],
        "pcs_by_name": src.get("pcs_by_name"),
        "outputs": src["outputs"],
        "weights": src["weights"],
        "profiles": src["profiles"],
        "true_profiles": src.get("true_profiles"),
        "LAT": lat,
        "LON": lon,
        "PRES": src["PRES"],
        "JULD": juld,
        "climatology": src.get("climatology"),
        "clim_profiles": src.get("clim_profiles"),
        "anomaly_targets": True,
        "ssh_obs_adt": src.get("ssh_obs_adt"),
        "ssh_obs_sla": src.get("ssh_obs_sla"),
        "clim_steric": src.get("clim_steric"),
        "steric_calibration": src.get("steric_calibration"),
        "input_params": src.get("input_params", {}),
        "config_hash": chash,
        "dataset_tag": "argo_field",
        "min_depth": src.get("min_depth", 0),
        "max_depth": src.get("max_depth"),
        "source_cache": io_cfg.get("field", {}).get("source_cache"),
    }
    return write_train_cache(payload, cache_dir, chash)
