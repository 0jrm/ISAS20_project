"""Basin-wide daily means for L4 satellite fields (GoM)."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import xarray as xr

_UTILS = Path(__file__).resolve().parents[2] / "utils"
if str(_UTILS) not in sys.path:
    sys.path.insert(0, str(_UTILS))

from retrieve_sat import get_coord_names, get_variable_name, select_candidate_file

PRODUCT_VAR = {
    "basin_sss": ("sss", "sos"),
    "basin_sst": ("ostia", "analysed_sst"),
    "basin_ssh": ("ssh", "adt"),
}

# ponytail: v1 SMAP 8-day running mean; used only when CMEMS daily SSS lookup fails
SMAP_SSS_ROOT = os.environ.get(
    "NESPRESO_SMAP_SSS_ROOT",
    "/Net/work/ozavala/DATA/GOFFISH/SSS/SMAP_Global",
)


def _lon_in_range(lon_grid: np.ndarray, min_lon: float, max_lon: float) -> np.ndarray:
    """Match basin lon bounds on either -180/180 or 0/360 grids."""
    inside = (lon_grid >= min_lon) & (lon_grid <= max_lon)
    if min_lon < 0:
        inside |= (lon_grid >= min_lon + 360.0) & (lon_grid <= max_lon + 360.0)
    return inside


def _lon_gt(lon_grid: np.ndarray, threshold: float) -> np.ndarray:
    if threshold >= 0:
        return lon_grid > threshold
    return (lon_grid > threshold) | (lon_grid > threshold + 360.0)


def _basin_mask(lats: np.ndarray, lons: np.ndarray, basin_cfg: Mapping[str, Any]) -> np.ndarray:
    min_lat = float(basin_cfg["min_lat"])
    max_lat = float(basin_cfg["max_lat"])
    min_lon = float(basin_cfg["min_lon"])
    max_lon = float(basin_cfg["max_lon"])
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    inclusion = (
        (lat_grid >= min_lat)
        & (lat_grid <= max_lat)
        & _lon_in_range(lon_grid, min_lon, max_lon)
    )
    if basin_cfg.get("exclude_lat") is not None and basin_cfg.get("exclude_lon") is not None:
        ex_lat = float(basin_cfg["exclude_lat"])
        ex_lon = float(basin_cfg["exclude_lon"])
        exclusion = (lat_grid < ex_lat) & _lon_gt(lon_grid, ex_lon)
        inclusion = inclusion & ~exclusion
    return inclusion


def _field2d(field: xr.DataArray) -> np.ndarray:
    values = np.squeeze(np.asarray(field.values, dtype=np.float64))
    if values.ndim != 2:
        raise ValueError(f"expected 2-D field after squeeze, got shape {values.shape}")
    return values


def _masked_nanmean(values: np.ndarray, mask: np.ndarray) -> float:
    if values.shape != mask.shape:
        return float("nan")
    selected = values[mask]
    if selected.size == 0:
        return float("nan")
    return float(np.nanmean(selected))


def _mean_from_dataset(
    ds: xr.Dataset,
    var_key: str,
    basin_cfg: Mapping[str, Any],
) -> float:
    lat_name, lon_name, _ = get_coord_names(ds)
    if lat_name is None or lon_name is None:
        return float("nan")
    field = ds[var_key]
    if field.ndim > 2:
        field = field.isel({field.dims[0]: 0})
    values = _field2d(field)
    lats = np.asarray(ds[lat_name].values, dtype=np.float64)
    lons = np.asarray(ds[lon_name].values, dtype=np.float64)
    mask = _basin_mask(lats, lons, basin_cfg)
    return _masked_nanmean(values, mask)


def _smap_sss_path(query_dt: datetime) -> str | None:
    day_of_year = (query_dt - datetime(query_dt.year, 1, 1)).days + 1
    folder = os.path.join(SMAP_SSS_ROOT, str(query_dt.year))
    for ver in ("v06.0", "v05.0"):
        path = os.path.join(
            folder,
            f"RSS_smap_SSS_L3_8day_running_{query_dt.year}_{day_of_year:03d}_FNL_{ver}.nc",
        )
        if os.path.isfile(path):
            return path
    return None


def _smap_sss_mean(query_dt: datetime, basin_cfg: Mapping[str, Any]) -> float:
    path = _smap_sss_path(query_dt)
    if path is None:
        return float("nan")
    try:
        with xr.open_dataset(path, decode_times=True) as ds:
            var_key = "sss_smap_40km" if "sss_smap_40km" in ds else "sss_smap"
            if var_key not in ds:
                return float("nan")
            return _mean_from_dataset(ds, var_key, basin_cfg)
    except Exception:
        return float("nan")


def _field_mean_for_date(product: str, var: str, query_dt: datetime, basin_cfg: Mapping[str, Any]) -> float:
    candidate = select_candidate_file(product, query_dt)
    mean = float("nan")
    if candidate is not None:
        try:
            with xr.open_dataset(candidate, decode_times=True) as ds:
                var_key = get_variable_name(product, var, ds)
                if var_key is not None:
                    mean = _mean_from_dataset(ds, var_key, basin_cfg)
        except Exception:
            mean = float("nan")
    if product == "sss" and not np.isfinite(mean):
        mean = _smap_sss_mean(query_dt, basin_cfg)
    return mean


def _cache_path(cache_dir: str, basin_cfg: Mapping[str, Any]) -> str:
    payload = json.dumps(dict(basin_cfg), sort_keys=True).encode()
    tag = hashlib.sha256(payload).hexdigest()[:10]
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"basin_daily_means_{tag}.pkl")


def compute_basin_daily_means(
    juld: np.ndarray,
    basin_cfg: Mapping[str, Any],
    *,
    cache_dir: str = "../data/cache",
    v2_src: str | None = None,
    force: bool = False,
    max_workers: int = 8,
) -> dict[str, np.ndarray]:
    """
    Per-station basin means aligned with ``juld`` (MATLAB datenum or day-of-year float).

    Returns dict with keys ``sss``, ``sst``, ``ssh`` each shape ``(N,)``.
    """
    if v2_src:
        sys.path.insert(0, str(v2_src))
    from nespreso.utils.time import datenum_to_datetime

    path = _cache_path(cache_dir, basin_cfg)
    if os.path.isfile(path) and not force:
        with open(path, "rb") as f:
            cached = pickle.load(f)
        lookup = cached["lookup"]
    else:
        unique_days: dict[float, datetime] = {}
        for t in np.unique(juld):
            try:
                dt = datenum_to_datetime(float(t))
            except Exception:
                day = float(t) % 365.0
                dt = datetime(2000, 1, 1) + __import__("datetime").timedelta(days=day)
            unique_days[float(t)] = dt
        lookup: dict[str, dict[float, float]] = {k: {} for k in ("sss", "sst", "ssh")}
        # I/O-bound (netCDF opens over NFS) — each (day, product) mean is independent,
        # so threads parallelize the actual wait time without touching shared state
        # (select_candidate_file/get_product_files are lru_cache-backed and thread-safe).
        tasks = [
            (day_key, dt, out_key, product, var)
            for day_key, dt in unique_days.items()
            for out_key, (product, var) in PRODUCT_VAR.items()
        ]

        def _run(task):
            day_key, dt, out_key, product, var = task
            var_key = out_key.replace("basin_", "")
            return day_key, var_key, _field_mean_for_date(product, var, dt, basin_cfg)

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for day_key, var_key, mean in ex.map(_run, tasks):
                lookup[var_key][day_key] = mean
        with open(path, "wb") as f:
            pickle.dump({"lookup": lookup, "basin_cfg": dict(basin_cfg)}, f)

    out = {k: np.empty(len(juld), dtype=np.float32) for k in ("sss", "sst", "ssh")}
    for i, t in enumerate(juld):
        key = float(t)
        for var in out:
            out[var][i] = lookup[var].get(key, np.nan)
    return out


def _selfcheck() -> None:
    basin_cfg = {
        "min_lat": 18.0,
        "max_lat": 31.0,
        "min_lon": -98.0,
        "max_lon": -81.0,
        "exclude_lat": 23.0,
        "exclude_lon": -88.0,
    }
    for dt in (datetime(2010, 6, 15), datetime(2015, 3, 1), datetime(2020, 1, 1)):
        sss = _field_mean_for_date("sss", "sos", dt, basin_cfg)
        sst = _field_mean_for_date("ostia", "analysed_sst", dt, basin_cfg)
        assert np.isfinite(sss) and 20.0 < sss < 40.0, (dt, sss)
        assert np.isfinite(sst) and 270.0 < sst < 320.0, (dt, sst)


if __name__ == "__main__":
    _selfcheck()
    print("basin_stats selfcheck ok")
