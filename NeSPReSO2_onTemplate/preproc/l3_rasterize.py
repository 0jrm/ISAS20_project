"""Rasterize sparse L3 surface observations into mask-native patch tensors around ARGO targets.

Aggregation per grid cell and time bin:
  inverse-variance weighting when sigma is available, else count-weighted mean.
  Empty cells: mask=0, value=0, age=large sentinel, uncertainty=nan, count=0.

ponytail: date-glob file discovery under data/raw; no product catalog DB.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Sequence

import numpy as np

FEATURE_NAMES = ("value", "mask", "age", "uncertainty", "count")
N_FEATURES = len(FEATURE_NAMES)
IDX_VALUE, IDX_MASK, IDX_AGE, IDX_UNC, IDX_COUNT = range(N_FEATURES)

EARTH_RADIUS_KM = 6371.0
MISSING_AGE_HOURS = 1.0e6


@dataclass(frozen=True)
class PatchGrid:
    """Lat/lon bin edges centered on target (iy, ix index center cell)."""

    lat0: float
    lon0: float
    half_deg: float
    step_deg: float
    lat_edges: np.ndarray
    lon_edges: np.ndarray

    @property
    def height(self) -> int:
        return len(self.lat_edges) - 1

    @property
    def width(self) -> int:
        return len(self.lon_edges) - 1

    @property
    def center_y(self) -> int:
        return self.height // 2

    @property
    def center_x(self) -> int:
        return self.width // 2


@dataclass(frozen=True)
class TargetPoint:
    lat: float
    lon: float
    time: datetime


def l3_geometry(l3_cfg: dict[str, Any]) -> tuple[int, int, int]:
    """Return (spatial_pad, temporal_pad, grid_size)."""
    half = float(l3_cfg["patch_half_deg"])
    step = float(l3_cfg["grid_step_deg"])
    pad = max(1, int(round(half / step)))
    windows = list(l3_cfg["time_windows_hours"])
    return pad, max(0, len(windows) - 1), 2 * pad + 1


def patch_grid(lat0: float, lon0: float, half_deg: float, step_deg: float) -> PatchGrid:
    pad = max(1, int(round(half_deg / step_deg)))
    # 2*pad+1 cells → 2*pad+2 edges (matches preproc_isas_sat spatial_size)
    lat_edges = lat0 + (np.arange(-pad, pad + 2, dtype=np.float64) * step_deg)
    lon_edges = lon0 + (np.arange(-pad, pad + 2, dtype=np.float64) * step_deg)
    return PatchGrid(lat0, lon0, half_deg, step_deg, lat_edges, lon_edges)


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    rlat1, rlon1, rlat2, rlon2 = map(math.radians, (lat1, lon1, lat2, lon2))
    dlat = rlat2 - rlat1
    dlon = rlon2 - rlon1
    a = math.sin(dlat / 2) ** 2 + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))


def _cell_index(edges: np.ndarray, val: float) -> int | None:
    idx = int(np.searchsorted(edges, val, side="right") - 1)
    if idx < 0 or idx >= len(edges) - 1:
        return None
    return idx


def time_bin_for_age(age_hours: float, windows_hours: Sequence[float]) -> int | None:
    """Cumulative lookback bins: age <= windows[i] assigns to bin i."""
    if age_hours < 0:
        return None
    for i, w in enumerate(windows_hours):
        if age_hours <= float(w):
            return i
    return len(windows_hours) - 1


def empty_bundle(n_time: int, height: int, width: int) -> np.ndarray:
    """Shape (N_FEATURES, T, H, W)."""
    out = np.zeros((N_FEATURES, n_time, height, width), dtype=np.float32)
    out[IDX_MASK] = 0.0
    out[IDX_AGE] = MISSING_AGE_HOURS
    out[IDX_UNC] = np.nan
    out[IDX_COUNT] = 0.0
    return out


def bin_observations(
    lats: np.ndarray,
    lons: np.ndarray,
    obs_times: Sequence[datetime],
    values: np.ndarray,
    sigmas: np.ndarray | None,
    target_time: datetime,
    grid: PatchGrid,
    windows_hours: Sequence[float],
) -> np.ndarray:
    """Rasterize point observations into feature bundle (5, T, H, W)."""
    n_time = len(windows_hours)
    bundle = empty_bundle(n_time, grid.height, grid.width)
    if len(lats) == 0:
        return bundle

    # ponytail: dict of cell lists; upgrade to histogram per bin for large obs counts
    cells: dict[tuple[int, int, int], list[tuple[float, float, float]]] = {}
    for i, (lat, lon, t_obs, val) in enumerate(zip(lats, lons, obs_times, values)):
        if not np.isfinite(val):
            continue
        iy = _cell_index(grid.lat_edges, float(lat))
        ix = _cell_index(grid.lon_edges, float(lon))
        if iy is None or ix is None:
            continue
        age_h = (target_time - t_obs).total_seconds() / 3600.0
        tb = time_bin_for_age(age_h, windows_hours)
        if tb is None:
            continue
        sigma = np.nan
        if sigmas is not None and i < len(sigmas):
            sigma = float(sigmas[i])
        cells.setdefault((tb, iy, ix), []).append((float(val), sigma, age_h))

    for (tb, iy, ix), obs_list in cells.items():
        vals = [o[0] for o in obs_list]
        sigs = [o[1] for o in obs_list]
        ages = [o[2] for o in obs_list]
        use_iv = all(np.isfinite(s) and s > 0 for s in sigs)
        if use_iv:
            w = np.array([1.0 / (s * s) for s in sigs], dtype=np.float64)
            v = float(np.sum(w * vals) / np.sum(w))
            unc = float(np.sqrt(1.0 / np.sum(w)))
        else:
            v = float(np.mean(vals))
            unc = float(np.std(vals)) if len(vals) > 1 else np.nan
        bundle[IDX_VALUE, tb, iy, ix] = v
        bundle[IDX_MASK, tb, iy, ix] = 1.0
        bundle[IDX_AGE, tb, iy, ix] = float(min(ages))
        bundle[IDX_UNC, tb, iy, ix] = unc if np.isfinite(unc) else np.nan
        bundle[IDX_COUNT, tb, iy, ix] = float(len(obs_list))

    return bundle


def coverage_metrics(
    bundle: np.ndarray,
    *,
    target: TargetPoint,
    track_lats: np.ndarray | None = None,
    track_lons: np.ndarray | None = None,
) -> dict[str, float]:
    mask = bundle[IDX_MASK]
    n_cells = mask.size
    observed = float(mask.sum())
    metrics: dict[str, float] = {
        "coverage_fraction": observed / max(1, n_cells),
        "observed_cells": observed,
        "total_cells": float(n_cells),
        "total_count": float(bundle[IDX_COUNT].sum()),
    }
    if track_lats is not None and len(track_lats):
        dists = [_haversine_km(target.lat, target.lon, la, lo) for la, lo in zip(track_lats, track_lons)]
        metrics["nearest_track_km"] = float(min(dists))
    else:
        metrics["nearest_track_km"] = float("nan")
    return metrics


def _nc_num_to_datetime(num, units: str, calendar: str | None = None) -> datetime:
    from netCDF4 import num2date

    dt = num2date(float(num), units=units, calendar=calendar or "standard")
    if hasattr(dt, "replace"):
        return dt.replace(tzinfo=None) if getattr(dt, "tzinfo", None) else dt
    return datetime.utcfromtimestamp(dt)


def _read_var(ds, names: Sequence[str]) -> np.ndarray:
    for name in names:
        if name in ds.variables:
            return np.asarray(ds.variables[name][:])
    raise KeyError(f"none of {names} found in {list(ds.variables.keys())[:20]}")


def _read_time_var(ds, names: Sequence[str] = ("time", "TIME", "t")) -> tuple[np.ndarray, str, str | None]:
    for name in names:
        if name not in ds.variables:
            continue
        var = ds.variables[name]
        units = getattr(var, "units", "days since 1950-01-01")
        cal = getattr(var, "calendar", None)
        return np.asarray(var[:]), units, cal
    raise KeyError("no time variable in netCDF")


def read_ssh_track(nc_path: str | Path) -> tuple[np.ndarray, np.ndarray, list[datetime], np.ndarray, np.ndarray | None]:
    """Return lats, lons, times, values, sigmas from Copernicus L3 along-track file."""
    from netCDF4 import Dataset

    with Dataset(str(nc_path), "r") as ds:
        lats = _read_var(ds, ("latitude", "lat", "LAT"))
        lons = _read_var(ds, ("longitude", "lon", "LON"))
        tnums, tunits, tcal = _read_time_var(ds)
        values = _read_var(ds, ("sla", "SLA", "sla_filtered", "adt"))
        sigmas = None
        for sname in ("sla_unfiltered", "sla_error", "err_sla", "sigma_sla"):
            if sname in ds.variables:
                sigmas = np.asarray(ds.variables[sname][:])
                break
        times = [_nc_num_to_datetime(t, tunits, tcal) for t in np.asarray(tnums).ravel()]
        lats = np.asarray(lats, dtype=np.float64).ravel()
        lons = np.asarray(lons, dtype=np.float64).ravel()
        lons = np.vectorize(_normalize_lon_deg)(lons)
        values = np.asarray(values, dtype=np.float64).ravel()
        if sigmas is not None:
            sigmas = np.asarray(sigmas, dtype=np.float64).ravel()
        n = min(len(lats), len(lons), len(times), len(values))
        return lats[:n], lons[:n], times[:n], values[:n], (sigmas[:n] if sigmas is not None else None)


def read_era5_wind(nc_path: str | Path) -> tuple[np.ndarray, np.ndarray, list[datetime], np.ndarray, np.ndarray]:
    """Return lat grid, lon grid, times, u10, v10 from ERA5 monthly netCDF."""
    from netCDF4 import Dataset

    with Dataset(str(nc_path), "r") as ds:
        lats = _read_var(ds, ("latitude", "lat"))
        lons = _read_var(ds, ("longitude", "lon"))
        tnums, tunits, tcal = _read_time_var(ds, ("valid_time", "time"))
        u10 = _read_var(ds, ("u10", "10m_u_component_of_wind"))
        v10 = _read_var(ds, ("v10", "10m_v_component_of_wind"))
        times = [_nc_num_to_datetime(t, tunits, tcal) for t in np.asarray(tnums).ravel()]
        return (
            np.asarray(lats, dtype=np.float64),
            np.asarray(lons, dtype=np.float64),
            times,
            np.asarray(u10, dtype=np.float64),
            np.asarray(v10, dtype=np.float64),
        )


def _normalize_lon_deg(lon: float) -> float:
    return ((float(lon) + 180.0) % 360.0) - 180.0


def _lon_delta_deg(lon: float, lon0: float) -> float:
    d = _normalize_lon_deg(lon) - _normalize_lon_deg(lon0)
    if d > 180.0:
        d -= 360.0
    elif d < -180.0:
        d += 360.0
    return d


def _in_bbox(lat, lon, lat0, lon0, half_deg) -> bool:
    return abs(lat - lat0) <= half_deg and abs(_lon_delta_deg(lon, lon0)) <= half_deg


def find_raw_files(raw_root: Path, subdir: str, pattern: str) -> list[Path]:
    d = raw_root / subdir
    if not d.is_dir():
        return []
    # ponytail: recursive glob — Copernicus toolbox keeps product subdirs
    return sorted(d.rglob(pattern))


def find_ssh_files_for_day(raw_root: Path, day: datetime) -> list[Path]:
    tag = day.strftime("%Y%m%d")
    return find_raw_files(raw_root, "altimetry_l3", f"*{tag}*.nc")


def find_era5_files_for_month(raw_root: Path, year: int, month: int) -> list[Path]:
    return find_raw_files(raw_root, "era5", f"era5_u10_v10_{year}{month:02d}_gom.nc")


def collect_ssh_paths(raw_root: Path, target_time: datetime, windows_hours: Sequence[float]) -> list[Path]:
    """Daily SSH files covering the lookback window."""
    max_h = float(max(windows_hours)) if windows_hours else 0.0
    start = target_time - timedelta(hours=max_h)
    paths: list[Path] = []
    day = start.date()
    end_day = target_time.date()
    while day <= end_day:
        paths.extend(find_ssh_files_for_day(raw_root, datetime.combine(day, datetime.min.time())))
        day += timedelta(days=1)
    return sorted(set(paths))


def collect_era5_paths(raw_root: Path, target_time: datetime, windows_hours: Sequence[float]) -> list[Path]:
    max_h = float(max(windows_hours)) if windows_hours else 0.0
    start = target_time - timedelta(hours=max_h)
    paths: list[Path] = []
    y, m = start.year, start.month
    while (y, m) <= (target_time.year, target_time.month):
        paths.extend(find_era5_files_for_month(raw_root, y, m))
        if m == 12:
            y, m = y + 1, 1
        else:
            m += 1
    return sorted(set(paths))


def rasterize_ssh_for_target(
    nc_paths: Sequence[Path],
    target: TargetPoint,
    grid: PatchGrid,
    windows_hours: Sequence[float],
) -> tuple[np.ndarray, dict[str, float]]:
    all_lat, all_lon, all_t, all_v, all_s = [], [], [], [], []
    for p in nc_paths:
        try:
            la, lo, ti, va, si = read_ssh_track(p)
        except (KeyError, OSError):
            continue
        for i in range(len(la)):
            if _in_bbox(la[i], lo[i], target.lat, target.lon, grid.half_deg):
                all_lat.append(la[i])
                all_lon.append(lo[i])
                all_t.append(ti[i])
                all_v.append(va[i])
                if si is not None:
                    all_s.append(si[i])
    sigmas = np.array(all_s, dtype=np.float64) if all_s else None
    bundle = bin_observations(
        np.array(all_lat),
        np.array(all_lon),
        all_t,
        np.array(all_v, dtype=np.float64),
        sigmas,
        target.time,
        grid,
        windows_hours,
    )
    metrics = coverage_metrics(
        bundle,
        target=target,
        track_lats=np.array(all_lat) if all_lat else None,
        track_lons=np.array(all_lon) if all_lon else None,
    )
    return bundle, metrics


def rasterize_era5_wind_for_target(
    nc_paths: Sequence[Path],
    target: TargetPoint,
    grid: PatchGrid,
    windows_hours: Sequence[float],
    component: str,
) -> tuple[np.ndarray, dict[str, float]]:
    """Rasterize u10 or v10; ERA5 cells treated as point obs at cell center."""
    all_lat, all_lon, all_t, all_v = [], [], [], []
    for p in nc_paths:
        try:
            lat_grid, lon_grid, times, u10, v10 = read_era5_wind(p)
        except (KeyError, OSError):
            continue
        field = u10 if component == "u" else v10
        if field.ndim != 3:
            continue
        if field.shape[0] == len(times):
            t_dim = 0
        elif field.shape[2] == len(times):
            t_dim = 2
        else:
            continue
        lat_min = target.lat - grid.half_deg
        lat_max = target.lat + grid.half_deg
        lon_min = target.lon - grid.half_deg
        lon_max = target.lon + grid.half_deg
        for it, t_obs in enumerate(times):
            max_age_h = max(windows_hours)
            age_h = (target.time - t_obs).total_seconds() / 3600.0
            if age_h < 0 or age_h > max_age_h:
                continue
            for iy, la in enumerate(lat_grid):
                if la < lat_min or la > lat_max:
                    continue
                for ix, lo in enumerate(lon_grid):
                    if lo < lon_min or lo > lon_max:
                        continue
                    if t_dim == 0:
                        val = field[it, iy, ix]
                    else:
                        val = field[iy, ix, it]
                    if not np.isfinite(val):
                        continue
                    all_lat.append(float(la))
                    all_lon.append(float(lo))
                    all_t.append(t_obs)
                    all_v.append(float(val))
    bundle = bin_observations(
        np.array(all_lat),
        np.array(all_lon),
        all_t,
        np.array(all_v, dtype=np.float64),
        None,
        target.time,
        grid,
        windows_hours,
    )
    metrics = coverage_metrics(bundle, target=target)
    return bundle, metrics


def build_target_from_cache(cache: dict, idx: int, *, dataset_tag: str, v2_src: str | None) -> TargetPoint:
    from base.split_utils import juld_to_datetime

    lat = float(np.asarray(cache["LAT"]).ravel()[idx])
    lon = float(np.asarray(cache["LON"]).ravel()[idx])
    juld = float(np.asarray(cache["JULD"]).ravel()[idx])
    t = juld_to_datetime(juld, dataset_tag=dataset_tag, v2_src=v2_src)
    return TargetPoint(lat=lat, lon=lon, time=t)


def l3_config_hash(l3_cfg: dict[str, Any], l4_cfg: dict[str, Any] | None = None) -> str:
    import hashlib
    import json

    blob: dict[str, Any] = {"l3": l3_cfg}
    if l4_cfg and l4_cfg.get("enabled"):
        blob["l4"] = l4_cfg
    payload = json.dumps(blob, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:12]


def center_value_from_bundle(bundle: np.ndarray) -> float:
    """Latest time bin, center cell value (mask may be 0)."""
    t_last = bundle.shape[1] - 1
    cy = bundle.shape[2] // 2
    cx = bundle.shape[3] // 2
    return float(bundle[IDX_VALUE, t_last, cy, cx])
