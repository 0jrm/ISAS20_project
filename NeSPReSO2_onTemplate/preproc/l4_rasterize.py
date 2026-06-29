"""Sample L4 gridded SSH onto the L3 patch grid for mask-augmentation (Phase 4).

ponytail: nearest-neighbor on 1-D lat/lon DUACS grids; no curvilinear support yet.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence

import numpy as np

from preproc.l3_rasterize import (
    PatchGrid,
    TargetPoint,
    _read_time_var,
    _read_var,
    find_raw_files,
)


def find_l4_ssh_files_for_day(raw_root: Path, day: datetime) -> list[Path]:
    tag = day.strftime("%Y%m%d")
    return find_raw_files(raw_root, "altimetry_l4_aux", f"*{tag}*.nc")


def collect_l4_ssh_paths(raw_root: Path, target_time: datetime, windows_hours: Sequence[float]) -> list[Path]:
    max_h = float(max(windows_hours)) if windows_hours else 0.0
    start = target_time - timedelta(hours=max_h)
    paths: list[Path] = []
    day = start.date()
    end_day = target_time.date()
    while day <= end_day:
        paths.extend(find_l4_ssh_files_for_day(raw_root, datetime.combine(day, datetime.min.time())))
        day += timedelta(days=1)
    return sorted(set(paths))


def _nearest_index(axis: np.ndarray, val: float) -> int:
    return int(np.argmin(np.abs(axis - val)))


def _pick_time_slice(field: np.ndarray, times: list[datetime], target_time: datetime) -> np.ndarray:
    if field.ndim != 3:
        raise ValueError(f"expected 3-D L4 field, got shape {field.shape}")
    if field.shape[0] == len(times):
        t_dim = 0
    elif field.shape[2] == len(times):
        t_dim = 2
    else:
        raise ValueError(f"time dimension mismatch: field {field.shape}, n_times={len(times)}")
    if not times:
        raise ValueError("empty L4 time axis")
    deltas = [abs((target_time - t).total_seconds()) for t in times]
    it = int(np.argmin(deltas))
    if t_dim == 0:
        return np.asarray(field[it], dtype=np.float64)
    return np.asarray(field[:, :, it], dtype=np.float64)


def read_l4_ssh_snapshot(nc_path: str | Path, target_time: datetime) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Return lat1d, lon1d, sla_2d, err_2d|None for the nearest daily snapshot."""
    from netCDF4 import Dataset

    from preproc.l3_rasterize import _nc_num_to_datetime

    with Dataset(str(nc_path), "r") as ds:
        lats = np.asarray(_read_var(ds, ("latitude", "lat", "LAT")), dtype=np.float64).ravel()
        lons = np.asarray(_read_var(ds, ("longitude", "lon", "LON")), dtype=np.float64).ravel()
        sla = np.asarray(_read_var(ds, ("sla", "SLA", "adt", "ADT")), dtype=np.float64)
        err = None
        for name in ("err_sla", "sla_error", "error_sla"):
            if name in ds.variables:
                err = np.asarray(ds.variables[name][:], dtype=np.float64)
                break
        tnums, tunits, tcal = _read_time_var(ds)
        times = [_nc_num_to_datetime(t, tunits, tcal) for t in np.asarray(tnums).ravel()]
        sla2 = _pick_time_slice(sla, times, target_time)
        err2 = _pick_time_slice(err, times, target_time) if err is not None else None
        if sla2.ndim != 2:
            sla2 = sla2.reshape(len(lats), len(lons))
        if err2 is not None and err2.ndim != 2:
            err2 = err2.reshape(len(lats), len(lons))
        return lats, lons, sla2, err2


def sample_l4_ssh_patch(
    nc_paths: Sequence[Path],
    target: TargetPoint,
    grid: PatchGrid,
    windows_hours: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-neighbor L4 SLA on patch cell centers; shape (T, H, W)."""
    n_time = len(windows_hours)
    h, w = grid.height, grid.width
    values = np.zeros((n_time, h, w), dtype=np.float64)
    errors = np.full((n_time, h, w), np.nan, dtype=np.float64)
    if not nc_paths:
        return values, errors

    cell_lats = 0.5 * (grid.lat_edges[:-1] + grid.lat_edges[1:])
    cell_lons = 0.5 * (grid.lon_edges[:-1] + grid.lon_edges[1:])

    for tb, window_h in enumerate(windows_hours):
        obs_time = target.time - timedelta(hours=float(window_h) * 0.5)
        for p in nc_paths:
            try:
                lats, lons, sla2, err2 = read_l4_ssh_snapshot(p, obs_time)
            except (KeyError, OSError, ValueError):
                continue
            for iy, la in enumerate(cell_lats):
                if la < lats.min() or la > lats.max():
                    continue
                jy = _nearest_index(lats, float(la))
                for ix, lo in enumerate(cell_lons):
                    if lo < lons.min() or lo > lons.max():
                        continue
                    jx = _nearest_index(lons, float(lo))
                    val = float(sla2[jy, jx])
                    if np.isfinite(val):
                        values[tb, iy, ix] = val
                        if err2 is not None:
                            e = float(err2[jy, jx])
                            if np.isfinite(e):
                                errors[tb, iy, ix] = e
            break
    return values, errors
