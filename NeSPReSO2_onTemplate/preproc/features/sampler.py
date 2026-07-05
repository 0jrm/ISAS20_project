"""Cube sampling, bilinear weights, and FieldProvider (Component B)."""

from __future__ import annotations

import sys
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping, Protocol

import numpy as np
import zarr
from astropy.time import Time
from scipy import sparse

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from preproc.cube import cube_schema  # noqa: E402
from preproc.cube.cube_schema import (  # noqa: E402
    ALLOWED_MISSING_DAYS,
    DEFAULT_BASIN_CFG,
    LOCAL_SIGMA_DEG,
    PRODUCT_SPECS,
    default_cube_path,
    time_index_of,
)
from preproc.features.operators import apply_operator  # noqa: E402


class MissingCubePlaneError(ValueError):
    """Raised when sampling a whitelisted missing cube day (explicit NaN plane)."""


class FieldProvider(Protocol):
    def sample(self, feature_spec: Mapping[str, Any], lats: np.ndarray, lons: np.ndarray, dates: np.ndarray) -> "FeatureTable": ...


@dataclass
class FeatureTable:
    names: list[str]
    values: np.ndarray
    units: list[str]
    valid_mask: np.ndarray


def time_index_of_jd(jd: float) -> int:
    """Convert astropy Julian date to cube time index."""
    dt = Time(jd, format="jd").datetime
    return time_index_of(dt)


def time_index_of_matlab_datenum(datenum: float) -> int:
    """Convert v2 MATLAB datenum to cube time index via astropy."""
    v2_src = Path("/unity/g2/jmiranda/v2-nespreso/src")
    if str(v2_src) not in sys.path:
        sys.path.insert(0, str(v2_src))
    from nespreso.utils.time import datenum_to_datetime

    return time_index_of(datenum_to_datetime(float(datenum)))


def _lon_in_range(lon_grid: np.ndarray, min_lon: float, max_lon: float) -> np.ndarray:
    inside = (lon_grid >= min_lon) & (lon_grid <= max_lon)
    if min_lon < 0:
        inside |= (lon_grid >= min_lon + 360.0) & (lon_grid <= max_lon + 360.0)
    return inside


def _lon_gt(lon_grid: np.ndarray, threshold: float) -> np.ndarray:
    if threshold >= 0:
        return lon_grid > threshold
    return (lon_grid > threshold) | (lon_grid > threshold + 360.0)


def basin_mask(lats: np.ndarray, lons: np.ndarray, basin_cfg: Mapping[str, Any]) -> np.ndarray:
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    inclusion = (
        (lat_grid >= float(basin_cfg["min_lat"]))
        & (lat_grid <= float(basin_cfg["max_lat"]))
        & _lon_in_range(lon_grid, float(basin_cfg["min_lon"]), float(basin_cfg["max_lon"]))
    )
    if basin_cfg.get("exclude_lat") is not None and basin_cfg.get("exclude_lon") is not None:
        ex = (lat_grid < float(basin_cfg["exclude_lat"])) & _lon_gt(lon_grid, float(basin_cfg["exclude_lon"]))
        inclusion &= ~ex
    return inclusion


def build_bilinear_weights(
    grid_lats: np.ndarray,
    grid_lons: np.ndarray,
    sample_lats: np.ndarray,
    sample_lons: np.ndarray,
) -> sparse.csr_matrix:
    """Sparse (n_profiles x n_grid) bilinear interpolation weights."""
    n = len(sample_lats)
    n_lat, n_lon = len(grid_lats), len(grid_lons)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    for p in range(n):
        lat = float(sample_lats[p])
        lon = float(sample_lons[p])
        if lon > 180:
            lon -= 360
        i = int(np.searchsorted(grid_lats, lat) - 1)
        j = int(np.searchsorted(grid_lons, lon) - 1)
        i = int(np.clip(i, 0, n_lat - 2))
        j = int(np.clip(j, 0, n_lon - 2))
        lat0, lat1 = float(grid_lats[i]), float(grid_lats[i + 1])
        lon0, lon1 = float(grid_lons[j]), float(grid_lons[j + 1])
        if lat1 == lat0 or lon1 == lon0:
            w = [1.0, 0.0, 0.0, 0.0]
        else:
            ty = (lat - lat0) / (lat1 - lat0)
            tx = (lon - lon0) / (lon1 - lon0)
            w = [
                (1 - ty) * (1 - tx),
                (1 - ty) * tx,
                ty * (1 - tx),
                ty * tx,
            ]
        corner_idx = [
            i * n_lon + j,
            i * n_lon + (j + 1),
            (i + 1) * n_lon + j,
            (i + 1) * n_lon + (j + 1),
        ]
        for k, wt in enumerate(w):
            if wt > 0:
                rows.append(p)
                cols.append(corner_idx[k])
                data.append(wt)
    n_grid = n_lat * n_lon
    return sparse.csr_matrix((data, (rows, cols)), shape=(n, n_grid))


def sample_plane(weights: sparse.csr_matrix, plane: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = np.asarray(plane, dtype=np.float64).reshape(-1)
    sampled = np.asarray(weights @ flat).ravel()
    valid_w = np.asarray(weights @ np.isfinite(flat).astype(np.float64)).ravel()
    valid = valid_w >= 0.5
    return sampled.astype(np.float32), valid


def resolve_scale_deg(channel: str, scale_label: str, features_cfg: Mapping[str, Any]) -> float:
    scale_mult = float(features_cfg.get("local_sigma_scale", 1.0))
    if scale_label == "local":
        return float(LOCAL_SIGMA_DEG[channel]) * scale_mult
    if scale_label.endswith("deg"):
        return float(scale_label.replace("deg", ""))
    raise ValueError(f"unknown scale label: {scale_label}")


def expand_feature_names(features_cfg: Mapping[str, Any]) -> list[tuple[str, str, str, str, float | int]]:
    """Return list of (name, op, channel, scale/window, param)."""
    out: list[tuple[str, str, str, str, float | int]] = []
    for scalar in features_cfg.get("scalars", []):
        out.append((scalar, "scalar", scalar, "", 0))
    for op_cfg in features_cfg.get("operators", []):
        op = op_cfg["op"]
        channels = op_cfg.get("channels", [])
        if op == "tendency":
            window = int(op_cfg.get("window_days", 7))
            for ch in channels:
                out.append((f"{ch}.tendency@{window}d", op, ch, "window", window))
        else:
            for ch in channels:
                for scale in op_cfg.get("scales", ["local"]):
                    scale_tag = scale if scale != "local" else "local"
                    if op == "grad":
                        out.append((f"{ch}.grad_x@{scale_tag}", op, ch, scale_tag, 0))
                        out.append((f"{ch}.grad_y@{scale_tag}", op, ch, scale_tag, 0))
                    elif op == "geo_uv":
                        out.append((f"{ch}.geo_u@{scale_tag}", op, ch, scale_tag, 0))
                        out.append((f"{ch}.geo_v@{scale_tag}", op, ch, scale_tag, 0))
                    else:
                        out.append((f"{ch}.{op}@{scale_tag}", op, ch, scale_tag, 0))
    return out


class CubeProvider:
    """Sample named features from a Zarr cube."""

    def __init__(self, cube_path: Path | str | None = None):
        if cube_path is None or str(cube_path).strip() == "":
            self.cube_path = default_cube_path(_ROOT)
        else:
            p = Path(cube_path)
            self.cube_path = p if p.is_absolute() else (_ROOT / p).resolve()
        if not self.cube_path.exists():
            raise FileNotFoundError(f"cube not found: {self.cube_path}")
        self.root = zarr.open(str(self.cube_path), mode="r")
        self._weights: dict[str, sparse.csr_matrix] = {}
        self._missing_days = set(self.root.attrs.get("missing_days", []))
        self._allowed_missing = set(cube_schema.ALLOWED_MISSING_DAYS)

    def grid_coords(self, channel: str) -> tuple[np.ndarray, np.ndarray]:
        lats = np.asarray(self.root[f"coords/{channel}_lat"], dtype=np.float64)
        lons = np.asarray(self.root[f"coords/{channel}_lon"], dtype=np.float64)
        return lats, lons

    def weights_for(self, channel: str, lats: np.ndarray, lons: np.ndarray) -> sparse.csr_matrix:
        key = channel
        if key not in self._weights:
            glat, glon = self.grid_coords(channel)
            self._weights[key] = build_bilinear_weights(glat, glon, lats, lons)
        return self._weights[key]

    def _date_str_for_t_idx(self, t_idx: int) -> str:
        return str(np.asarray(self.root["coords/time"][t_idx]))

    def _assert_plane_available(self, channel: str, t_idx: int) -> None:
        day = self._date_str_for_t_idx(t_idx)
        if day in self._missing_days and day in self._allowed_missing:
            raise MissingCubePlaneError(
                f"cannot sample {channel} on whitelisted missing day {day} (t_idx={t_idx})"
            )

    def plane(self, channel: str, t_idx: int) -> np.ndarray:
        self._assert_plane_available(channel, t_idx)
        return np.asarray(self.root[channel][t_idx], dtype=np.float32)

    def bathy_depth_plane(self) -> np.ndarray:
        return np.asarray(self.root["bathy"], dtype=np.float32)

    def basin_mean(self, channel: str, t_idx: int, basin_cfg: Mapping[str, Any] | None = None) -> float:
        basin_cfg = basin_cfg or DEFAULT_BASIN_CFG
        lats, lons = self.grid_coords(channel)
        mask = basin_mask(lats, lons, basin_cfg)
        plane = self.plane(channel, t_idx)
        vals = plane[mask]
        if vals.size == 0 or not np.any(np.isfinite(vals)):
            raise ValueError(f"missing basin mean for {channel} at t={t_idx}")
        return float(np.nanmean(vals))

    def sample(
        self,
        feature_spec: Mapping[str, Any],
        lats: np.ndarray,
        lons: np.ndarray,
        dates_jd: np.ndarray,
    ) -> FeatureTable:
        n = len(lats)
        expanded = expand_feature_names(feature_spec)
        names = [e[0] for e in expanded]
        values = np.full((n, len(names)), np.nan, dtype=np.float32)
        valid = np.zeros((n, len(names)), dtype=bool)
        units = ["1"] * len(names)

        time_idx = np.array([time_index_of_jd(float(d)) for d in dates_jd], dtype=int)

        # Scalars
        for j, (name, op, ch, _scale, _param) in enumerate(expanded):
            if op != "scalar":
                continue
            if name in ("timecos", "timesin"):
                continue  # filled later from JULD
            if name in ("latcos", "latsin", "loncos", "lonsin"):
                continue
            if name == "bathy_depth":
                w = self.weights_for("sst", lats, lons)
                plane = self.bathy_depth_plane()
                v, m = sample_plane(w, plane)
                values[:, j] = v
                valid[:, j] = m
            elif name.startswith("basin_"):
                ch_map = {"basin_sss": "sss", "basin_sst": "sst", "basin_ssh": "ssh"}
                channel = ch_map[name]
                basin_cache: dict[tuple[str, int], float] = getattr(self, "_basin_cache", {})
                for i in range(n):
                    t_idx = int(time_idx[i])
                    bkey = (channel, t_idx)
                    if bkey not in basin_cache:
                        basin_cache[bkey] = self.basin_mean(channel, t_idx)
                    values[i, j] = basin_cache[bkey]
                    valid[i, j] = True
                self._basin_cache = basin_cache

        plane_cache: dict[tuple[str, int], np.ndarray] = {}
        stack_cache: dict[tuple[str, int, int], np.ndarray] = {}
        derived_planes: dict[tuple, np.ndarray] = {}

        def _field_plane(ch: str, t_idx: int) -> np.ndarray:
            key = (ch, t_idx)
            if key not in plane_cache:
                plane_cache[key] = self.plane(ch, t_idx)
            return plane_cache[key]

        n_ops = sum(1 for e in expanded if e[1] != "scalar")
        op_idx = 0
        for j, (name, op, ch, scale_lbl, param) in enumerate(expanded):
            if op == "scalar":
                continue
            op_idx += 1
            print(f"[sampler] feature {op_idx}/{n_ops}: {name}", file=sys.stderr, flush=True)
            grid_step = PRODUCT_SPECS[ch].grid_step_deg
            glat, glon = self.grid_coords(ch)

            for i in range(n):
                t_idx = int(time_idx[i])
                lat_i = float(lats[i])
                if op == "tendency":
                    window = int(param)
                    dkey = (name, ch, t_idx, window)
                elif op == "geo_uv":
                    dkey = (name, ch, t_idx, scale_lbl, round(lat_i, 3))
                else:
                    dkey = (name, ch, t_idx, scale_lbl)

                if dkey not in derived_planes:
                    if op == "tendency":
                        window = int(param)
                        skey = (ch, t_idx, window)
                        if skey not in stack_cache:
                            planes = []
                            for k in range(window - 1, -1, -1):
                                t_k = max(0, t_idx - k)
                                self._assert_plane_available(ch, t_k)
                                planes.append(self.plane(ch, t_k))
                            stack_cache[skey] = np.stack(planes, axis=0)
                        derived_planes[dkey] = apply_operator(
                            "tendency", _field_plane(ch, t_idx), stack=stack_cache[skey], window_days=window
                        )
                    else:
                        scale_deg = resolve_scale_deg(ch, scale_lbl, feature_spec)
                        field = _field_plane(ch, t_idx)
                        if op == "grad":
                            gx, gy = apply_operator(
                                "grad", field, lats=glat, lons=glon, scale_deg=scale_deg, grid_step_deg=grid_step
                            )
                            derived_planes[dkey] = gx if ".grad_x@" in name else gy
                        elif op == "geo_uv":
                            u, v = apply_operator(
                                "geo_uv",
                                field,
                                lats=glat,
                                lons=glon,
                                scale_deg=scale_deg,
                                grid_step_deg=grid_step,
                                profile_lat=lat_i,
                            )
                            derived_planes[dkey] = u if ".geo_u@" in name else v
                        elif op == "value":
                            derived_planes[dkey] = apply_operator(
                                "value", field, lats=glat, lons=glon, scale_deg=scale_deg, grid_step_deg=grid_step
                            )
                        elif op == "value_centered":
                            bkey = (ch, t_idx)
                            vc_cache: dict[tuple[str, int], float] = getattr(self, "_basin_cache", {})
                            if bkey not in vc_cache:
                                vc_cache[bkey] = self.basin_mean(ch, t_idx)
                            self._basin_cache = vc_cache
                            derived_planes[dkey] = apply_operator(
                                "value_centered",
                                field,
                                lats=glat,
                                lons=glon,
                                scale_deg=scale_deg,
                                grid_step_deg=grid_step,
                                basin_value=vc_cache[bkey],
                            )
                        elif op == "laplacian":
                            derived_planes[dkey] = apply_operator(
                                "laplacian", field, lats=glat, lons=glon, scale_deg=scale_deg, grid_step_deg=grid_step
                            )
                        else:
                            raise ValueError(f"unknown op {op}")

                w = self.weights_for(ch, lats, lons)
                flat = np.asarray(derived_planes[dkey], dtype=np.float64).reshape(-1)
                wt = w.getrow(i)
                val = float((wt @ flat).sum())
                wsum = float((wt @ np.isfinite(flat).astype(np.float64)).sum())
                values[i, j] = val
                valid[i, j] = wsum >= 0.5

        return FeatureTable(names=names, values=values, units=units, valid_mask=valid)
