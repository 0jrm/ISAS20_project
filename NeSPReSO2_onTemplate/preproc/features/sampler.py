"""Cube sampling, bilinear weights, and FieldProvider (Component B)."""

from __future__ import annotations

import sys
from collections import OrderedDict
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
    time_indices_of_days,
)
from preproc.features.operators import GRAVITY, apply_operator  # noqa: E402

EARTH_ROTATION_RATE = 7.2921e-5


class MissingCubePlaneError(ValueError):
    """Raised when sampling a whitelisted missing cube day (explicit NaN plane)."""


class _LRUDict(OrderedDict):
    """Bounded cache: evicts the least-recently-used entry past ``maxsize``.

    Derived cube planes are large (float32 grids); an unbounded dict here is a
    memory-growth risk on full-dataset exports (thousands of unique days).
    """

    def __init__(self, maxsize: int = 256):
        super().__init__()
        self.maxsize = maxsize

    def __setitem__(self, key: Any, value: Any) -> None:
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        if len(self) > self.maxsize:
            oldest = next(iter(self))
            del self[oldest]

    def __getitem__(self, key: Any) -> Any:
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value


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


def time_indices_of_jd(dates_jd: np.ndarray) -> np.ndarray:
    """Vectorized batch of ``time_index_of_jd`` — one astropy conversion for all dates."""
    jd = np.asarray(dates_jd, dtype=np.float64)
    days = Time(jd, format="jd").datetime64.astype("datetime64[D]")
    return time_indices_of_days(days)


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

    def __init__(
        self,
        cube_path: Path | str | None = None,
        *,
        plane_cache_size: int = 512,
        stack_cache_size: int = 128,
        derived_cache_size: int = 512,
    ):
        if cube_path is None or str(cube_path).strip() == "":
            self.cube_path = default_cube_path(_ROOT)
        else:
            p = Path(cube_path)
            self.cube_path = p if p.is_absolute() else (_ROOT / p).resolve()
        if not self.cube_path.exists():
            raise FileNotFoundError(f"cube not found: {self.cube_path}")
        self.root = zarr.open(str(self.cube_path), mode="r")
        self.plane_cache_size = plane_cache_size
        self.stack_cache_size = stack_cache_size
        self.derived_cache_size = derived_cache_size
        self._weights: dict[str, sparse.csr_matrix] = {}
        self._missing_days = set(self.root.attrs.get("missing_days", []))
        self._allowed_missing = set(cube_schema.ALLOWED_MISSING_DAYS)
        self._basin_mask_cache: dict[tuple, np.ndarray] = {}
        self._basin_cache: dict[tuple[str, int], float] = {}

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

    def _basin_mask_for(self, channel: str, basin_cfg: Mapping[str, Any]) -> np.ndarray:
        key = (channel, tuple(sorted(basin_cfg.items())))
        if key not in self._basin_mask_cache:
            lats, lons = self.grid_coords(channel)
            self._basin_mask_cache[key] = basin_mask(lats, lons, basin_cfg)
        return self._basin_mask_cache[key]

    def basin_mean(self, channel: str, t_idx: int, basin_cfg: Mapping[str, Any] | None = None) -> float:
        basin_cfg = basin_cfg or DEFAULT_BASIN_CFG
        mask = self._basin_mask_for(channel, basin_cfg)
        plane = self.plane(channel, t_idx)
        vals = plane[mask]
        if vals.size == 0 or not np.any(np.isfinite(vals)):
            raise ValueError(f"missing basin mean for {channel} at t={t_idx}")
        return float(np.nanmean(vals))

    def _read_tendency_stack(self, ch: str, t_idx: int, window: int) -> np.ndarray:
        """Stack of ``window`` consecutive daily planes ending at ``t_idx`` (oldest first).

        Contiguous ranges are read as a single chunk-aligned slab instead of one
        ``plane()`` call per day. Falls back to the clamped per-day loop only near
        the start of the cube time axis, matching the original boundary behavior.
        """
        if t_idx - window + 1 >= 0:
            for t_chk in range(t_idx - window + 1, t_idx + 1):
                self._assert_plane_available(ch, t_chk)
            return np.asarray(self.root[ch][t_idx - window + 1 : t_idx + 1], dtype=np.float32)
        planes = []
        for k in range(window - 1, -1, -1):
            t_k = max(0, t_idx - k)
            self._assert_plane_available(ch, t_k)
            planes.append(self.plane(ch, t_k))
        return np.stack(planes, axis=0)

    @staticmethod
    def _group_by_t_idx(time_idx: np.ndarray) -> list[tuple[int, np.ndarray]]:
        """Profile indices grouped by cube day, ascending — one group per unique day."""
        order = np.argsort(time_idx, kind="stable")
        sorted_t = time_idx[order]
        boundaries = np.flatnonzero(np.diff(sorted_t)) + 1
        starts = np.concatenate(([0], boundaries))
        ends = np.concatenate((boundaries, [len(order)]))
        return [(int(sorted_t[s]), order[s:e]) for s, e in zip(starts, ends)]

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

        time_idx = time_indices_of_jd(np.asarray(dates_jd, dtype=np.float64))
        t_groups = self._group_by_t_idx(time_idx)

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
                for t_idx, idx_arr in t_groups:
                    bkey = (channel, t_idx)
                    if bkey not in self._basin_cache:
                        self._basin_cache[bkey] = self.basin_mean(channel, t_idx)
                    values[idx_arr, j] = self._basin_cache[bkey]
                    valid[idx_arr, j] = True

        plane_cache: _LRUDict = _LRUDict(maxsize=self.plane_cache_size)
        stack_cache: _LRUDict = _LRUDict(maxsize=self.stack_cache_size)
        derived_planes: _LRUDict = _LRUDict(maxsize=self.derived_cache_size)

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
            w_full = self.weights_for(ch, lats, lons)

            for t_idx, idx_arr in t_groups:
                if op == "tendency":
                    window = int(param)
                    dkey = ("tendency", ch, t_idx, window)
                elif op in ("grad", "geo_uv"):
                    # geo_uv reuses grad's gx/gy planes; f(lat) is applied per-profile after sampling.
                    dkey = ("grad", ch, t_idx, scale_lbl)
                else:
                    dkey = (op, ch, t_idx, scale_lbl)

                if dkey not in derived_planes:
                    if op == "tendency":
                        window = int(param)
                        skey = (ch, t_idx, window)
                        if skey not in stack_cache:
                            stack_cache[skey] = self._read_tendency_stack(ch, t_idx, window)
                        derived_planes[dkey] = apply_operator(
                            "tendency", _field_plane(ch, t_idx), stack=stack_cache[skey], window_days=window
                        )
                    else:
                        scale_deg = resolve_scale_deg(ch, scale_lbl, feature_spec)
                        field = _field_plane(ch, t_idx)
                        if op in ("grad", "geo_uv"):
                            derived_planes[dkey] = apply_operator(
                                "grad", field, lats=glat, lons=glon, scale_deg=scale_deg, grid_step_deg=grid_step
                            )
                        elif op == "value":
                            derived_planes[dkey] = apply_operator(
                                "value", field, lats=glat, lons=glon, scale_deg=scale_deg, grid_step_deg=grid_step
                            )
                        elif op == "value_centered":
                            bkey = (ch, t_idx)
                            if bkey not in self._basin_cache:
                                self._basin_cache[bkey] = self.basin_mean(ch, t_idx)
                            derived_planes[dkey] = apply_operator(
                                "value_centered",
                                field,
                                lats=glat,
                                lons=glon,
                                scale_deg=scale_deg,
                                grid_step_deg=grid_step,
                                basin_value=self._basin_cache[bkey],
                            )
                        elif op == "laplacian":
                            derived_planes[dkey] = apply_operator(
                                "laplacian", field, lats=glat, lons=glon, scale_deg=scale_deg, grid_step_deg=grid_step
                            )
                        else:
                            raise ValueError(f"unknown op {op}")

                w_group = w_full[idx_arr]
                if op == "grad":
                    gx, gy = derived_planes[dkey]
                    plane_pick = gx if ".grad_x@" in name else gy
                    sampled, val_w = sample_plane(w_group, plane_pick)
                    values[idx_arr, j] = sampled
                    valid[idx_arr, j] = val_w
                elif op == "geo_uv":
                    gx, gy = derived_planes[dkey]
                    lat_group = lats[idx_arr].astype(np.float64)
                    f = 2.0 * EARTH_ROTATION_RATE * np.sin(np.radians(lat_group))
                    f = np.where(np.abs(f) < 1e-8, 1e-8, f)
                    if ".geo_u@" in name:
                        # u = (-g/f) * gy — validity tracks gy alone, matching the pre-refactor
                        # single-plane cache (op_geo_uv returned u/v; isfinite(u) == isfinite(gy)).
                        gy_s, gy_valid = sample_plane(w_group, gy)
                        sampled = ((-GRAVITY / f) * gy_s).astype(np.float32)
                        valid[idx_arr, j] = gy_valid
                    else:
                        gx_s, gx_valid = sample_plane(w_group, gx)
                        sampled = ((GRAVITY / f) * gx_s).astype(np.float32)
                        valid[idx_arr, j] = gx_valid
                    values[idx_arr, j] = sampled
                else:
                    sampled, val_w = sample_plane(w_group, derived_planes[dkey])
                    values[idx_arr, j] = sampled
                    valid[idx_arr, j] = val_w

        return FeatureTable(names=names, values=values, units=units, valid_mask=valid)
