"""Cube layout, domain, grids, and product metadata (Component A schema)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

CUBE_SCHEMA_VERSION = 1
DATA_REVISION = 2

# Core GoM domain (degrees)
CORE_LAT_MIN = 18.0
CORE_LAT_MAX = 31.0
CORE_LON_MIN = -98.0
CORE_LON_MAX = -81.0

REGIONAL_STENCIL_HALO_DEG = 1.0
SPATIAL_PAD_CELLS = 2

TIME_START = np.datetime64("2015-01-01", "D")
TIME_END = np.datetime64("2022-03-01", "D")

# Basin polygon for masked means (matches L4 config)
DEFAULT_BASIN_CFG: dict[str, Any] = {
    "min_lat": 18.0,
    "max_lat": 31.0,
    "min_lon": -98.0,
    "max_lon": -81.0,
    "exclude_lat": 23.0,
    "exclude_lon": -88.0,
}

# Explicit whitelist for known missing archive days (extend after review)
ALLOWED_MISSING_DAYS: list[str] = [
    "2022-03-01",  # CMEMS SSS archive ends 2022-02-28; endpoint is NaN not forward-filled
]

# Physical plausibility bounds (stored units)
PHYSICAL_BOUNDS: dict[str, tuple[float, float]] = {
    "sst": (-3.0, 35.0),
    "sss": (20.0, 40.0),
    "ssh": (-1.5, 1.5),
    "bathy": (0.0, 4500.0),
}

ROOT_DIRS = {
    "ostia": Path("/unity/g2/jmiranda/SubsurfaceFields/Data/OISST/OSTIA"),
    "sss": Path("/unity/g2/jmiranda/SubsurfaceFields/Data/CMEMS/SSS"),
    "ssh": Path("/unity/g2/jmiranda/SubsurfaceFields/Data/CMEMS/SSH"),
    "bathymetry": Path("/unity/g2/jmiranda/SubsurfaceFields/Data/Bathymetry"),
}


@dataclass(frozen=True)
class ProductSpec:
    """Metadata for one cube channel."""

    name: str
    root_key: str
    source_var: str
    lat_name: str
    lon_name: str
    time_name: str | None
    grid_step_deg: float
    native_units: str
    stored_units: str
    unit_transform: str | None
    file_regex: str | None
    date_fmt: str | None
    is_yearly: bool
    chunk_t: int
    transform: Callable[[np.ndarray], np.ndarray] | None = None


def _k_to_c(data: np.ndarray) -> np.ndarray:
    return (data - 273.15).astype(np.float32)


def _identity(data: np.ndarray) -> np.ndarray:
    return data.astype(np.float32)


def _gebco_depth(data: np.ndarray) -> np.ndarray:
    """GEBCO elevation (negative below sea level) -> positive depth."""
    out = -np.asarray(data, dtype=np.float32)
    return np.where(out < 0, 0.0, out).astype(np.float32)


PRODUCT_SPECS: dict[str, ProductSpec] = {
    "sst": ProductSpec(
        name="sst",
        root_key="ostia",
        source_var="analysed_sst",
        lat_name="lat",
        lon_name="lon",
        time_name="time",
        grid_step_deg=0.05,
        native_units="K",
        stored_units="degC",
        unit_transform="K-273.15",
        file_regex=r"(\d{14})-.*OSTIA.*\.nc",
        date_fmt="%Y%m%d%H%M%S",
        is_yearly=False,
        chunk_t=64,
        transform=_k_to_c,
    ),
    "sss": ProductSpec(
        name="sss",
        root_key="sss",
        source_var="sos",
        lat_name="latitude",
        lon_name="longitude",
        time_name="time",
        grid_step_deg=0.125,
        native_units="PSU",
        stored_units="PSU",
        unit_transform=None,
        file_regex=r"SSS_(\d{8})\.nc",
        date_fmt="%Y%m%d",
        is_yearly=False,
        chunk_t=64,
        transform=_identity,
    ),
    "ssh": ProductSpec(
        name="ssh",
        root_key="ssh",
        source_var="adt",
        lat_name="latitude",
        lon_name="longitude",
        time_name="time",
        grid_step_deg=0.25,
        native_units="m",
        stored_units="m",
        unit_transform=None,
        file_regex=r"SSH_(\d{4})\.nc",
        date_fmt="%Y",
        is_yearly=True,
        chunk_t=128,
        transform=_identity,
    ),
    "bathy": ProductSpec(
        name="bathy",
        root_key="bathymetry",
        source_var="elevation",
        lat_name="lat",
        lon_name="lon",
        time_name=None,
        grid_step_deg=0.05,
        native_units="m",
        stored_units="m_depth",
        unit_transform="neg_elev_to_depth",
        file_regex=None,
        date_fmt=None,
        is_yearly=False,
        chunk_t=1,
        transform=_gebco_depth,
    ),
}

LOCAL_SIGMA_DEG: dict[str, float] = {
    "sst": 0.15,
    "sss": 0.35,
    "ssh": 0.4,
}


def margin_deg_for_product(grid_step_deg: float) -> float:
    """Domain margin in degrees per spec §3.1."""
    stencil = max(REGIONAL_STENCIL_HALO_DEG, SPATIAL_PAD_CELLS * grid_step_deg)
    return stencil + 2.0 * grid_step_deg


def domain_bounds(product: str) -> tuple[float, float, float, float]:
    """Return (lat_min, lat_max, lon_min, lon_max) with product-specific margin."""
    spec = PRODUCT_SPECS[product]
    margin = margin_deg_for_product(spec.grid_step_deg)
    return (
        CORE_LAT_MIN - margin,
        CORE_LAT_MAX + margin,
        CORE_LON_MIN - margin,
        CORE_LON_MAX + margin,
    )


def build_lat_lon_grid(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    step_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Ascending lat/lon vectors snapped to ``step_deg``."""
    lat0 = np.ceil(lat_min / step_deg) * step_deg
    lat1 = np.floor(lat_max / step_deg) * step_deg
    lon0 = np.ceil(lon_min / step_deg) * step_deg
    lon1 = np.floor(lon_max / step_deg) * step_deg
    lats = np.arange(lat0, lat1 + 0.5 * step_deg, step_deg, dtype=np.float64)
    lons = np.arange(lon0, lon1 + 0.5 * step_deg, step_deg, dtype=np.float64)
    return lats, lons


def time_index_axis() -> np.ndarray:
    """Contiguous daily datetime64[D] from TIME_START through TIME_END inclusive."""
    end_exclusive = TIME_END + np.timedelta64(1, "D")
    return np.arange(TIME_START, end_exclusive, dtype="datetime64[D]")


def time_indices_of_days(days: np.ndarray) -> np.ndarray:
    """Vectorized map from ``datetime64[D]`` array to cube time indices; raises if any out of range."""
    days = np.asarray(days, dtype="datetime64[D]")
    idx = (days - TIME_START).astype(np.int64)
    bad = (idx < 0) | (days > TIME_END)
    if np.any(bad):
        bad_days = days[bad]
        raise IndexError(
            f"date(s) {bad_days.ravel()[:5].tolist()} not on cube time axis [{TIME_START}, {TIME_END}]"
        )
    return idx


def time_index_of(dt: date | datetime | np.datetime64) -> int:
    """Map calendar date to cube time index; raises if out of range."""
    if isinstance(dt, datetime):
        d = np.datetime64(dt.date(), "D")
    elif isinstance(dt, date):
        d = np.datetime64(dt, "D")
    else:
        d = np.datetime64(dt, "D")
    return int(time_indices_of_days(np.array([d]))[0])


def slice_indices(coord: np.ndarray, vmin: float, vmax: float) -> tuple[int, int]:
    """Index range [i0, i1) for ascending ``coord`` covering [vmin, vmax]."""
    coord = np.asarray(coord, dtype=np.float64)
    i0 = int(np.searchsorted(coord, vmin, side="left"))
    i1 = int(np.searchsorted(coord, vmax, side="right"))
    i0 = max(0, i0)
    i1 = min(len(coord), i1)
    if i0 >= i1:
        raise ValueError(f"empty slice for [{vmin}, {vmax}] on coord span [{coord.min()}, {coord.max()}]")
    return i0, i1


def lon_slice_indices(lons: np.ndarray, lon_min: float, lon_max: float) -> tuple[int, int]:
    """Like ``slice_indices`` but accepts 0–360° archives for negative-West domains."""
    lons = np.asarray(lons, dtype=np.float64)
    if lons.size == 0:
        raise ValueError("empty longitude coordinate")
    if lons.min() >= 0 and lons.max() > 180 and lon_max < 0:
        lon_min = lon_min % 360.0
        lon_max = lon_max % 360.0
    return slice_indices(lons, lon_min, lon_max)


def default_cube_path(template_root: Path | None = None) -> Path:
    root = template_root or Path(__file__).resolve().parents[2]
    return root / "data" / "cube" / "gom_cube.zarr"


def default_manifest_path(cube_path: Path) -> Path:
    return cube_path.parent / "build_manifest.json"


def default_validation_report_path(cube_path: Path) -> Path:
    return cube_path.parent / "validation_report.json"


def read_corrupt_sss_files(root: Path) -> set[str]:
    """Parse CMEMS SSS corrupt_files.txt if present."""
    path = root / "corrupt_files.txt"
    if not path.is_file():
        return set()
    bad: set[str] = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            bad.add(line)
    return bad


def cube_hash_metadata(cube_path: Path) -> dict[str, Any]:
    """Read attrs used in downstream cache hashing."""
    import zarr

    root = zarr.open(str(cube_path), mode="r")
    return {
        "cube_schema_version": int(root.attrs.get("cube_schema_version", CUBE_SCHEMA_VERSION)),
        "data_revision": int(root.attrs.get("data_revision", DATA_REVISION)),
        "data_through": str(root.attrs.get("data_through", TIME_END)),
    }


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_, np.generic)):
        return obj.item()
    return obj


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(dict(payload)), indent=2, sort_keys=True) + "\n")
