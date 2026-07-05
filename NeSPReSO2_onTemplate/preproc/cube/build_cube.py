#!/usr/bin/env python3
"""Build regional GoM space-time Zarr cube from NetCDF archives (Component A)."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path
from typing import Any

import netCDF4 as nc
import numpy as np
import zarr
from numcodecs import Blosc
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from preproc.cube.cube_schema import (  # noqa: E402
    ALLOWED_MISSING_DAYS,
    CUBE_SCHEMA_VERSION,
    DATA_REVISION,
    PRODUCT_SPECS,
    ProductSpec,
    ROOT_DIRS,
    TIME_END,
    TIME_START,
    build_lat_lon_grid,
    default_cube_path,
    default_manifest_path,
    domain_bounds,
    read_corrupt_sss_files,
    lon_slice_indices,
    slice_indices,
    time_index_axis,
    time_index_of,
    write_json,
)


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return "unknown"


def _coord_names(ds: nc.Dataset, spec: ProductSpec) -> tuple[str, str]:
    lat_candidates = [spec.lat_name, "latitude", "lat"]
    lon_candidates = [spec.lon_name, "longitude", "lon"]
    lat = next((n for n in lat_candidates if n in ds.variables), None)
    lon = next((n for n in lon_candidates if n in ds.variables), None)
    if lat is None or lon is None:
        raise KeyError(f"could not resolve lat/lon in {getattr(ds, 'filepath', '?')}")
    return lat, lon


def _decode_array(data: np.ndarray, var: nc.Variable) -> np.ndarray:
    arr = np.array(data, dtype=np.float64)
    fill = getattr(var, "_FillValue", None)
    if fill is not None:
        arr = np.where(arr == fill, np.nan, arr)
    scale = getattr(var, "scale_factor", None)
    offset = getattr(var, "add_offset", None)
    if scale is not None:
        arr = arr * float(scale)
    if offset is not None:
        arr = arr + float(offset)
    return arr


def _decode_var(var: nc.Variable) -> np.ndarray:
    return _decode_array(var[:], var)


def _list_product_files(spec_name: str) -> list[Path]:
    spec = PRODUCT_SPECS[spec_name]
    root = ROOT_DIRS[spec.root_key]
    if spec_name == "bathy":
        hits = sorted(root.glob("*.nc"))
        return hits[:1]

    pattern = re.compile(spec.file_regex) if spec.file_regex else None
    corrupt = read_corrupt_sss_files(root) if spec_name == "sss" else set()
    files: list[Path] = []
    for path in sorted(root.glob("*.nc")):
        name = path.name
        if name in corrupt:
            continue
        if spec_name == "ssh" and re.search(r"\(\d+\)\.nc$", name):
            continue
        if pattern is None or pattern.search(name):
            files.append(path)
    return files


def _time_to_date(t: Any) -> date:
    if isinstance(t, date) and not isinstance(t, datetime):
        return t
    if isinstance(t, datetime):
        return t.date()
    if hasattr(t, "date") and callable(getattr(t, "date")):
        return t.date()
    return date(int(t.year), int(t.month), int(t.day))


def _parse_file_date(spec_name: str, path: Path) -> date | None:
    spec = PRODUCT_SPECS[spec_name]
    if spec.file_regex is None:
        return None
    m = re.search(spec.file_regex, path.name)
    if not m:
        return None
    token = m.group(1)
    if spec.is_yearly:
        return date(int(token), 1, 1)
    if len(token) == 14:
        return datetime.strptime(token, spec.date_fmt).date()
    return datetime.strptime(token, spec.date_fmt).date()


def _read_hyperslab_plane(
    path: str,
    spec_name: str,
    lat_slice: tuple[int, int],
    lon_slice: tuple[int, int],
    time_idx: int | None = None,
) -> np.ndarray:
    spec = PRODUCT_SPECS[spec_name]
    with nc.Dataset(path, "r") as ds:
        var = ds.variables[spec.source_var]
        var.set_auto_maskandscale(False)
        if var.ndim == 2:
            slab = var[lat_slice[0] : lat_slice[1], lon_slice[0] : lon_slice[1]]
        elif var.ndim == 3:
            t = 0 if time_idx is None else time_idx
            slab = var[t, lat_slice[0] : lat_slice[1], lon_slice[0] : lon_slice[1]]
        elif var.ndim == 4:
            t = 0 if time_idx is None else time_idx
            slab = var[t, 0, lat_slice[0] : lat_slice[1], lon_slice[0] : lon_slice[1]]
        else:
            raise ValueError(f"unexpected ndim {var.ndim} for {spec.source_var} in {path}")
        data = _decode_array(slab, var)
        if spec.transform is not None:
            data = spec.transform(data)
        return np.squeeze(data).astype(np.float32)


def _pick_reference_file(spec_name: str) -> Path:
    """Prefer an archive file on the cube time axis (avoids legacy 0–360° SSS headers)."""
    files = _list_product_files(spec_name)
    if not files:
        raise FileNotFoundError(f"no files for {spec_name}")
    for path in files:
        fd = _parse_file_date(spec_name, path)
        if fd is None:
            continue
        d64 = np.datetime64(fd, "D")
        if TIME_START <= d64 <= TIME_END:
            return path
    return files[0]


def _indices_for_file(path: str, spec_name: str) -> tuple[int, int, int, int]:
    lat_min, lat_max, lon_min, lon_max = domain_bounds(spec_name)
    spec = PRODUCT_SPECS[spec_name]
    with nc.Dataset(path, "r") as ds:
        lat_name, lon_name = _coord_names(ds, spec)
        full_lat = np.asarray(ds.variables[lat_name][:], dtype=np.float64)
        full_lon = np.asarray(ds.variables[lon_name][:], dtype=np.float64)
    i0, i1 = slice_indices(full_lat, lat_min, lat_max)
    j0, j1 = lon_slice_indices(full_lon, lon_min, lon_max)
    return i0, i1, j0, j1


def _slice_coords(
    path: str,
    spec_name: str,
    lat_slice: tuple[int, int],
    lon_slice: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    spec = PRODUCT_SPECS[spec_name]
    i0, i1 = lat_slice
    j0, j1 = lon_slice
    with nc.Dataset(path, "r") as ds:
        lat_name, lon_name = _coord_names(ds, spec)
        src_lats = np.asarray(ds.variables[lat_name][i0:i1], dtype=np.float64)
        src_lons = np.asarray(ds.variables[lon_name][j0:j1], dtype=np.float64)
    if src_lons.size and src_lons.min() >= 0 and src_lons.max() > 180:
        src_lons = ((src_lons + 180.0) % 360.0) - 180.0
    return src_lats, src_lons


def _regrid_plane(
    plane: np.ndarray,
    src_lats: np.ndarray,
    src_lons: np.ndarray,
    dst_lats: np.ndarray,
    dst_lons: np.ndarray,
) -> np.ndarray:
    """Bilinear regrid onto the cube SST lat/lon vectors."""
    plane = np.asarray(plane, dtype=np.float32)
    dst_lats = np.asarray(dst_lats, dtype=np.float64)
    dst_lons = np.asarray(dst_lons, dtype=np.float64)
    if plane.shape == (len(dst_lats), len(dst_lons)):
        return plane
    if plane.ndim != 2 or src_lats.size == 0 or src_lons.size == 0:
        return np.full((len(dst_lats), len(dst_lons)), np.nan, dtype=np.float32)
    interp = RegularGridInterpolator(
        (src_lats, src_lons),
        plane,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )
    lon_g, lat_g = np.meshgrid(dst_lons, dst_lats)
    pts = np.column_stack([lat_g.ravel(), lon_g.ravel()])
    out = interp(pts).reshape(len(dst_lats), len(dst_lons)).astype(np.float32)
    nan_m = ~np.isfinite(out)
    if np.any(nan_m):
        nearest = RegularGridInterpolator(
            (src_lats, src_lons),
            plane,
            method="nearest",
            bounds_error=False,
            fill_value=np.nan,
        )
        flat = out.ravel()
        nan_flat = nan_m.ravel()
        flat[nan_flat] = nearest(pts[nan_flat]).astype(np.float32)
        out = flat.reshape(out.shape)
    return out


def _worker_read_daily(args: tuple[str, str, str]) -> tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    path, spec_name, date_str = args
    i0, i1, j0, j1 = _indices_for_file(path, spec_name)
    lat_slice, lon_slice = (i0, i1), (j0, j1)
    plane = _read_hyperslab_plane(path, spec_name, lat_slice, lon_slice, time_idx=0)
    src_lats, src_lons = _slice_coords(path, spec_name, lat_slice, lon_slice)
    return date_str, plane, src_lats, src_lons


def _worker_read_ssh_day(args: tuple[str, str]) -> tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    path, date_str = args
    d = date.fromisoformat(date_str)
    i0, i1, j0, j1 = _indices_for_file(path, "ssh")
    lat_slice, lon_slice = (i0, i1), (j0, j1)
    with nc.Dataset(path, "r") as ds:
        times = nc.num2date(ds.variables["time"][:], ds.variables["time"].units)
        idx = min(range(len(times)), key=lambda i: abs((_time_to_date(times[i]) - d).days))
        plane = _read_hyperslab_plane(path, "ssh", lat_slice, lon_slice, time_idx=idx)
    src_lats, src_lons = _slice_coords(path, "ssh", lat_slice, lon_slice)
    return date_str, plane, src_lats, src_lons


def _init_zarr_array(
    root: zarr.Group,
    name: str,
    spec_name: str,
    lats: np.ndarray,
    lons: np.ndarray,
    times: np.ndarray | None,
) -> zarr.Array:
    spec = PRODUCT_SPECS[spec_name]
    compressor = Blosc(cname="zlib", clevel=3, shuffle=Blosc.SHUFFLE)
    if times is None:
        shape = (len(lats), len(lons))
        chunks = (len(lats), len(lons))
        arr = root.create_dataset(
            name,
            shape=shape,
            chunks=chunks,
            dtype="f4",
            compressor=compressor,
            fill_value=np.nan,
        )
        arr.attrs.update(
            _array_attrs(spec_name, lats, lons, None),
        )
        return arr

    shape = (len(times), len(lats), len(lons))
    chunks = (min(spec.chunk_t, len(times)), len(lats), len(lons))
    arr = root.create_dataset(
        name,
        shape=shape,
        chunks=chunks,
        dtype="f4",
        compressor=compressor,
        fill_value=np.nan,
    )
    arr.attrs.update(_array_attrs(spec_name, lats, lons, times))
    return arr


def _array_attrs(spec_name: str, lats: np.ndarray, lons: np.ndarray, times: np.ndarray | None) -> dict[str, Any]:
    spec = PRODUCT_SPECS[spec_name]
    attrs: dict[str, Any] = {
        "product_id": spec.name,
        "source_var": spec.source_var,
        "native_units": spec.native_units,
        "stored_units": spec.stored_units,
        "unit_transform": spec.unit_transform or "identity",
        "grid_step_deg": spec.grid_step_deg,
        "margin_deg": margin_deg_for_product(spec.grid_step_deg),
    }
    if times is not None:
        attrs["time_units"] = "days since 2015-01-01"
    return attrs


def margin_deg_for_product(grid_step_deg: float) -> float:
    from preproc.cube.cube_schema import margin_deg_for_product as _m

    return _m(grid_step_deg)


def _open_or_create_cube(cube_path: Path, mode: str = "a") -> zarr.Group:
    cube_path.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open(str(cube_path), mode=mode)
    if "cube_schema_version" not in root.attrs:
        root.attrs["cube_schema_version"] = CUBE_SCHEMA_VERSION
        root.attrs["build_git_sha"] = _git_sha()
        root.attrs["build_date"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        root.attrs["core_lat_min"] = 18.0
        root.attrs["core_lat_max"] = 31.0
        root.attrs["core_lon_min"] = -98.0
        root.attrs["core_lon_max"] = -81.0
        root.attrs["allowed_missing_days"] = list(ALLOWED_MISSING_DAYS)
        root.attrs["missing_days"] = []
    root.attrs["data_revision"] = DATA_REVISION
    return root


def _reference_slice(spec_name: str) -> tuple[np.ndarray, np.ndarray, tuple[int, int], tuple[int, int]]:
    """Lat/lon vectors and index slices from a representative archive file."""
    ref_path = _pick_reference_file(spec_name)
    i0, i1, j0, j1 = _indices_for_file(str(ref_path), spec_name)
    spec = PRODUCT_SPECS[spec_name]
    with nc.Dataset(ref_path, "r") as ds:
        lat_name, lon_name = _coord_names(ds, spec)
        full_lat = np.asarray(ds.variables[lat_name][:], dtype=np.float64)
        full_lon = np.asarray(ds.variables[lon_name][:], dtype=np.float64)
    lons = full_lon[j0:j1]
    if lons.min() >= 0 and lons.max() > 180:
        lons = ((lons + 180.0) % 360.0) - 180.0
    return full_lat[i0:i1], lons, (i0, i1), (j0, j1)


def _ensure_coords(
    root: zarr.Group,
    spec_name: str,
    *,
    lats: np.ndarray | None = None,
    lons: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, zarr.Array | None]:
    coord_grp = root.require_group("coords")
    if lats is None or lons is None:
        if spec_name != "sst" and "sst_lat" in coord_grp:
            lats = np.asarray(coord_grp["sst_lat"])
            lons = np.asarray(coord_grp["sst_lon"])
        else:
            lats, lons, _, _ = _reference_slice(spec_name)

    lat_key = f"{spec_name}_lat"
    lon_key = f"{spec_name}_lon"
    if lat_key not in coord_grp:
        coord_grp.array(lat_key, lats.astype(np.float32))
    else:
        lats = np.asarray(coord_grp[lat_key])
    if lon_key not in coord_grp:
        coord_grp.array(lon_key, lons.astype(np.float32))
    else:
        lons = np.asarray(coord_grp[lon_key])

    if spec_name == "bathy":
        arr = root["bathy"] if "bathy" in root else _init_zarr_array(root, "bathy", "bathy", lats, lons, None)
        return lats, lons, arr

    times = time_index_axis()
    if "time" not in coord_grp:
        coord_grp.array("time", times.astype("datetime64[D]"))
    arr = root[spec_name] if spec_name in root else _init_zarr_array(root, spec_name, spec_name, lats, lons, times)
    return lats, lons, arr


def _load_manifest(path: Path) -> dict[str, Any]:
    if path.is_file():
        return json.loads(path.read_text())
    return {"products": {}, "entries": {}}


def _save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    write_json(path, manifest)


def _should_skip_date(
    manifest: dict[str, Any],
    product: str,
    date_str: str,
    force_dates: set[str],
    resume: bool,
) -> bool:
    if any(_date_matches_pattern(date_str, pat) for pat in force_dates):
        return False
    if not resume:
        return False
    key = f"{product}:{date_str}"
    return manifest.get("entries", {}).get(key, {}).get("status") == "written"


def _date_matches_pattern(date_str: str, pattern: str) -> bool:
    if pattern.endswith("*"):
        return date_str.startswith(pattern[:-1])
    return date_str == pattern


def build_product(
    spec_name: str,
    *,
    cube_path: Path,
    workers: int = 8,
    resume: bool = False,
    force_dates: set[str] | None = None,
) -> dict[str, Any]:
    force_dates = force_dates or set()
    manifest_path = default_manifest_path(cube_path)
    manifest = _load_manifest(manifest_path)

    root = _open_or_create_cube(cube_path)
    lats, lons, zarr_arr = _ensure_coords(root, spec_name)
    if zarr_arr is None:
        raise RuntimeError(f"failed to init array for {spec_name}")

    spec = PRODUCT_SPECS[spec_name]

    if spec_name == "bathy":
        blats, blons, bi, bj = _reference_slice("sst")
        lats, lons, zarr_arr = _ensure_coords(root, spec_name, lats=blats, lons=blons)
        return _build_bathy(cube_path, root, lats, lons, zarr_arr, manifest, manifest_path, bi, bj)

    tasks: list[tuple] = []
    missing: list[str] = []

    if spec.is_yearly:
        start_year = int(str(TIME_START)[:4])
        end_year = int(str(TIME_END)[:4])
        for path in _list_product_files(spec_name):
            year = int(re.search(r"(\d{4})", path.name).group(1))
            if year < start_year or year > end_year:
                continue
            try:
                with nc.Dataset(path, "r") as ds:
                    times = nc.num2date(ds.variables["time"][:], ds.variables["time"].units)
            except OSError as exc:
                print(f"WARN skip corrupt yearly archive {path}: {exc}", file=sys.stderr)
                continue
            for t in times:
                d = _time_to_date(t)
                d64 = np.datetime64(d, "D")
                if d64 < TIME_START or d64 > TIME_END:
                    continue
                ds_str = d.isoformat()
                if _should_skip_date(manifest, spec_name, ds_str, force_dates, resume):
                    continue
                tasks.append((str(path), ds_str))
    else:
        file_by_date: dict[date, Path] = {}
        for path in _list_product_files(spec_name):
            fd = _parse_file_date(spec_name, path)
            if fd is not None:
                file_by_date[fd] = path

        axis = time_index_axis()
        for t in axis:
            d = t.astype(date)
            ds_str = d.isoformat()
            if _should_skip_date(manifest, spec_name, ds_str, force_dates, resume):
                continue
            path = file_by_date.get(d)
            if path is None:
                missing.append(ds_str)
                continue
            tasks.append((str(path), spec_name, ds_str))

    worker_fn = _worker_read_ssh_day if spec.is_yearly else _worker_read_daily
    task_by_date = {task[-1]: task for task in tasks}
    written = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(worker_fn, task): task[-1] for task in tasks}
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"build {spec_name}"):
            ds_str = futures[fut]
            try:
                date_str, plane, src_lats, src_lons = fut.result()
                t_idx = time_index_of(date.fromisoformat(date_str))
                plane = _regrid_plane(plane, src_lats, src_lons, lats, lons)
                zarr_arr[t_idx, :, :] = plane
                src = task_by_date[date_str][0]
                key = f"{spec_name}:{date_str}"
                manifest.setdefault("entries", {})[key] = {
                    "status": "written",
                    "source": str(src),
                    "mtime": os.path.getmtime(str(src)),
                }
                written += 1
            except Exception as exc:
                print(f"WARN {spec_name} {ds_str}: {exc}", file=sys.stderr)

    # Explicit NaN planes for whitelisted missing days
    allowed = set(ALLOWED_MISSING_DAYS)
    for ds_str in missing:
        if ds_str not in allowed:
            continue
        t_idx = time_index_of(date.fromisoformat(ds_str))
        zarr_arr[t_idx, :, :] = np.nan
        key = f"{spec_name}:{ds_str}"
        manifest.setdefault("entries", {})[key] = {"status": "missing_whitelisted"}
        root.attrs["missing_days"] = sorted(set(root.attrs.get("missing_days", [])) | {ds_str})

    manifest.setdefault("products", {})[spec_name] = {
        "written": written,
        "missing_unwhitelisted": [d for d in missing if d not in allowed],
    }
    root.attrs["data_through"] = str(TIME_END)
    _save_manifest(manifest_path, manifest)
    return manifest["products"][spec_name]


def _build_bathy(
    cube_path: Path,
    root: zarr.Group,
    lats: np.ndarray,
    lons: np.ndarray,
    zarr_arr: zarr.Array,
    manifest: dict[str, Any],
    manifest_path: Path,
    lat_slice: tuple[int, int],
    lon_slice: tuple[int, int],
) -> dict[str, Any]:
    """Block-average GEBCO to SST grid."""
    spec = PRODUCT_SPECS["bathy"]
    root_dir = ROOT_DIRS[spec.root_key]
    path = sorted(root_dir.glob("*.nc"))[0]
    i0, i1 = lat_slice
    j0, j1 = lon_slice

    with nc.Dataset(path, "r") as ds:
        full_lat = np.asarray(ds.variables["lat"][:], dtype=np.float64)
        gebco_step = (full_lat[-1] - full_lat[0]) / max(1, (len(full_lat) - 1))
        elev = ds.variables["elevation"]
        elev.set_auto_maskandscale(False)
        raw = _decode_array(elev[i0:i1, j0:j1], elev)
    depth = spec.transform(raw)

    step = spec.grid_step_deg
    factor = max(1, int(round(step / gebco_step)))
    ny, nx = depth.shape
    ny2 = (ny // factor) * factor
    nx2 = (nx // factor) * factor
    if ny2 > 0 and nx2 > 0:
        block = depth[:ny2, :nx2].reshape(ny2 // factor, factor, nx2 // factor, factor)
        plane = np.nanmean(block, axis=(1, 3)).astype(np.float32)
    else:
        plane = depth.astype(np.float32)

    if plane.shape != (len(lats), len(lons)):
        from scipy.ndimage import zoom

        zy = len(lats) / plane.shape[0]
        zx = len(lons) / plane.shape[1]
        plane = zoom(plane, (zy, zx), order=1).astype(np.float32)

    zarr_arr[:, :] = plane
    manifest.setdefault("products", {})["bathy"] = {"written": 1, "source": str(path)}
    _save_manifest(manifest_path, manifest)
    return manifest["products"]["bathy"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build GoM regional Zarr cube")
    parser.add_argument("--product", default="all", choices=["ostia", "sss", "ssh", "bathy", "sst", "all"])
    parser.add_argument("--cube-path", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-dates", nargs="*", default=[])
    parser.add_argument("--validate", action="store_true", help="Run validation only")
    args = parser.parse_args(argv)

    cube_path = args.cube_path or default_cube_path(_ROOT)
    force_dates = set(args.force_dates or [])

    if args.validate:
        from preproc.cube.validate_cube import run_validation

        return run_validation(cube_path)

    product_map = {"ostia": "sst", "sst": "sst", "sss": "sss", "ssh": "ssh", "bathy": "bathy"}
    if args.product == "all":
        products = ["sst", "sss", "ssh", "bathy"]
    else:
        products = [product_map[args.product]]

    all_missing_unwhitelisted: list[tuple[str, list[str]]] = []
    for prod in products:
        print(f"Building {prod} -> {cube_path}")
        info = build_product(
            prod,
            cube_path=cube_path,
            workers=args.workers,
            resume=args.resume,
            force_dates=force_dates,
        )
        print(info)
        missing = info.get("missing_unwhitelisted", [])
        if missing:
            all_missing_unwhitelisted.append((prod, missing))

    if all_missing_unwhitelisted:
        for prod, days in all_missing_unwhitelisted:
            print(
                f"ERROR {prod}: {len(days)} missing day(s) not in ALLOWED_MISSING_DAYS: "
                f"{days[:5]}{'...' if len(days) > 5 else ''}",
                file=sys.stderr,
            )
        return 1

    from preproc.cube.validate_cube import run_validation

    return run_validation(cube_path)


if __name__ == "__main__":
    raise SystemExit(main())
