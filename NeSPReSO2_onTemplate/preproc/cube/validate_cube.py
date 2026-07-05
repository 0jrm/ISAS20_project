#!/usr/bin/env python3
"""Cube validation checks A-V1..A-V5 (Component A)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import zarr

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from preproc.cube.cube_schema import (  # noqa: E402
    ALLOWED_MISSING_DAYS,
    CUBE_SCHEMA_VERSION,
    PHYSICAL_BOUNDS,
    PRODUCT_SPECS,
    TIME_END,
    TIME_START,
    default_cube_path,
    default_manifest_path,
    default_validation_report_path,
    write_json,
    _json_safe,
)

ANTI_STALE_RUN = 3
NAN_FRAC_THRESHOLD = 0.35
OUT_OF_RANGE_FRAC = 0.001


def _check_contiguity(root: zarr.Group) -> dict[str, Any]:
    times = np.asarray(root["coords/time"])
    expected = np.arange(TIME_START, TIME_END + np.timedelta64(1, "D"), dtype="datetime64[D]")
    ok = times.shape == expected.shape and np.all(times == expected)
    missing = root.attrs.get("missing_days", [])
    allowed = set(ALLOWED_MISSING_DAYS)
    unallowed = [d for d in missing if d not in allowed]
    return {
        "id": "A-V1",
        "pass": bool(ok and not unallowed),
        "n_days": int(times.shape[0]),
        "expected_n": int(expected.shape[0]),
        "unallowed_missing": unallowed,
    }


def _check_endpoint_coverage(root: zarr.Group, *, manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    end_idx = -1
    end_day = str(np.asarray(root["coords/time"][end_idx]))
    allowed = set(ALLOWED_MISSING_DAYS)
    entries = (manifest or {}).get("entries", {})
    per_product: dict[str, bool] = {}
    for name in ("sst", "sss", "ssh"):
        if name not in root:
            per_product[name] = False
            continue
        plane = np.asarray(root[name][end_idx])
        finite_frac = float(np.isfinite(plane).mean())
        if finite_frac > 0.05:
            per_product[name] = True
            continue
        key = f"{name}:{end_day}"
        whitelisted = (
            end_day in allowed
            and entries.get(key, {}).get("status") == "missing_whitelisted"
        )
        per_product[name] = bool(whitelisted)
    ok = all(per_product.values())
    return {"id": "A-V2", "pass": ok, "per_product": per_product, "endpoint_day": end_day}


def _check_anti_stale(root: zarr.Group, *, max_planes: int = 400) -> dict[str, Any]:
    allowed = set(ALLOWED_MISSING_DAYS)
    failures: list[dict[str, Any]] = []
    for name in ("sst", "sss", "ssh"):
        if name not in root:
            continue
        arr = root[name]
        n_t = min(arr.shape[0], max_planes)
        run = 0
        prev: np.ndarray | None = None
        for t in range(n_t):
            day = str(np.asarray(root["coords/time"][t]))
            if day in allowed:
                run = 0
                prev = None
                continue
            plane = np.asarray(arr[t])
            if prev is not None and plane.shape == prev.shape:
                identical = np.allclose(plane, prev, equal_nan=True, rtol=0, atol=0)
                if identical:
                    run += 1
                    if run >= ANTI_STALE_RUN - 1:
                        failures.append({"product": name, "time_index": t, "day": day, "run": run + 1})
                else:
                    run = 0
            prev = plane
    return {"id": "A-V3", "pass": len(failures) == 0, "failures": failures[:20]}


def _check_physical_range(root: zarr.Group) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    ok = True
    for name, (lo, hi) in PHYSICAL_BOUNDS.items():
        if name not in root:
            continue
        if name == "bathy":
            sample = np.asarray(root[name])
        else:
            sample = np.asarray(root[name][:: max(1, root[name].shape[0] // 32)])
        finite = sample[np.isfinite(sample)]
        if finite.size == 0:
            reports[name] = {"pass": False, "reason": "no finite values"}
            ok = False
            continue
        oob = np.mean((finite < lo) | (finite > hi))
        passed = bool(oob <= OUT_OF_RANGE_FRAC)
        reports[name] = {"pass": passed, "oob_fraction": float(oob), "bounds": [lo, hi]}
        ok = ok and passed
    return {"id": "A-V4", "pass": ok, "per_channel": reports}


def _ocean_mask_on_grid(bathy: np.ndarray, ny: int, nx: int) -> np.ndarray:
    """Resample SST-grid bathy to (ny, nx); ocean pixels have depth > 0."""
    if bathy.shape == (ny, nx):
        return bathy > 0
    from scipy.ndimage import zoom

    zy = ny / bathy.shape[0]
    zx = nx / bathy.shape[1]
    resampled = zoom(bathy.astype(np.float32), (zy, zx), order=1)
    return resampled > 0


def _check_nan_budget(root: zarr.Group) -> dict[str, Any]:
    if "bathy" not in root:
        return {"id": "A-V5", "pass": False, "reason": "missing bathy"}
    bathy = np.asarray(root["bathy"])
    coord = root.get("coords")
    per_product: dict[str, Any] = {}
    ok = True
    for name in ("sst", "sss", "ssh"):
        if name not in root:
            ok = False
            continue
        lat_key, lon_key = f"{name}_lat", f"{name}_lon"
        if coord is None or lat_key not in coord or lon_key not in coord:
            per_product[name] = {"pass": False, "reason": f"missing coords/{lat_key} or coords/{lon_key}"}
            ok = False
            continue
        prod_lats = np.asarray(coord[lat_key])
        prod_lons = np.asarray(coord[lon_key])
        ny, nx = len(prod_lats), len(prod_lons)
        ocean_m = _ocean_mask_on_grid(bathy, ny, nx)
        arr = root[name]
        idx = arr.shape[0] // 2
        plane = np.asarray(arr[idx])
        if plane.ndim != 2:
            plane = np.squeeze(plane)
        if ocean_m.shape != plane.shape:
            per_product[name] = {"pass": False, "reason": f"shape mismatch {plane.shape} vs {ocean_m.shape}"}
            ok = False
            continue
        ocean_vals = plane[ocean_m]
        nan_frac = float(np.mean(~np.isfinite(ocean_vals))) if ocean_vals.size else 1.0
        passed = bool(nan_frac < NAN_FRAC_THRESHOLD)
        per_product[name] = {"nan_fraction_ocean": nan_frac, "pass": passed}
        ok = ok and passed
    return {"id": "A-V5", "pass": ok, "per_product": per_product}


def run_validation(cube_path: Path | None = None) -> int:
    cube_path = cube_path or default_cube_path(_ROOT)
    report_path = default_validation_report_path(cube_path)

    if not cube_path.exists():
        payload = {"pass": False, "error": f"cube not found: {cube_path}"}
        write_json(report_path, payload)
        print(json.dumps(payload, indent=2))
        return 1

    root = zarr.open(str(cube_path), mode="r")
    manifest_path = default_manifest_path(cube_path)
    manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
    checks = [
        _check_contiguity(root),
        _check_endpoint_coverage(root, manifest=manifest),
        _check_anti_stale(root),
        _check_physical_range(root),
        _check_nan_budget(root),
    ]
    passed = all(bool(c.get("pass", False)) for c in checks)
    payload = {
        "cube_path": str(cube_path),
        "cube_schema_version": int(root.attrs.get("cube_schema_version", CUBE_SCHEMA_VERSION)),
        "pass": bool(passed),
        "checks": checks,
    }
    write_json(report_path, payload)
    print(json.dumps(_json_safe(payload), indent=2))
    return 0 if passed else 1


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cube-path", type=Path, default=None)
    args = parser.parse_args(argv)
    return run_validation(args.cube_path)


if __name__ == "__main__":
    raise SystemExit(main())
