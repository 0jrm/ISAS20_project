"""Tests for cube validation (A-V1..A-V5)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr
from numcodecs import Blosc

from preproc.cube.cube_schema import CUBE_SCHEMA_VERSION, TIME_END, TIME_START
from preproc.cube.validate_cube import (
    _check_anti_stale,
    _check_contiguity,
    _check_endpoint_coverage,
    _check_nan_budget,
    _check_physical_range,
)


@pytest.fixture
def synthetic_cube(tmp_path: Path) -> Path:
    cube_path = tmp_path / "gom_cube.zarr"
    times = np.arange(TIME_START, TIME_END + np.timedelta64(1, "D"), dtype="datetime64[D]")
    root = zarr.open(str(cube_path), mode="w")
    root.attrs["cube_schema_version"] = CUBE_SCHEMA_VERSION
    root.attrs["missing_days"] = []
    root.attrs["data_through"] = str(TIME_END)
    coord = root.create_group("coords")
    coord.array("time", times)
    n_t, n_y, n_x = len(times), 32, 40
    lats = np.linspace(17.0, 32.0, n_y).astype(np.float32)
    lons = np.linspace(-99.0, -80.0, n_x).astype(np.float32)
    for ch in ("sst", "sss", "ssh"):
        coord.array(f"{ch}_lat", lats)
        coord.array(f"{ch}_lon", lons)
        t_axis = np.arange(n_t, dtype=np.float32)[:, None, None]
        if ch == "sst":
            data = (20.0 + 0.5 * np.sin(t_axis * 0.01) + 0.01 * lats[None, :, None]).astype(np.float32)
            data = np.broadcast_to(data, (n_t, n_y, n_x)).copy()
        elif ch == "sss":
            data = (35.0 + 0.1 * np.sin(t_axis * 0.02)).astype(np.float32)
            data = np.broadcast_to(data, (n_t, n_y, n_x)).copy()
        else:
            data = (0.05 * np.sin(t_axis * 0.01) + 0.02 * lons[None, None, :] * 0.001).astype(np.float32)
            data = np.broadcast_to(data, (n_t, n_y, n_x)).copy()
        root.create_dataset(
            ch,
            data=data,
            chunks=(64, n_y, n_x),
            compressor=Blosc(cname="zlib", clevel=1),
        )
    bathy = np.full((n_y, n_x), 100.0, dtype=np.float32)
    root.create_dataset("bathy", data=bathy, chunks=(n_y, n_x))
    return cube_path


def test_av1_contiguity(synthetic_cube: Path):
    root = zarr.open(str(synthetic_cube), mode="r")
    assert _check_contiguity(root)["pass"]


def test_av2_endpoint(synthetic_cube: Path):
    root = zarr.open(str(synthetic_cube), mode="r")
    assert _check_endpoint_coverage(root)["pass"]


def test_av3_anti_stale(synthetic_cube: Path):
    root = zarr.open(str(synthetic_cube), mode="r")
    assert _check_anti_stale(root, max_planes=50)["pass"]


def test_av4_physical_range(synthetic_cube: Path):
    root = zarr.open(str(synthetic_cube), mode="r")
    assert _check_physical_range(root)["pass"]


def test_av5_nan_budget(synthetic_cube: Path):
    root = zarr.open(str(synthetic_cube), mode="r")
    assert _check_nan_budget(root)["pass"]
