"""Sampler bilinear weights and time_index_of tests."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import numpy as np
import pytest
import zarr
from astropy.time import Time
from numcodecs import Blosc

from preproc.cube import cube_schema
from preproc.cube.cube_schema import time_index_of
from preproc.features.sampler import (
    CubeProvider,
    MissingCubePlaneError,
    build_bilinear_weights,
    sample_plane,
    time_index_of_jd,
)


def test_bilinear_exact_plane():
    lats = np.linspace(20.0, 30.0, 11)
    lons = np.linspace(-95.0, -85.0, 11)
    lon_g, lat_g = np.meshgrid(lons, lats)
    plane = (2.0 * lat_g + 3.0 * lon_g).astype(np.float32)
    sample_lats = np.array([25.0])
    sample_lons = np.array([-90.0])
    w = build_bilinear_weights(lats, lons, sample_lats, sample_lons)
    val, valid = sample_plane(w, plane)
    expected = 2.0 * 25.0 + 3.0 * (-90.0)
    assert valid[0]
    assert val[0] == pytest.approx(expected, rel=1e-3, abs=0.05)


@pytest.mark.parametrize(
    "iso,expected",
    [
        ("2015-01-01", 0),
        ("2022-03-01", time_index_of(date(2022, 3, 1))),
    ],
)
def test_time_index_of_known(iso: str, expected: int):
    assert time_index_of(date.fromisoformat(iso)) == expected


def test_time_index_of_jd_roundtrip():
    from datetime import datetime

    d = date(2019, 6, 15)
    jd = Time(datetime.combine(d, datetime.min.time())).jd
    assert time_index_of_jd(jd) == time_index_of(d)


def _make_missing_day_cube(tmp_path: Path) -> Path:
    cube_path = tmp_path / "missing_day_cube.zarr"
    times = np.array(["2015-01-01", "2015-01-02", "2015-01-03"], dtype="datetime64[D]")
    root = zarr.open(str(cube_path), mode="w")
    root.attrs["missing_days"] = ["2015-01-02"]
    coord = root.create_group("coords")
    coord.array("time", times)
    n_t, n_y, n_x = 3, 4, 5
    lats = np.linspace(20.0, 25.0, n_y).astype(np.float32)
    lons = np.linspace(-95.0, -90.0, n_x).astype(np.float32)
    for ch in ("sst", "sss", "ssh"):
        coord.array(f"{ch}_lat", lats)
        coord.array(f"{ch}_lon", lons)
        data = np.full((n_t, n_y, n_x), 20.0, dtype=np.float32)
        if ch == "sst":
            data[1] = np.nan
        root.create_dataset(
            ch,
            data=data,
            chunks=(1, n_y, n_x),
            compressor=Blosc(cname="zlib", clevel=1),
        )
    root.create_dataset("bathy", data=np.full((n_y, n_x), 100.0, dtype=np.float32))
    return cube_path


def test_plane_raises_on_whitelisted_missing_day(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(cube_schema, "ALLOWED_MISSING_DAYS", ["2015-01-02"])
    cube_path = _make_missing_day_cube(tmp_path)
    provider = CubeProvider(cube_path)
    with pytest.raises(MissingCubePlaneError, match="2015-01-02"):
        provider.plane("sst", 1)


def test_tendency_raises_when_window_includes_missing_day(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(cube_schema, "ALLOWED_MISSING_DAYS", ["2015-01-02"])
    cube_path = _make_missing_day_cube(tmp_path)
    provider = CubeProvider(cube_path)
    feature_spec = {
        "operators": [{"op": "tendency", "channels": ["sst"], "window_days": 3}],
    }
    d = date(2015, 1, 3)
    jd = Time(datetime.combine(d, datetime.min.time())).jd
    lats = np.array([22.5])
    lons = np.array([-92.5])
    dates_jd = np.array([jd])
    with pytest.raises(MissingCubePlaneError, match="2015-01-02"):
        provider.sample(feature_spec, lats, lons, dates_jd)
