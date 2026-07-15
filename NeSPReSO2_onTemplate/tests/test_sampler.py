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


def test_weights_for_keys_on_points_not_just_channel(tmp_path: Path):
    """A provider reused across two profile sets must not hand back the first set's weights.

    Regression: the cache keyed on ``channel`` alone, so the second call silently sampled the
    first call's locations. Harmless in the bench (fresh provider per run) — a wrong-results bug
    anywhere a provider is reused.
    """
    cube_path = _make_missing_day_cube(tmp_path)
    provider = CubeProvider(cube_path)

    lats_a, lons_a = np.array([21.0]), np.array([-94.0])
    lats_b, lons_b = np.array([24.0]), np.array([-91.0])

    w_a = provider.weights_for("sst", lats_a, lons_a)
    w_b = provider.weights_for("sst", lats_b, lons_b)
    w_b_fresh = CubeProvider(cube_path).weights_for("sst", lats_b, lons_b)

    assert (w_b - w_b_fresh).nnz == 0, "reused provider returned stale weights for new points"
    assert (w_b - w_a).nnz != 0, "distinct points collapsed onto one cache entry"
    # And the cache must still hit when the same points come back.
    assert (provider.weights_for("sst", lats_a, lons_a) - w_a).nnz == 0


def _make_linear_cube(tmp_path: Path, a: float = 2.0, b: float = 3.0, c: float = 10.0) -> Path:
    """Cube whose channels hold an exact linear field ``c + a*lat + b*lon``."""
    cube_path = tmp_path / "linear_cube.zarr"
    times = np.array(["2015-01-01", "2015-01-02", "2015-01-03"], dtype="datetime64[D]")
    root = zarr.open(str(cube_path), mode="w")
    root.attrs["missing_days"] = []
    coord = root.create_group("coords")
    coord.array("time", times)
    lats = np.arange(18.0, 32.0 + 1e-9, 0.05).astype(np.float32)
    lons = np.arange(-99.0, -80.0 + 1e-9, 0.05).astype(np.float32)
    lon_g, lat_g = np.meshgrid(lons.astype(np.float64), lats.astype(np.float64))
    field = (c + a * lat_g + b * lon_g).astype(np.float32)
    for ch in ("sst", "sss", "ssh"):
        coord.array(f"{ch}_lat", lats)
        coord.array(f"{ch}_lon", lons)
        root.create_dataset(
            ch,
            data=np.repeat(field[None, :, :], len(times), axis=0),
            chunks=(1, len(lats), len(lons)),
            compressor=Blosc(cname="zlib", clevel=1),
        )
    root.create_dataset("bathy", data=field.copy())
    return cube_path


def test_sample_end_to_end_on_linear_field(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """End-to-end ``sample()`` against an analytically known field.

    The golden ``.npz`` was the *only* thing checking ``sample()`` end-to-end, and it silently
    encoded a double-decoded cube (Gulf of Mexico SST at 3 degC) for ten days without any test
    noticing. This asserts against mathematics instead of a recorded artifact: Gaussian smoothing
    preserves a linear field exactly, so ``value@local`` must return the field itself, and the
    gradient of a linear field is constant everywhere.
    """
    monkeypatch.setattr(cube_schema, "ALLOWED_MISSING_DAYS", [])
    a, b, c = 2.0, 3.0, 10.0
    cube_path = _make_linear_cube(tmp_path, a=a, b=b, c=c)
    provider = CubeProvider(cube_path)

    # Well inside the domain so the smoothing kernel never touches a boundary.
    lats = np.array([24.0, 25.5, 26.25])
    lons = np.array([-92.0, -90.5, -88.25])
    d = date(2015, 1, 2)
    dates_jd = np.full(lats.shape, Time(datetime.combine(d, datetime.min.time())).jd)

    spec = {
        "scalars": [],
        "operators": [
            {"op": "value", "channels": ["sst"], "scales": ["local"]},
            {"op": "grad", "channels": ["sst"], "scales": ["1.0deg"]},
        ],
    }
    table = provider.sample(spec, lats, lons, dates_jd)
    j = {n: i for i, n in enumerate(table.names)}

    assert table.valid_mask.all(), "linear field with no NaNs must be fully valid"

    got = table.values[:, j["sst.value@local"]]
    expected = c + a * lats + b * lons
    np.testing.assert_allclose(got, expected, rtol=1e-3, atol=1e-2)

    # Gradient of a linear field is the same at every point, whatever the unit convention.
    for comp in ("sst.grad_x@1.0deg", "sst.grad_y@1.0deg"):
        g = table.values[:, j[comp]]
        assert np.all(np.isfinite(g))
        assert np.std(g) < 1e-4 * max(1.0, abs(float(np.mean(g)))), f"{comp} not constant on a linear field"


def test_golden_files_are_physically_plausible():
    """The golden is the correctness gate — so the gate itself needs a sanity check.

    A stale golden built from a double-decoded cube asserted Gulf of Mexico SST of ~3 degC and
    would have rejected every *correct* candidate while looking like a working safety net.
    Values here are deliberately loose bounds: this catches decode/unit regressions, not physics.
    """
    golden_dir = Path(__file__).resolve().parents[1] / "tests" / "golden"
    files = sorted(golden_dir.glob("sampler_golden_*.npz"))
    if not files:
        pytest.skip("no golden files present")

    # (feature, lo, hi) — deliberately generous envelopes for the Gulf of Mexico. The bug this
    # guards against scaled values by ~9x (27 degC -> 3 degC), so loose bounds still catch it
    # while leaving room for real extremes: SSS runs low in the Mississippi plume (<25 PSU).
    bounds = {
        "sst.value@local": (5.0, 40.0),      # degC
        "sss.value@local": (15.0, 40.0),     # PSU — river plumes go well below open-Gulf values
        "ssh.value@local": (-2.0, 2.0),      # m
    }
    for f in files:
        d = np.load(f, allow_pickle=False)
        names = list(d["names"])
        for feat, (lo, hi) in bounds.items():
            if feat not in names:
                continue
            v = d["values"][:, names.index(feat)]
            v = v[np.isfinite(v)]
            if v.size == 0:
                continue
            assert lo <= float(np.min(v)) and float(np.max(v)) <= hi, (
                f"{f.name}: {feat} spans {np.min(v):.2f}..{np.max(v):.2f}, outside "
                f"[{lo}, {hi}] — golden likely built from a mis-decoded cube"
            )
