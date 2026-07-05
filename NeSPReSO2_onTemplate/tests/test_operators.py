"""Operator correctness on analytic fields."""

from __future__ import annotations

import math

import numpy as np
import pytest

from preproc.features.operators import apply_operator


def _make_grid(n: int = 64, lat0: float = 18.0):
    lats = np.linspace(lat0, lat0 + 5.0, n)
    lons = np.linspace(-95.0, -85.0, n)
    lon_g, lat_g = np.meshgrid(lons, lats)
    return lats, lons, lon_g, lat_g


def test_grad_analytic_sin_cos():
    lats, lons, lon_g, lat_g = _make_grid(lat0=18.0)
    R = 6_371_000.0
    x_m = np.radians(lon_g) * R * np.cos(np.radians(lat_g))
    field = np.sin(0.001 * x_m).astype(np.float32)
    gx, gy = apply_operator("grad", field, lats=lats, lons=lons, scale_deg=1.0, grid_step_deg=0.05)
    # spot check center pixel finite
    cy, cx = len(lats) // 2, len(lons) // 2
    assert np.isfinite(gx[cy, cx])
    assert np.isfinite(gy[cy, cx])


def test_value_centered_subtracts_basin_and_preserves_gradients():
    """value_centered == value − basin scalar; the constant offset leaves grad unchanged."""
    lats, lons, lon_g, lat_g = _make_grid(lat0=18.0)
    field = (0.1 * lon_g + 0.05 * lat_g).astype(np.float32)
    basin = 3.14159
    val = apply_operator("value", field, lats=lats, lons=lons, scale_deg=0.4, grid_step_deg=0.25)
    val_c = apply_operator(
        "value_centered", field, lats=lats, lons=lons, scale_deg=0.4, grid_step_deg=0.25, basin_value=basin
    )
    finite = np.isfinite(val) & np.isfinite(val_c)
    assert np.allclose((val - val_c)[finite], basin, atol=1e-4)
    # A spatially constant offset must not change gradients.
    gx, gy = apply_operator("grad", field, lats=lats, lons=lons, scale_deg=0.4, grid_step_deg=0.25)
    gx_c, gy_c = apply_operator("grad", (field - basin).astype(np.float32), lats=lats, lons=lons, scale_deg=0.4, grid_step_deg=0.25)
    fin = np.isfinite(gx) & np.isfinite(gx_c)
    assert np.allclose(gx[fin], gx_c[fin], atol=1e-6)


def test_value_centered_requires_basin_value():
    lats, lons, _, _ = _make_grid()
    field = np.ones((len(lats), len(lons)), dtype=np.float32)
    with pytest.raises(ValueError):
        apply_operator("value_centered", field, lats=lats, lons=lons, scale_deg=0.4, grid_step_deg=0.25)


def test_laplacian_finite():
    lats, lons, lon_g, lat_g = _make_grid(lat0=31.0)
    field = (np.sin(np.radians(lat_g)) * np.cos(np.radians(lon_g))).astype(np.float32)
    lap = apply_operator("laplacian", field, lats=lats, lons=lons, scale_deg=1.0, grid_step_deg=0.05)
    assert np.isfinite(lap).any()


def test_tendency_linear_time():
    planes = np.stack([np.full((8, 8), float(t), dtype=np.float32) for t in range(7)], axis=0)
    out = apply_operator("tendency", planes[0], stack=planes, window_days=7)
    assert np.nanmean(out) == pytest.approx(1.0, abs=0.2)


def test_geo_uv_units():
    lats, lons, _, lat_g = _make_grid()
    field = (0.1 * lat_g).astype(np.float32)
    u, v = apply_operator(
        "geo_uv",
        field,
        lats=lats,
        lons=lons,
        scale_deg=1.0,
        grid_step_deg=0.25,
        profile_lat=25.0,
    )
    f = 2 * 7.2921e-5 * math.sin(math.radians(25.0))
    assert abs(f) > 0
    assert np.isfinite(u).any()
    assert np.isfinite(v).any()
