"""Engineered patch feature tests (Phase 5)."""

from __future__ import annotations

import numpy as np

from preproc.preproc_isas_sat import (
    build_argo_residual_input_matrix,
    compute_argo_residual_input_dim,
    count_engineered_patch_channels,
)


def _synthetic_sat_vars(n: int, spatial_pad: int, temporal_pad: int) -> dict:
    t_win = temporal_pad + 1
    h = w = 2 * spatial_pad + 1
    shape = (n, t_win * h * w)
    rng = np.random.default_rng(3)
    return {
        "sss": rng.normal(size=shape).astype(np.float32),
        "sst": (rng.normal(size=shape) + 273.15).astype(np.float32),
        "ssh": rng.normal(size=shape).astype(np.float32) * 0.1,
    }


def test_phase5_engineered_channels_increase_dim():
    base_params = {
        "timecos": True,
        "timesin": True,
        "latcos": True,
        "latsin": True,
        "loncos": True,
        "lonsin": True,
        "center_sss": True,
        "center_sst": True,
        "center_ssh": True,
        "sss": True,
        "sst": True,
        "ssh": True,
    }
    eng_params = dict(
        base_params,
        patch_ssh_gradient=True,
        patch_ssh_laplacian=True,
        patch_temporal_tendency=True,
        patch_sst_gradient=True,
    )
    assert count_engineered_patch_channels(eng_params) == 8
    d0 = compute_argo_residual_input_dim(base_params, 2, 3)
    d1 = compute_argo_residual_input_dim(eng_params, 2, 3)
    assert d1 > d0


def test_phase5_builder_with_engineered_features():
    n = 4
    spatial_pad = 2
    temporal_pad = 3
    input_params = {
        "timecos": True,
        "timesin": True,
        "latcos": True,
        "latsin": True,
        "loncos": True,
        "lonsin": True,
        "center_sss": True,
        "center_sst": True,
        "center_ssh": True,
        "sss": True,
        "sst": True,
        "ssh": True,
        "patch_ssh_gradient": True,
        "patch_ssh_laplacian": True,
    }
    sat_vars = _synthetic_sat_vars(n, spatial_pad, temporal_pad)
    juld = np.linspace(0, 364, n)
    lat = np.linspace(18, 30, n)
    lon = np.linspace(-95, -82, n)
    inputs, _ = build_argo_residual_input_matrix(
        juld, lat, lon, sat_vars, input_params, spatial_pad=spatial_pad, temporal_pad=temporal_pad
    )
    expected = compute_argo_residual_input_dim(input_params, spatial_pad, temporal_pad)
    assert inputs.shape == (n, expected)
