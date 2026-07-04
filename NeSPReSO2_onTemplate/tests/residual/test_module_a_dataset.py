"""Module A tests: dataset builder / standardization."""

from __future__ import annotations

import numpy as np

from preproc.preproc_isas_sat import (
    _make_center_relative,
    _reshape_sat_patch_block,
    apply_train_standardization,
    build_argo_residual_input_matrix,
)


def _synthetic_sat_vars(n: int, spatial_pad: int, temporal_pad: int) -> dict:
    t_win = temporal_pad + 1
    h = w = 2 * spatial_pad + 1
    shape = (n, t_win * h * w)
    rng = np.random.default_rng(0)
    return {
        "sss": rng.normal(size=shape).astype(np.float32),
        "sst": (rng.normal(size=shape) + 273.15).astype(np.float32),
        "ssh": rng.normal(size=shape).astype(np.float32) * 0.1,
    }


def test_a1_train_split_zscore_stats():
    n = 200
    inputs = np.random.default_rng(1).normal(loc=5.0, scale=2.0, size=(n, 12)).astype(np.float32)
    juld = np.arange(n, dtype=np.float32) + 730000.0
    config = {
        "data_loader": {
            "args": {
                "split_mode": "chronological",
                "train_frac": 0.7,
                "val_frac": 0.15,
                "test_frac": 0.15,
            }
        }
    }
    standardized, meta = apply_train_standardization(
        inputs, juld, config, feature_names=[f"f{i}" for i in range(12)], dataset_tag="isas20"
    )
    tr = meta["train_indices"]
    mu = standardized[tr].mean(axis=0)
    sd = standardized[tr].std(axis=0)
    assert np.all(np.abs(mu) < 0.02)
    assert np.all(sd > 0.98) and np.all(sd < 1.02)


def test_a2_no_train_test_leakage_in_stats():
    n = 100
    inputs = np.random.default_rng(2).normal(size=(n, 5)).astype(np.float32)
    juld = np.arange(n, dtype=np.float32) + 730000.0
    config = {"data_loader": {"args": {"split_mode": "chronological", "train_frac": 0.7, "val_frac": 0.15, "test_frac": 0.15}}}
    _, meta = apply_train_standardization(inputs, juld, config, dataset_tag="isas20")
    tr = set(meta["train_indices"].tolist())
    mu = meta["mean"]
    manual_mu = inputs[list(tr)].mean(axis=0)
    np.testing.assert_allclose(mu, manual_mu, rtol=1e-5, atol=1e-5)


def test_a3_center_relative_center_is_zero():
    n = 4
    spatial_pad = 2
    temporal_pad = 3
    flat = np.arange(n * 5 * 5 * 4, dtype=np.float32).reshape(n, -1) + 10.0
    vol = _reshape_sat_patch_block(flat, spatial_pad, temporal_pad)
    rel = _make_center_relative(vol, spatial_pad)
    cy = cx = spatial_pad
    center_vals = rel[:, -1, cy, cx]
    np.testing.assert_array_equal(center_vals, np.zeros(n, dtype=np.float32))


def test_a3_builder_center_pixel_zero_in_patch_block():
    n = 6
    spatial_pad = 2
    temporal_pad = 3
    sat_vars = _synthetic_sat_vars(n, spatial_pad, temporal_pad)
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
    }
    juld = np.linspace(0, 364, n)
    lat = np.linspace(18, 30, n)
    lon = np.linspace(-95, -82, n)
    inputs, layout = build_argo_residual_input_matrix(
        juld, lat, lon, sat_vars, input_params, spatial_pad=spatial_pad, temporal_pad=temporal_pad
    )
    from preproc.preproc_isas_sat import sat_patch_center_index

    center_idx = sat_patch_center_index(spatial_pad, temporal_pad)
    patch_offset = layout["patch_offset"]
    # center of first patch variable block (sss)
    col = patch_offset + center_idx
    np.testing.assert_allclose(inputs[:, col], 0.0, atol=1e-6)
