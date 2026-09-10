"""Synthetic step profiles must score a displaced front as a depth shift, not extra mix RMSE."""

from __future__ import annotations

import numpy as np
import pandas as pd

from evalphys.cast_table import woa_feature_table
from evalphys.profile_features import peak_dtdz, shape_features, smooth_temperature


def _step_temperature(depth: np.ndarray, z_front: float, t_warm: float = 25.0, t_cold: float = 15.0) -> np.ndarray:
    t = np.where(depth < z_front, t_warm, t_cold).astype(np.float64)
    return t[None, :]


def test_displaced_step_shifts_20c_and_peak_depth():
    depth = np.concatenate(
        [np.arange(0.0, 201.0, 5.0), np.array([225.0, 250.0, 300.0, 400.0])]
    )
    sharp = _step_temperature(depth, 100.0)
    displaced = _step_temperature(depth, 140.0)
    feat_s = shape_features(depth, sharp)
    feat_d = shape_features(depth, displaced)
    dz20 = float(feat_d["z20_m"][0] - feat_s["z20_m"][0])
    dz_peak = float(feat_d["peak_dtdz_z_m"][0] - feat_s["peak_dtdz_z_m"][0])
    assert 35.0 < dz20 < 45.0
    assert 35.0 < dz_peak < 45.0
    assert feat_s["peak_dtdz_c_per_m"][0] > 0.05
    rel = abs(feat_d["peak_dtdz_c_per_m"][0] - feat_s["peak_dtdz_c_per_m"][0]) / feat_s["peak_dtdz_c_per_m"][0]
    assert rel < 0.25


def test_two_front_mix_weakens_peak_gradient():
    depth = np.arange(0.0, 401.0, 5.0)
    sharp = _step_temperature(depth, 100.0)
    mix = 0.5 * (_step_temperature(depth, 80.0) + _step_temperature(depth, 160.0))
    peak_s, _ = peak_dtdz(*smooth_temperature(depth, sharp))
    peak_m, _ = peak_dtdz(*smooth_temperature(depth, mix))
    assert peak_s[0] > peak_m[0] * 1.4


def test_smoother_name_is_stable():
    from evalphys.profile_features import SMOOTHER_NAME, RULE_VERSION

    assert SMOOTHER_NAME == "interp1m_savgol_w11_p2"
    assert RULE_VERSION == "shape-v1"


def test_woa_long_table_recovers_step_z20():
    depth = np.arange(0.0, 201.0, 5.0)
    t = np.where(depth < 100.0, 25.0, 15.0)
    long = pd.DataFrame(
        {
            "cast_id": np.zeros(depth.size, dtype=np.int32),
            "z": depth,
            "t_woa": t,
        }
    )
    feat = woa_feature_table(long)
    assert abs(float(feat["woa_z20_m"].iloc[0]) - 100.0) < 8.0


def test_align_product_by_cast_id_leaves_gaps():
    from evalphys.cast_table import align_product_by_cast_id

    z = np.arange(3.0)
    ids = np.array([1, 3])
    t = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    out = align_product_by_cast_id(5, z, ids, t, z)
    assert np.isnan(out[0]).all()
    assert np.allclose(out[1], [1.0, 2.0, 3.0])
    assert np.allclose(out[3], [4.0, 5.0, 6.0])
