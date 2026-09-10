from __future__ import annotations

import numpy as np
import pandas as pd

from evalphys.opponent_report import format_a2_report, mean_per_level_r, z20_rmse


def test_z20_rmse_is_zero_when_pred_equals_truth():
    s = pd.Series([10.0, 20.0, 30.0])
    assert z20_rmse(s, s) == 0.0


def test_identical_fields_have_unit_per_level_r():
    rng = np.random.default_rng(0)
    t = rng.normal(size=(20, 5))
    ok = np.ones_like(t, dtype=bool)
    assert abs(mean_per_level_r(t, t, ok) - 1.0) < 1e-12


def test_report_lists_four_opponents_and_is_not_a_gate():
    n, nz = 12, 6
    z = np.linspace(50.0, 180.0, nz)
    casts = pd.DataFrame(
        {
            "cast_id": np.tile(np.arange(n), 3),
            "model": np.repeat(["A_CRPS", "xb_noprofile", "persistence"], n),
            "era": "2025",
            "in_bbox": True,
            "argo_z20_m": 100.0,
            "xb_z20_m": 110.0,
            "nes_z20_m": 105.0,
        }
    )
    levels = pd.DataFrame(
        {
            "cast_id": np.tile(np.repeat(np.arange(n), nz), 3),
            "model": np.repeat(["A_CRPS", "xb_noprofile", "persistence"], n * nz),
            "z": np.tile(z, n * 3),
            "t_argo": 20.0,
            "t_xb": 19.0,
            "t_nes": 19.5,
            "valid_t": True,
        }
    )
    perm_z = pd.DataFrame(
        {
            "model": ["A_CRPS", "xb_noprofile"],
            "slice": ["in_bbox", "in_bbox"],
            "era": ["2025", "2025"],
            "z": [50.0, 50.0],
            "r_nes": [0.9, 0.8],
            "r_xb": [0.7, 0.7],
        }
    )
    woa_long = pd.DataFrame(
        {
            "cast_id": np.repeat(np.arange(n), nz),
            "z": np.tile(z, n),
            "t_woa": 18.0,
        }
    )
    woa_feat = pd.DataFrame({"cast_id": np.arange(n), "woa_z20_m": 120.0})
    text = format_a2_report(casts, levels, perm_z, woa_long, woa_feat, "cast_features_deadbeef_shape-v1")
    assert "not a go-rule" in text
    assert "A_CRPS" in text
    assert "xb_noprofile" in text
    assert "persistence" in text
    assert "WOA23" in text
    assert "gate" not in text
