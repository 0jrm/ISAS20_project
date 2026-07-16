#!/usr/bin/env python3
"""Unit checks for T2 stale-patch detector (synthetic + H5 key presence)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from diagnostics.stale_sat.split_vs_stale import DEFAULT_H5, _stale_mask


def test_inject_time_constant_patch():
    rng = np.random.default_rng(0)
    arr = rng.normal(size=(40, 7, 5, 5))
    assert _stale_mask(arr).sum() == 0
    arr[5:15] = arr[5:15, :1]  # freeze time axis
    m = _stale_mask(arr)
    assert m[5:15].all()
    assert not m[0]


def test_nan_does_not_invent_constancy():
    rng = np.random.default_rng(1)
    arr = rng.normal(size=(10, 7, 5, 5))
    arr[0, 2, 1, 1] = np.nan
    assert not _stale_mask(arr)[0], "nan_to_num must not turn varying data into time-constant"


def test_all_nan_is_flagged():
    arr = np.full((4, 7, 5, 5), np.nan)
    assert _stale_mask(arr).all()


def test_h5_variable_keys_exist():
    import h5py

    if not Path(DEFAULT_H5).is_file():
        return
    required = ("ostia/analysed_sst", "ssh/adt", "sss/sos", "stations/julian_date")
    with h5py.File(DEFAULT_H5, "r") as f:
        for key in required:
            assert key in f, f"missing H5 key {key}"
            assert f[key].shape[0] > 0


if __name__ == "__main__":
    test_inject_time_constant_patch()
    test_nan_does_not_invent_constancy()
    test_all_nan_is_flagged()
    test_h5_variable_keys_exist()
    print("stale detector checks OK")
