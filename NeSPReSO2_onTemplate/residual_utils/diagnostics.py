"""Lightweight normalization diagnostics."""

from __future__ import annotations

import numpy as np


def normalization_report(cache: dict) -> dict:
    std_block = cache.get("input_standardization") or {}
    inputs = np.asarray(cache["inputs"], dtype=np.float32)
    tr = np.asarray(std_block.get("train_indices", []), dtype=int)
    if tr.size == 0:
        from base.split_utils import build_split_indices

        dl_cfg = {"split_mode": "chronological", "train_frac": 0.7, "val_frac": 0.15, "test_frac": 0.15}
        splits = build_split_indices(
            inputs.shape[0],
            cache.get("JULD"),
            dl_cfg,
            dataset_tag=cache.get("dataset_tag", "argo_residual"),
        )
        tr = np.asarray(splits["train"], dtype=int)
    col_std = inputs[tr].std(axis=0)
    bad_low = np.where(col_std < 0.05)[0].tolist()
    bad_high = np.where(col_std > 5.0)[0].tolist()
    return {
        "train_col_std_min": float(col_std.min()),
        "train_col_std_max": float(col_std.max()),
        "bad_low_std_cols": bad_low,
        "bad_high_std_cols": bad_high,
        "normalization_version": std_block.get("normalization_version"),
    }
