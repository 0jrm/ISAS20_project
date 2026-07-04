"""Module G tests: diagnostics."""

from __future__ import annotations

import numpy as np

from residual_utils.diagnostics import normalization_report as _normalization_report


def test_g1_normalization_report_flags():
    n, d = 100, 8
    cache = {
        "inputs": np.random.default_rng(0).normal(size=(n, d)).astype(np.float32),
        "JULD": np.arange(n, dtype=np.float32),
        "dataset_tag": "argo_residual",
        "input_standardization": {
            "train_indices": np.arange(int(0.7 * n)),
            "normalization_version": "train_zscore_v1",
        },
    }
    report = _normalization_report(cache)
    assert report["train_col_std_min"] > 0.05
    assert report["train_col_std_max"] < 5.0
    assert report["bad_low_std_cols"] == []
    assert report["bad_high_std_cols"] == []


def test_g2_residual_utilization_threshold(residual_model, synthetic_batch):
    import torch

    residual_model.eval()
    residual_model.patch_head_out.weight.data.normal_(0, 0.02)
    residual_model.gate.data.fill_(1.0)
    with torch.no_grad():
        out = residual_model(synthetic_batch)
        base = residual_model.forward_base(synthetic_batch)
        contrib = (out - base).norm(dim=1).mean().item()
    assert contrib > 1e-6


def test_g3_ablation_branch_dependence(residual_model, synthetic_batch):
    import torch

    residual_model.eval()
    with torch.no_grad():
        full = residual_model(synthetic_batch)
        base_only = residual_model.forward_base(synthetic_batch)
    assert torch.allclose(full, base_only, atol=1e-7)
    residual_model.patch_head_out.weight.data.normal_(0, 0.02)
    residual_model.gate.data.fill_(0.5)
    with torch.no_grad():
        full2 = residual_model(synthetic_batch)
    assert (full2 - base_only).abs().mean().item() > 0.0
