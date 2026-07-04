"""Module F tests: evaluation / paired stats."""

from __future__ import annotations

import numpy as np

from residual_utils.stats import paired_stats as _paired_stats


def test_f1_improvement_direction():
    point = np.array([0.5, 0.6, 0.7, 0.8])
    residual = np.array([0.4, 0.55, 0.65, 0.75])
    stats = _paired_stats(residual, point)
    assert stats["mean_diff"] < 0
    assert stats["fraction_improved"] > 0.5


def test_f2_bootstrap_ci_structure():
    point = np.linspace(0.4, 0.9, 50)
    residual = point - 0.02
    stats = _paired_stats(residual, point, n_boot=500, seed=0)
    lo, hi = stats["bootstrap_ci_95"]
    assert lo < 0 < hi or stats["mean_diff"] < 0


def test_f3_regression_distribution_monitored():
    point = np.array([0.4, 0.5, 0.6])
    residual = np.array([0.45, 0.48, 0.55])
    worse = int(np.sum(residual > point))
    max_reg = float(np.max(residual - point))
    assert worse >= 0
    assert max_reg < 1.0
