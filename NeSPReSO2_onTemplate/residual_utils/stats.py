"""Lightweight paired-stat helpers (no torch/eval_run imports)."""

from __future__ import annotations

import numpy as np


def paired_stats(residual_rmse: np.ndarray, point_rmse: np.ndarray, *, n_boot: int = 2000, seed: int = 42):
    diff = residual_rmse - point_rmse
    mean_diff = float(np.mean(diff))
    try:
        from scipy import stats

        t_stat, t_p = stats.ttest_rel(residual_rmse, point_rmse)
        try:
            w_stat, w_p = stats.wilcoxon(residual_rmse, point_rmse)
        except ValueError:
            w_stat, w_p = float("nan"), float("nan")
    except ImportError:
        t_stat = t_p = w_stat = w_p = float("nan")

    rng = np.random.default_rng(seed)
    n = len(diff)
    boots = [float(np.mean(diff[rng.integers(0, n, size=n)])) for _ in range(n_boot)]
    ci_low, ci_high = np.percentile(boots, [2.5, 97.5])
    return {
        "mean_diff": mean_diff,
        "paired_t_stat": float(t_stat),
        "paired_t_p": float(t_p),
        "wilcoxon_stat": float(w_stat),
        "wilcoxon_p": float(w_p),
        "bootstrap_ci_95": [float(ci_low), float(ci_high)],
        "n_profiles": int(n),
        "fraction_improved": float(np.mean(diff < 0)),
        "fraction_worse": float(np.mean(diff > 0)),
    }
