#!/usr/bin/env python3
"""ISOP/MODAS-class synthetic-profile baseline (PLAN §6.2).

Ridge-regress joint-EOF PC scores from (SLA, SST_anom, month harmonics) on the
train era; decode to T/S. Depth-dependent R = per-level test RMSE².

CPU-only; does not require Phase 5 winner. Selfcheck: synthetic smoke + shape pins.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def month_harmonics(juld: np.ndarray) -> np.ndarray:
    """(N, 4) = [cos ωt, sin ωt, cos 2ωt, sin 2ωt] with ω = 2π/365.25."""
    t = np.asarray(juld, dtype=np.float64).reshape(-1)
    w = 2.0 * np.pi / 365.25
    return np.column_stack(
        [np.cos(w * t), np.sin(w * t), np.cos(2 * w * t), np.sin(2 * w * t)]
    )


def design_matrix(sla: np.ndarray, sst_anom: np.ndarray, juld: np.ndarray) -> np.ndarray:
    """(N, 7) = [1, SLA, SST_anom, month harmonics]."""
    sla = np.asarray(sla, dtype=np.float64).reshape(-1)
    sst = np.asarray(sst_anom, dtype=np.float64).reshape(-1)
    harm = month_harmonics(juld)
    return np.column_stack([np.ones(sla.size), sla, sst, harm])


def fit_ridge(X: np.ndarray, Y: np.ndarray, *, lam: float = 1.0) -> np.ndarray:
    """Y (N, R) → coef (P, R). Closed-form ridge."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    p = X.shape[1]
    A = X.T @ X + float(lam) * np.eye(p)
    return np.linalg.solve(A, X.T @ Y)


def predict(X: np.ndarray, coef: np.ndarray) -> np.ndarray:
    return np.asarray(X, dtype=np.float64) @ np.asarray(coef, dtype=np.float64)


def selfcheck() -> None:
    rng = np.random.default_rng(0)
    n, r = 80, 8
    juld = rng.uniform(0, 365, size=n)
    sla = rng.normal(size=n)
    sst = rng.normal(size=n)
    # plant a recoverable signal
    Y = (
        0.5 * sla[:, None]
        + 0.3 * sst[:, None]
        + 0.2 * np.cos(2 * np.pi * juld / 365.25)[:, None]
        + rng.normal(scale=0.05, size=(n, r))
    )
    X = design_matrix(sla, sst, juld)
    coef = fit_ridge(X[:60], Y[:60], lam=0.1)
    pred = predict(X[60:], coef)
    rmse = float(np.sqrt(np.mean((pred - Y[60:]) ** 2)))
    assert coef.shape == (7, r), coef.shape
    assert rmse < 0.5, rmse
    print(f"isop_modas_baseline selfcheck OK (holdout RMSE={rmse:.4f})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        selfcheck()
        return 0
    print("ponytail: full cache-backed fit lands with Phase 6 runner; use --selfcheck for now")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
