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


def per_level_rmse(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """(n_z,) RMSE; NaNs ignored per level."""
    err = np.asarray(pred, dtype=np.float64) - np.asarray(truth, dtype=np.float64)
    return np.sqrt(np.nanmean(err**2, axis=0))


def r_fixed_diag(rmse: np.ndarray, *, floor: float = 1e-8) -> np.ndarray:
    """Dai-convention R = diag(RMSE²), floored."""
    r = np.asarray(rmse, dtype=np.float64).reshape(-1) ** 2
    return np.maximum(r, float(floor))


def fit_predict_decode(
    sla: np.ndarray,
    sst_anom: np.ndarray,
    juld: np.ndarray,
    pcs_train: np.ndarray,
    *,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    joint_meta: dict,
    pca,
    lam: float = 1.0,
) -> dict:
    """Train-era ridge on PCs → test decode to T/S + R_fixed from test RMSE."""
    from model.joint_eof import reconstruct_joint_eof

    X = design_matrix(sla, sst_anom, juld)
    coef = fit_ridge(X[train_idx], pcs_train[train_idx], lam=lam)
    pcs_hat = predict(X[test_idx], coef)
    T_hat, S_hat = reconstruct_joint_eof(pcs_hat, joint_meta, pca)
    return {
        "coef": coef,
        "pcs_hat": pcs_hat,
        "T_hat": T_hat,
        "S_hat": S_hat,
        "rmse_T": None,  # filled by caller with truth
        "rmse_S": None,
    }


def selfcheck() -> None:
    from model.joint_eof import fit_joint_eof, reconstruct_joint_eof, transform_joint_eof

    rng = np.random.default_rng(0)
    n, r, n_z = 80, 8, 16
    juld = rng.uniform(0, 365, size=n)
    sla = rng.normal(size=n)
    sst = rng.normal(size=n)
    # plant recoverable signal in T/S → joint PCs
    depth = np.linspace(0, 200, n_z)
    T = 20.0 - 0.02 * depth + 0.4 * sla[:, None] + 0.2 * sst[:, None]
    S = 36.0 + 0.001 * depth + 0.05 * sla[:, None]
    T += rng.normal(scale=0.02, size=T.shape)
    S += rng.normal(scale=0.01, size=S.shape)
    tr, te = np.arange(60), np.arange(60, n)
    meta = fit_joint_eof(T[tr], S[tr], n_comp=r)
    pca = meta["pca"]
    pcs = transform_joint_eof(T, S, meta, pca)
    X = design_matrix(sla, sst, juld)
    coef = fit_ridge(X[tr], pcs[tr], lam=0.1)
    pcs_hat = predict(X[te], coef)
    T_hat, S_hat = reconstruct_joint_eof(pcs_hat, meta, pca)
    rmse_T = per_level_rmse(T_hat, T[te])
    R = r_fixed_diag(rmse_T)
    assert coef.shape == (7, r), coef.shape
    assert T_hat.shape == (te.size, n_z)
    assert R.shape == (n_z,) and np.all(R > 0)
    assert float(np.mean(rmse_T)) < 0.5, float(np.mean(rmse_T))
    print(
        f"isop_modas_baseline selfcheck OK "
        f"(holdout mean T RMSE={float(np.mean(rmse_T)):.4f}, R_diag[:3]={R[:3]})"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="train_ready cache (Phase 6 runner); not required for --selfcheck",
    )
    args = ap.parse_args()
    if args.selfcheck:
        selfcheck()
        return 0
    if args.cache is None:
        print("ponytail: pass --cache PATH for cache-backed fit, or --selfcheck")
        return 2
    # Phase 6 runner will wire split + SLA/SST columns from cache inputs.
    print(f"ponytail: cache-backed fit deferred to run_osse.py (got {args.cache})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
