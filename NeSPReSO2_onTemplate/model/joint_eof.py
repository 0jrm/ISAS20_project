"""Joint T/S EOF (Phase 5 matrix cell B / T1 variant B).

Protocol (frozen to T1): per-level z-score on train, concat [T;S], PCA-32,
decode → destandardize. Not a stability fix — ablation arm only.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.decomposition import PCA


def fit_level_stats(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(X, dtype=np.float64)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float64), std.astype(np.float64)


def level_zscore(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (np.asarray(X, dtype=np.float64) - mean) / std


def fit_joint_eof(
    T_tr: np.ndarray,
    S_tr: np.ndarray,
    n_comp: int,
) -> dict[str, Any]:
    """Fit joint EOF on train profiles (N, n_z). Returns meta + sklearn PCA."""
    T_mean, T_std = fit_level_stats(T_tr)
    S_mean, S_std = fit_level_stats(S_tr)
    joint = np.hstack([level_zscore(T_tr, T_mean, T_std), level_zscore(S_tr, S_mean, S_std)])
    pca = PCA(n_components=int(n_comp)).fit(joint)
    return {
        "T_mean": T_mean,
        "T_std": T_std,
        "S_mean": S_mean,
        "S_std": S_std,
        "n_lev": int(T_tr.shape[1]),
        "n_comp": int(n_comp),
        "pca": pca,
    }


def transform_joint_eof(
    T: np.ndarray,
    S: np.ndarray,
    meta: dict[str, Any],
    pca: PCA | None = None,
) -> np.ndarray:
    """Return PCs (N, n_comp)."""
    pca = pca if pca is not None else meta["pca"]
    n_lev = int(meta["n_lev"])
    joint = np.hstack(
        [
            level_zscore(T, meta["T_mean"], meta["T_std"]),
            level_zscore(S, meta["S_mean"], meta["S_std"]),
        ]
    )
    if joint.shape[1] != 2 * n_lev:
        raise ValueError(f"joint width {joint.shape[1]} != 2*n_lev={2 * n_lev}")
    return pca.transform(joint).astype(np.float32)


def reconstruct_joint_eof(
    pcs: np.ndarray,
    meta: dict[str, Any],
    pca: PCA | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode PCs → (T, S) each (N, n_z)."""
    pca = pca if pca is not None else meta["pca"]
    n_lev = int(meta["n_lev"])
    rec = pca.inverse_transform(np.asarray(pcs, dtype=np.float64))
    T = rec[:, :n_lev] * meta["T_std"] + meta["T_mean"]
    S = rec[:, n_lev:] * meta["S_std"] + meta["S_mean"]
    return T.astype(np.float32), S.astype(np.float32)


def torch_reconstruct_joint_eof(
    pcs: torch.Tensor,
    components: torch.Tensor,
    pca_mean: torch.Tensor,
    T_mean: torch.Tensor,
    T_std: torch.Tensor,
    S_mean: torch.Tensor,
    S_std: torch.Tensor,
    n_lev: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """pcs (B, R) → T,S (B, n_z). ``components`` is (R, 2 n_z) sklearn layout."""
    # sklearn: X_hat = pcs @ components + mean
    rec = pcs @ components + pca_mean
    T = rec[:, :n_lev] * T_std + T_mean
    S = rec[:, n_lev:] * S_std + S_mean
    return T, S


def selfcheck_joint_eof_roundtrip(*, n: int = 40, n_z: int = 50, n_comp: int = 8, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    T = 20 + rng.normal(size=(n, n_z)).cumsum(axis=1) * 0.01
    S = 36 + rng.normal(size=(n, n_z)) * 0.05
    tr, te = slice(0, 30), slice(30, None)
    meta = fit_joint_eof(T[tr], S[tr], n_comp)
    pca = meta.pop("pca")
    pcs = transform_joint_eof(T[te], S[te], meta, pca)
    T_hat, S_hat = reconstruct_joint_eof(pcs, meta, pca)
    assert T_hat.shape == T[te].shape and S_hat.shape == S[te].shape
    # train recon should be tight with enough components
    pcs_tr = transform_joint_eof(T[tr], S[tr], meta, pca)
    Tr, Sr = reconstruct_joint_eof(pcs_tr, meta, pca)
    assert float(np.mean((Tr - T[tr]) ** 2)) < 1.0
    assert float(np.mean((Sr - S[tr]) ** 2)) < 0.05
