"""Phase 4.4 — latent diagonal variance → profile-space covariance (DA deliverable)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def spice_covariance(
    sigma_z: np.ndarray,
    pca,
    spice_std: np.ndarray,
    *,
    floor: float = 1e-8,
) -> np.ndarray:
    """Σ_τ = V diag(σ_z²) Vᵀ then de-standardize by spice_std (per level).

    ``sigma_z``: (n_comp,) or (B, n_comp) predictive std in PC space.
    Returns (n_z, n_z) or (B, n_z, n_z).
    """
    V = np.asarray(pca.components_.T, dtype=np.float64)  # (n_z, n_comp)
    sd = np.asarray(spice_std, dtype=np.float64).reshape(-1)
    sz = np.asarray(sigma_z, dtype=np.float64)
    single = sz.ndim == 1
    if single:
        sz = sz[None, :]
    out = []
    for i in range(sz.shape[0]):
        # standardized profile cov, then scale rows/cols by sd
        cov_z = (V * (sz[i] ** 2)) @ V.T
        cov = cov_z * np.outer(sd, sd)
        cov = cov + np.eye(cov.shape[0]) * float(floor)
        out.append(cov)
    arr = np.stack(out, axis=0)
    return arr[0] if single else arr


def density_ctrl_covariance(
    sigma_a: np.ndarray,
    dz_tilde: np.ndarray,
    *,
    floor: float = 1e-8,
) -> np.ndarray:
    """Linearized Σ_ρ on control grid from diagonal var on ``a``.

    Decode: σ₀[0]=a[0]; σ₀[k]=a[0]+Σ softplus(a_j)Δz̃_j.
    Jacobian approximation at softplus'(a)≈sigmoid(a); for export use
    G where G[:,0]=1 and G[k,j]=Δz̃_j * softplus'(a_j) for j<=k (ponytail:
    use softplus'≈1 near typical operating point → G = cumsum incidence).
    """
    dz = np.asarray(dz_tilde, dtype=np.float64).reshape(-1)
    sa = np.asarray(sigma_a, dtype=np.float64)
    single = sa.ndim == 1
    if single:
        sa = sa[None, :]
    k = sa.shape[1]
    # Incidence: ∂σ_i/∂a_0 = 1; ∂σ_i/∂a_j = dz[j] for i>=j>=1 (softplus'≈1)
    G = np.zeros((k, k), dtype=np.float64)
    G[:, 0] = 1.0
    for j in range(1, k):
        G[j:, j] = dz[j]
    out = []
    for i in range(sa.shape[0]):
        cov = G @ np.diag(sa[i] ** 2) @ G.T
        cov = cov + np.eye(k) * float(floor)
        out.append(cov)
    arr = np.stack(out, axis=0)
    return arr[0] if single else arr


def density_lowrank_covariance(
    sigma_z: np.ndarray,
    basis: np.ndarray,
    *,
    floor: float = 1e-8,
) -> np.ndarray:
    """Σ_ρ = V diag(σ_z²) Vᵀ on the control grid (low-rank δσ₀ path).

    ``basis`` is PCA components ``(R, K)``; ``sigma_z`` is ``(R,)`` or ``(B, R)``.
    """
    V = np.asarray(basis, dtype=np.float64)  # (R, K)
    if V.ndim != 2:
        raise ValueError(f"basis must be (R, K), got {V.shape}")
    sz = np.asarray(sigma_z, dtype=np.float64)
    single = sz.ndim == 1
    if single:
        sz = sz[None, :]
    if sz.shape[-1] != V.shape[0]:
        raise ValueError(f"sigma_z last dim {sz.shape[-1]} != R={V.shape[0]}")
    out = []
    Vt = V.T  # (K, R)
    for i in range(sz.shape[0]):
        cov = (Vt * (sz[i] ** 2)) @ V
        cov = cov + np.eye(cov.shape[0]) * float(floor)
        out.append(cov)
    arr = np.stack(out, axis=0)
    return arr[0] if single else arr


def export_profile_covariance(
    mu_a: np.ndarray,
    sigma_a: np.ndarray,
    mu_z: np.ndarray,
    sigma_z: np.ndarray,
    *,
    pca_spice,
    spice_mean: np.ndarray,
    spice_std: np.ndarray,
    dz_tilde: np.ndarray,
    floor: float = 1e-8,
) -> dict[str, Any]:
    """Export μ and Σ for density-ctrl + spice (pre-inversion)."""
    Sigma_rho = density_ctrl_covariance(sigma_a, dz_tilde, floor=floor)
    Sigma_tau = spice_covariance(sigma_z, pca_spice, spice_std, floor=floor)
    return {
        "mu_a": np.asarray(mu_a, dtype=np.float64),
        "mu_z_spice": np.asarray(mu_z, dtype=np.float64),
        "Sigma_rho_ctrl": Sigma_rho,
        "Sigma_tau": Sigma_tau,
        "spice_mean": np.asarray(spice_mean, dtype=np.float64),
        "spice_std": np.asarray(spice_std, dtype=np.float64),
        "note": "T/S Σ via inversion Jacobian is Phase 4.4 follow-up; ctrl+spice is the DA R-matrix seed",
    }


def assert_psd(cov: np.ndarray, *, tol: float = -1e-8) -> float:
    w = np.linalg.eigvalsh(np.asarray(cov, dtype=np.float64))
    amin = float(np.min(w))
    if amin < tol:
        raise AssertionError(f"covariance not PSD: min eig={amin}")
    return amin


def mc_vs_diag_agreement(
    sigma_z: np.ndarray,
    pca,
    spice_std: np.ndarray,
    *,
    n_draw: int = 200,
    seed: int = 0,
    rtol: float = 0.15,
) -> dict[str, float]:
    """Diagonal of analytic Σ must match MC var of V z draws within rtol."""
    rng = np.random.default_rng(seed)
    V = np.asarray(pca.components_.T, dtype=np.float64)
    sd = np.asarray(spice_std, dtype=np.float64).reshape(-1)
    sz = np.asarray(sigma_z, dtype=np.float64).reshape(-1)
    z = rng.normal(size=(n_draw, sz.size)) * sz
    prof = (z @ V.T) * sd  # (n_draw, n_z)
    mc_var = prof.var(axis=0)
    analytic = spice_covariance(sz, pca, sd)
    diag = np.diag(analytic)
    # avoid /0
    rel = np.abs(diag - mc_var) / np.maximum(mc_var, 1e-12)
    return {"max_rel": float(np.max(rel)), "mean_rel": float(np.mean(rel)), "pass": bool(np.max(rel) <= rtol)}


def mc_vs_diag_agreement_lowrank(
    sigma_z: np.ndarray,
    basis: np.ndarray,
    *,
    n_draw: int = 200,
    seed: int = 0,
    rtol: float = 0.15,
    floor: float = 1e-8,
) -> dict[str, float]:
    """§4.4 for score-σ export: diag(Σ_ρ) vs MC var of ``clim + scores @ V``.

    ``basis`` is ``(R, K)`` PCA components (same layout as ``delta_sigma0_basis``).
    """
    rng = np.random.default_rng(seed)
    V = np.asarray(basis, dtype=np.float64)  # (R, K)
    sz = np.asarray(sigma_z, dtype=np.float64).reshape(-1)
    if sz.size != V.shape[0]:
        raise ValueError(f"sigma_z length {sz.size} != R={V.shape[0]}")
    scores = rng.normal(size=(n_draw, sz.size)) * sz
    prof = scores @ V  # (n_draw, K) — anomaly; clim cancels in var
    mc_var = prof.var(axis=0)
    analytic = density_lowrank_covariance(sz, V, floor=floor)
    diag = np.diag(analytic)
    rel = np.abs(diag - mc_var) / np.maximum(mc_var, 1e-12)
    return {
        "max_rel": float(np.max(rel)),
        "mean_rel": float(np.mean(rel)),
        "pass": bool(np.max(rel) <= rtol),
    }
