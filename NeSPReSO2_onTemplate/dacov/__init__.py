"""Phase 4.4 — latent diagonal variance → profile-space covariance (DA deliverable)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def load_sigma_recalib(path: str | Path) -> np.ndarray:
    """Load val-fitted per-dim α from ``sigma_recalib_per_dim.json`` next to a ckpt."""
    obj = json.loads(Path(path).read_text())
    alphas = obj.get("alphas")
    if alphas is None:
        raise KeyError(f"{path}: missing 'alphas'")
    return np.asarray(alphas, dtype=np.float64).reshape(-1)


def apply_sigma_recalib(sigma_z: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    """σ' = α ⊙ σ. Required before Σ export, CRPS, ENCE, PIT, spread-skill."""
    sz = np.asarray(sigma_z, dtype=np.float64)
    a = np.asarray(alphas, dtype=np.float64).reshape(-1)
    if sz.shape[-1] != a.size:
        raise ValueError(f"alpha length {a.size} != sigma last dim {sz.shape[-1]}")
    return sz * a


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
    alphas: np.ndarray | None = None,
) -> np.ndarray:
    """Σ_ρ = V diag(σ_z²) Vᵀ on the control grid (low-rank δσ₀ path).

    ``basis`` is PCA components ``(R, K)``; ``sigma_z`` is ``(R,)`` or ``(B, R)``.
    If ``alphas`` is given, uses σ' = α ⊙ σ (val-fitted recalib) before forming Σ.
    """
    V = np.asarray(basis, dtype=np.float64)  # (R, K)
    if V.ndim != 2:
        raise ValueError(f"basis must be (R, K), got {V.shape}")
    sz = np.asarray(sigma_z, dtype=np.float64)
    if alphas is not None:
        sz = apply_sigma_recalib(sz, alphas)
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
    }


def inversion_jacobian_per_level(
    T: np.ndarray,
    S: np.ndarray,
    depth: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    *,
    eps: float = 1e-4,
) -> np.ndarray:
    """Per-level 2×2 J = ∂(σ₀, τ)/∂(T, S) via central FD. Shape (..., n_z, 2, 2)."""
    from evalphys.gsw_backend import get_gsw
    from evalphys.inversion import sigma0_spice_from_ts

    gsw = get_gsw()
    T = np.asarray(T, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    depth = np.asarray(depth, dtype=np.float64).reshape(-1)
    # broadcast lat/lon to (..., n_z) if needed
    n_z = depth.size
    if T.ndim == 1:
        T = T[None, :]
        S = S[None, :]
        squeeze = True
    else:
        squeeze = False
    B = T.shape[0]
    lat_b = np.broadcast_to(np.asarray(lat, dtype=np.float64).reshape(B, 1), (B, n_z))
    lon_b = np.broadcast_to(np.asarray(lon, dtype=np.float64).reshape(B, 1), (B, n_z))
    p = gsw.p_from_z(-np.broadcast_to(depth, (B, n_z)), lat_b)

    def _st(tt, ss):
        return sigma0_spice_from_ts(tt, ss, p, lon_b, lat_b)

    dT = eps * np.maximum(1.0, np.abs(T))
    dS = eps * np.maximum(1.0, np.abs(S))
    s0_tp, tau_tp = _st(T + dT, S)
    s0_tm, tau_tm = _st(T - dT, S)
    s0_sp, tau_sp = _st(T, S + dS)
    s0_sm, tau_sm = _st(T, S - dS)
    J = np.zeros((B, n_z, 2, 2), dtype=np.float64)
    J[..., 0, 0] = (s0_tp - s0_tm) / (2 * dT)
    J[..., 1, 0] = (tau_tp - tau_tm) / (2 * dT)
    J[..., 0, 1] = (s0_sp - s0_sm) / (2 * dS)
    J[..., 1, 1] = (tau_sp - tau_sm) / (2 * dS)
    return J[0] if squeeze else J


def ts_covariance_from_sigma0_spice(
    Sigma_rho: np.ndarray,
    Sigma_tau: np.ndarray,
    T: np.ndarray,
    S: np.ndarray,
    depth: np.ndarray,
    lat: float | np.ndarray,
    lon: float | np.ndarray,
    *,
    floor: float = 1e-8,
    cross_rho_tau: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Δ-method: Σ_{T,S} from (Σ_ρ, Σ_τ) via per-level inversion Jacobian.

    Builds block-diag J_inv over levels (ponytail: ignore ρ–τ cross unless given),
    then Σ_TS = J_inv Σ_στ J_invᵀ. Returns full vertical Σ_T, Σ_S.
    Ceiling: no horizontal structure; upgrade = dense ρ–τ cross + ctrl→native P.
    """
    Sr = np.asarray(Sigma_rho, dtype=np.float64)
    St = np.asarray(Sigma_tau, dtype=np.float64)
    if Sr.ndim != 2 or St.ndim != 2 or Sr.shape != St.shape:
        raise ValueError(f"Σ_ρ and Σ_τ must be matching square, got {Sr.shape} vs {St.shape}")
    n_z = Sr.shape[0]
    T1 = np.asarray(T, dtype=np.float64).reshape(-1)
    S1 = np.asarray(S, dtype=np.float64).reshape(-1)
    if T1.size != n_z or S1.size != n_z:
        raise ValueError(f"T/S length must be n_z={n_z}")
    lat_a = np.asarray(lat, dtype=np.float64).reshape(-1)
    lon_a = np.asarray(lon, dtype=np.float64).reshape(-1)
    if lat_a.size == 1:
        lat_a = np.full(n_z, float(lat_a[0]))
    if lon_a.size == 1:
        lon_a = np.full(n_z, float(lon_a[0]))

    J = inversion_jacobian_per_level(
        T1, S1, depth, float(np.asarray(lat).reshape(-1)[0]), float(np.asarray(lon).reshape(-1)[0])
    )  # (n_z, 2, 2)
    # J_inv per level
    Jinv = np.zeros_like(J)
    for k in range(n_z):
        det = J[k, 0, 0] * J[k, 1, 1] - J[k, 0, 1] * J[k, 1, 0]
        if abs(det) < 1e-14:
            Jinv[k] = np.eye(2)
        else:
            Jinv[k] = np.array(
                [[J[k, 1, 1], -J[k, 0, 1]], [-J[k, 1, 0], J[k, 0, 0]]],
                dtype=np.float64,
            ) / det

    # Σ_στ = [[Σ_ρ, C], [Cᵀ, Σ_τ]]
    C = np.zeros((n_z, n_z), dtype=np.float64) if cross_rho_tau is None else np.asarray(cross_rho_tau)
    Sig = np.block([[Sr, C], [C.T, St]])

    # Big J_inv: maps [δσ₀; δτ] → [δT; δS] with per-level 2×2 (reorder to interleaved)
    # Use stacked order [σ₀_0..σ₀_{n-1}, τ_0..τ_{n-1}] → [T_0..T_{n-1}, S_0..S_{n-1}]
    Big = np.zeros((2 * n_z, 2 * n_z), dtype=np.float64)
    for k in range(n_z):
        # [T_k, S_k] = Jinv_k @ [σ_k, τ_k]
        Big[k, k] = Jinv[k, 0, 0]
        Big[k, n_z + k] = Jinv[k, 0, 1]
        Big[n_z + k, k] = Jinv[k, 1, 0]
        Big[n_z + k, n_z + k] = Jinv[k, 1, 1]

    Cov = Big @ Sig @ Big.T
    Sigma_T = Cov[:n_z, :n_z] + np.eye(n_z) * float(floor)
    Sigma_S = Cov[n_z:, n_z:] + np.eye(n_z) * float(floor)
    return {"Sigma_T": Sigma_T, "Sigma_S": Sigma_S, "J_per_level": J, "Jinv_per_level": Jinv}


def upsample_cov_linear(cov_ctrl: np.ndarray, z_ctrl: np.ndarray, z_native: np.ndarray) -> np.ndarray:
    """P cov Pᵀ with linear interp rows (ponytail ctrl→native for low-rank Σ_ρ)."""
    zc = np.asarray(z_ctrl, dtype=np.float64).reshape(-1)
    zn = np.asarray(z_native, dtype=np.float64).reshape(-1)
    K, N = zc.size, zn.size
    P = np.zeros((N, K), dtype=np.float64)
    for i, z in enumerate(zn):
        if z <= zc[0]:
            P[i, 0] = 1.0
            continue
        if z >= zc[-1]:
            P[i, -1] = 1.0
            continue
        j = int(np.searchsorted(zc, z) - 1)
        w = (z - zc[j]) / (zc[j + 1] - zc[j])
        P[i, j] = 1.0 - w
        P[i, j + 1] = w
    C = np.asarray(cov_ctrl, dtype=np.float64)
    return P @ C @ P.T


def export_ts_covariance_lowrank(
    sigma_z_rho: np.ndarray,
    basis_rho: np.ndarray,
    sigma_z_spice: np.ndarray,
    pca_spice,
    spice_std: np.ndarray,
    *,
    T: np.ndarray,
    S: np.ndarray,
    z_ctrl: np.ndarray,
    depth: np.ndarray,
    lat: float | np.ndarray,
    lon: float | np.ndarray,
    alphas_rho: np.ndarray | None = None,
    floor: float = 1e-8,
) -> dict[str, np.ndarray]:
    """Winning-path export: score-σ → Σ_ρ_ctrl → native → Σ_T/Σ_S via inversion J."""
    Sigma_rho_ctrl = density_lowrank_covariance(sigma_z_rho, basis_rho, floor=floor, alphas=alphas_rho)
    Sigma_rho = upsample_cov_linear(Sigma_rho_ctrl, z_ctrl, depth)
    Sigma_tau = spice_covariance(sigma_z_spice, pca_spice, spice_std, floor=floor)
    # spice may be native-length already
    if Sigma_tau.shape[0] != Sigma_rho.shape[0]:
        raise ValueError(
            f"Σ_τ n_z={Sigma_tau.shape[0]} != upsampled Σ_ρ n_z={Sigma_rho.shape[0]}"
        )
    out = ts_covariance_from_sigma0_spice(
        Sigma_rho, Sigma_tau, T, S, depth, lat, lon, floor=floor
    )
    out["Sigma_rho_native"] = Sigma_rho
    out["Sigma_tau"] = Sigma_tau
    out["Sigma_rho_ctrl"] = Sigma_rho_ctrl
    return out


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
