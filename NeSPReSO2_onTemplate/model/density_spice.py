"""Phase 3: monotone density control-grid + spice PCA helpers.

Hard RC-1: softplus increments ⇒ strictly increasing σ₀ on the control grid.
PCHIP upsample preserves monotonicity (eval/export). Torch path uses linear interp.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.interpolate import PchipInterpolator
from sklearn.isotonic import IsotonicRegression


def make_control_grid(depth: np.ndarray, K: int = 64) -> np.ndarray:
    """Log-spaced depth control grid denser near surface (matches T1)."""
    depth = np.asarray(depth, dtype=np.float64)
    z_max = float(np.nanmax(depth))
    z_min = float(np.nanmin(depth))
    z0 = max(z_min if z_min > 0 else (float(depth[1]) if depth.size > 1 else 1.0), 1e-3)
    z = np.logspace(np.log10(z0), np.log10(max(z_max, z0 + 1.0)), K).astype(np.float64)
    z[0] = z0
    z[-1] = max(z_max, z0 + 1.0)
    return z


def normalized_dz(z_ctrl: np.ndarray) -> np.ndarray:
    """Δz̃ with mean 1; length K, first entry unused (surface unconstrained)."""
    z = np.asarray(z_ctrl, dtype=np.float64)
    dz = np.diff(z, prepend=z[0])
    dz[0] = 1.0
    mean = float(np.mean(dz[1:])) if z.size > 1 else 1.0
    mean = max(mean, 1e-12)
    return (dz / mean).astype(np.float64)


def decode_sigma0_ctrl(a: torch.Tensor, dz_tilde: torch.Tensor) -> torch.Tensor:
    """a (B,K) → σ₀ on control grid (B,K); softplus increments ⇒ monotone."""
    if a.ndim != 2:
        raise ValueError(f"a must be (B,K), got {tuple(a.shape)}")
    dz = dz_tilde.to(device=a.device, dtype=a.dtype)
    if dz.shape[-1] != a.shape[-1]:
        raise ValueError(f"dz_tilde length {dz.shape[-1]} != K={a.shape[-1]}")
    incr = F.softplus(a[:, 1:]) * dz[1:]
    out = torch.empty_like(a)
    out[:, 0] = a[:, 0]
    out[:, 1:] = a[:, 0:1] + torch.cumsum(incr, dim=1)
    return out


def decode_a_from_output(mu_raw, a_clim, n_scores, n_spice, basis=None):
    """Full-rank residual-δa path: ``a = a_clim + δa`` (``basis`` unused / identity).

    Low-rank mode uses ``decode_sigma0_from_scores`` (σ₀-space PCA) instead —
    a-space PCA has a catastrophic σ₀ recon ceiling (PLAN §3.6 erratum).
    """
    scores = mu_raw[..., :n_scores]
    z_tau = mu_raw[..., n_scores : n_scores + n_spice]
    delta = scores if basis is None else scores @ basis
    return a_clim + delta, z_tau


def decode_sigma0_from_scores(mu_raw, sigma0_clim, n_scores, n_spice, basis):
    """Low-rank density: ``σ̂₀ = σ₀_clim + scores @ basis`` + spice slice.

    ``basis`` is ``(R, K)`` from PCA on train ``(σ₀_ctrl − σ₀_clim)``. Works for
    numpy or torch. **Not monotone by construction** — callers MUST apply
    ``project_monotone_sigma0_ctrl`` before upsample/invert (PLAN §3.6 opt-2 /
    T1-D). Claim is *"stable by construction at inference"*, not hard-head.
    """
    scores = mu_raw[..., :n_scores]
    z_tau = mu_raw[..., n_scores : n_scores + n_spice]
    return sigma0_clim + scores @ basis, z_tau


def project_monotone_sigma0_ctrl(sig_ctrl: np.ndarray, z_ctrl: np.ndarray) -> np.ndarray:
    """L2-optimal increasing projection on the control grid (isotonic).

    Linear interp of native σ₀ onto a coarse ctrl grid leaves ~12% negative
    increments (GoM ARGO). Softplus encode clamps those to ~0 and the bias
    accumulates with depth — the T1-E >800 m RMSE failure. Isotonic before
    encode removes the pathology (see reports/e_deep_band_diagnostic.md).
    """
    sig = np.asarray(sig_ctrl, dtype=np.float64)
    z = np.asarray(z_ctrl, dtype=np.float64)
    single = sig.ndim == 1
    if single:
        sig = sig[None, :]
    out = np.empty_like(sig)
    for i in range(sig.shape[0]):
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
        out[i] = iso.fit_transform(z, sig[i])
    return out[0] if single else out


def encode_a_from_sigma0_ctrl(
    sig_ctrl: np.ndarray,
    dz_tilde: np.ndarray,
    z_ctrl: np.ndarray | None = None,
    *,
    monotone: bool = True,
) -> np.ndarray:
    """Invert softplus decode for truth projection / diagnostics.

    When ``monotone=True`` (default), isotonic-project ``sig_ctrl`` first.
    ``z_ctrl`` is required in that case. Pass ``monotone=False`` only for
    isolating the clamp pathology in diagnostics.
    """
    sig = np.asarray(sig_ctrl, dtype=np.float64)
    dz = np.asarray(dz_tilde, dtype=np.float64)
    if sig.ndim == 1:
        sig = sig[None, :]
        squeeze = True
    else:
        squeeze = False
    if monotone:
        if z_ctrl is None:
            raise ValueError("encode_a_from_sigma0_ctrl(..., monotone=True) requires z_ctrl")
        sig = project_monotone_sigma0_ctrl(sig, z_ctrl)
    a = np.empty_like(sig)
    a[:, 0] = sig[:, 0]
    # softplus^{-1}(x) = log(expm1(x)); clamp for near-zero increments
    raw = np.diff(sig, axis=1) / np.maximum(dz[1:], 1e-12)
    a[:, 1:] = np.log(np.expm1(np.maximum(raw, 1e-12)))
    out = a.astype(np.float32)
    return out[0] if squeeze else out


def interp_linear_torch(
    sigma_ctrl: torch.Tensor,
    z_ctrl: torch.Tensor,
    z_native: torch.Tensor,
) -> torch.Tensor:
    """Monotone-preserving linear upsample (B,K) → (B,n_z)."""
    zc = z_ctrl.to(device=sigma_ctrl.device, dtype=sigma_ctrl.dtype)
    zn = z_native.to(device=sigma_ctrl.device, dtype=sigma_ctrl.dtype)
    z0, z1 = zc[0], zc[-1]
    zn = zn.clamp(z0, z1)
    idx = torch.searchsorted(zc.contiguous(), zn)
    idx = idx.clamp(1, zc.numel() - 1)
    z_lo = zc[idx - 1]
    z_hi = zc[idx]
    w = (zn - z_lo) / torch.clamp(z_hi - z_lo, min=1e-12)
    s_lo = sigma_ctrl[:, idx - 1]
    s_hi = sigma_ctrl[:, idx]
    return s_lo + w * (s_hi - s_lo)


def upsample_pchip(
    sigma_ctrl: np.ndarray,
    z_ctrl: np.ndarray,
    z_native: np.ndarray,
) -> np.ndarray:
    """PCHIP upsample; monotone ctrl ⇒ monotone native.

    Query depths are clamped to the control-grid span so PCHIP never
    extrapolates (extrapolate=True can invent non-monotone tails above/below).
    """
    sig = np.asarray(sigma_ctrl, dtype=np.float64)
    zc = np.asarray(z_ctrl, dtype=np.float64)
    zn = np.clip(np.asarray(z_native, dtype=np.float64), zc[0], zc[-1])
    single = sig.ndim == 1
    if single:
        sig = sig[None, :]
    out = np.empty((sig.shape[0], zn.size), dtype=np.float64)
    for i in range(sig.shape[0]):
        out[i] = PchipInterpolator(zc, sig[i], extrapolate=False)(zn)
    return out[0] if single else out


def selfcheck_monotone_pchip(n: int = 1000, K: int = 64, n_z: int = 201, seed: int = 0) -> None:
    """Softplus decode + PCHIP upsample stay non-decreasing (PLAN §3.2)."""
    rng = np.random.default_rng(seed)
    depth = np.linspace(0.0, 2000.0, n_z)
    z_ctrl = make_control_grid(depth, K=K)
    dz = normalized_dz(z_ctrl)
    a = rng.normal(size=(n, K)).astype(np.float32)
    a[:, 0] = 24.0 + rng.normal(scale=0.5, size=n).astype(np.float32)
    with torch.no_grad():
        sig_c = decode_sigma0_ctrl(torch.from_numpy(a), torch.from_numpy(dz)).numpy()
    assert np.all(np.diff(sig_c, axis=1) >= -1e-12), "control-grid not monotone"
    sig_n = upsample_pchip(sig_c, z_ctrl, depth)
    assert np.all(np.diff(sig_n, axis=1) >= -1e-12), "PCHIP upsample not monotone"
    # Isotonic-before-encode roundtrip on a mildly non-monotone ctrl sample
    sig_wiggle = sig_c.copy()
    sig_wiggle[:, K // 4] -= 0.05  # inject a local inversion
    a_rt = encode_a_from_sigma0_ctrl(sig_wiggle, dz, z_ctrl, monotone=True)
    with torch.no_grad():
        sig_rt = decode_sigma0_ctrl(torch.from_numpy(a_rt), torch.from_numpy(dz)).numpy()
    assert np.all(np.diff(sig_rt, axis=1) >= -1e-12), "encode(monotone)+decode not monotone"
    assert float(np.max(np.abs(sig_rt - project_monotone_sigma0_ctrl(sig_wiggle, z_ctrl)))) < 1e-4


def selfcheck_lowrank_delta_a(n: int = 400, K: int = 64, R: int = 16, n_z: int = 201, seed: int = 1) -> None:
    """Low-rank δσ₀: PCA(R) on (σ₀ − clim) then isotonic stays near-monotone (PLAN §3.6)."""
    from sklearn.decomposition import PCA

    rng = np.random.default_rng(seed)
    depth = np.linspace(0.0, 2000.0, n_z)
    z_ctrl = make_control_grid(depth, K=K)
    zt = (z_ctrl - z_ctrl[0]) / (z_ctrl[-1] - z_ctrl[0])
    base = 24.0 + 3.0 * np.tanh(4.0 * zt)
    modes = np.stack([np.cos((m + 1) * np.pi * zt) for m in range(6)], axis=0)
    amps = rng.normal(scale=[0.3, 0.15, 0.08, 0.05, 0.03, 0.02], size=(n, 6))
    sig = project_monotone_sigma0_ctrl(base[None, :] + amps @ modes, z_ctrl)
    clim = sig.mean(0)
    pca = PCA(n_components=R).fit(sig - clim)
    basis = pca.components_.astype(np.float64)
    clim_eff = clim + pca.mean_
    scores = pca.transform(sig - clim).astype(np.float64)
    mu_raw = np.hstack([scores, np.zeros((n, 1))])
    sig_hat, z_tau = decode_sigma0_from_scores(mu_raw, clim_eff, R, 1, basis=basis)
    assert sig_hat.shape == (n, K) and z_tau.shape == (n, 1)
    sig_mono = project_monotone_sigma0_ctrl(sig_hat, z_ctrl)
    rmse = float(np.sqrt(np.mean((sig_mono - sig) ** 2)))
    rmse_clim = float(np.sqrt(np.mean((clim_eff - sig) ** 2)))
    assert rmse < 0.1 * rmse_clim, f"σ₀-space low-rank too weak: {rmse:.4f} vs clim {rmse_clim:.4f}"
    # round-trip identity before isotonic
    assert float(np.max(np.abs(scores @ basis + clim_eff - sig_hat))) < 1e-6


if __name__ == "__main__":
    selfcheck_monotone_pchip()
    selfcheck_lowrank_delta_a()
    print("density_spice selfcheck: OK")
