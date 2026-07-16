"""Phase 3: monotone density control-grid + spice PCA helpers.

Hard RC-1: softplus increments ⇒ strictly increasing σ₀ on the control grid.
PCHIP upsample preserves monotonicity (eval/export). Torch path uses linear interp.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.interpolate import PchipInterpolator


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


def encode_a_from_sigma0_ctrl(sig_ctrl: np.ndarray, dz_tilde: np.ndarray) -> np.ndarray:
    """Invert softplus decode (approx) for truth projection / diagnostics."""
    sig = np.asarray(sig_ctrl, dtype=np.float64)
    dz = np.asarray(dz_tilde, dtype=np.float64)
    if sig.ndim == 1:
        sig = sig[None, :]
    a = np.empty_like(sig)
    a[:, 0] = sig[:, 0]
    # softplus^{-1}(x) = log(expm1(x)); clamp for near-zero increments
    raw = np.diff(sig, axis=1) / np.maximum(dz[1:], 1e-12)
    a[:, 1:] = np.log(np.expm1(np.maximum(raw, 1e-12)))
    return a.astype(np.float32)


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


if __name__ == "__main__":
    selfcheck_monotone_pchip()
    print("density_spice selfcheck: OK")
