"""(σ₀, τ) → (T, S) Newton inversion — numpy path for T1 / Phase 3.4."""

from __future__ import annotations

import gsw
import numpy as np

_SA_BOUNDS = (30.0, 40.0)
_CT_BOUNDS = (-2.0, 35.0)
_FD_EPS = 1e-4
_MAX_ITERS = 12
_TOL = 1e-6


def sigma0_spice_from_ts(
    T: np.ndarray,
    S: np.ndarray,
    p: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    sa = gsw.SA_from_SP(S, p, lon, lat)
    ct = gsw.CT_from_t(sa, T, p)
    return gsw.sigma0(sa, ct), gsw.spiciness0(sa, ct)


def _F(sa: np.ndarray, ct: np.ndarray, s0: np.ndarray, tau: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return gsw.sigma0(sa, ct) - s0, gsw.spiciness0(sa, ct) - tau


def ts_from_sigma0_spice(
    sigma0_tgt: np.ndarray,
    spice_tgt: np.ndarray,
    p: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    *,
    sa0: np.ndarray | None = None,
    ct0: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized Newton inversion; returns T, S, converged mask."""
    shape = np.broadcast_shapes(sigma0_tgt.shape, spice_tgt.shape, p.shape, lon.shape, lat.shape)
    s0 = np.broadcast_to(sigma0_tgt, shape).astype(np.float64).copy()
    tau = np.broadcast_to(spice_tgt, shape).astype(np.float64).copy()
    p_b = np.broadcast_to(p, shape).astype(np.float64)
    lon_b = np.broadcast_to(lon, shape).astype(np.float64)
    lat_b = np.broadcast_to(lat, shape).astype(np.float64)

    if sa0 is None:
        sa = np.full(shape, 35.0, dtype=np.float64)
        ct = np.full(shape, 20.0, dtype=np.float64)
    else:
        sa = np.broadcast_to(sa0, shape).astype(np.float64).copy()
        ct = np.broadcast_to(ct0, shape).astype(np.float64).copy()

    active = np.isfinite(s0) & np.isfinite(tau)
    converged = np.zeros(shape, dtype=bool)

    for _ in range(_MAX_ITERS):
        f0, f1 = _F(sa, ct, s0, tau)
        Finf = np.maximum(np.abs(f0), np.abs(f1))
        converged |= active & (Finf < _TOL)
        active &= ~converged
        if not active.any():
            break

        # Central finite-difference Jacobian columns
        eps_sa = _FD_EPS * np.maximum(1.0, np.abs(sa))
        eps_ct = _FD_EPS * np.maximum(1.0, np.abs(ct))
        fs_p, ft_p = _F(sa + eps_sa, ct, s0, tau)
        fs_m, ft_m = _F(sa - eps_sa, ct, s0, tau)
        fc_p, ft2_p = _F(sa, ct + eps_ct, s0, tau)
        fc_m, ft2_m = _F(sa, ct - eps_ct, s0, tau)
        j11 = (fs_p - fs_m) / (2 * eps_sa)
        j21 = (ft_p - ft_m) / (2 * eps_sa)
        j12 = (fc_p - fc_m) / (2 * eps_ct)
        j22 = (ft2_p - ft2_m) / (2 * eps_ct)

        det = j11 * j22 - j12 * j21
        bad = active & (np.abs(det) < 1e-14)
        active &= ~bad

        step_sa = np.zeros_like(sa)
        step_ct = np.zeros_like(ct)
        m = active
        step_sa[m] = (j22[m] * f0[m] - j12[m] * f1[m]) / det[m]
        step_ct[m] = (-j21[m] * f0[m] + j11[m] * f1[m]) / det[m]

        sa_new = np.clip(sa - step_sa, *_SA_BOUNDS)
        ct_new = np.clip(ct - step_ct, *_CT_BOUNDS)
        f0n, f1n = _F(sa_new, ct_new, s0, tau)
        Fn = np.maximum(np.abs(f0n), np.abs(f1n))
        improve = active & (Fn < Finf)
        sa[improve] = sa_new[improve]
        ct[improve] = ct_new[improve]
        damp = active & ~improve
        sa[damp] = np.clip(sa[damp] - 0.5 * step_sa[damp], *_SA_BOUNDS)
        ct[damp] = np.clip(ct[damp] - 0.5 * step_ct[damp], *_CT_BOUNDS)

    f0, f1 = _F(sa, ct, s0, tau)
    converged |= active & (np.maximum(np.abs(f0), np.abs(f1)) < _TOL)

    T = gsw.t_from_CT(sa, ct, p_b)
    S = gsw.SP_from_SA(sa, p_b, lon_b, lat_b)
    return T, S, converged
