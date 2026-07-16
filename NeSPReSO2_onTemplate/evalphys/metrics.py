"""Physical profile metrics — headline path uses reference ``gsw`` via ``get_gsw()``."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from evalphys.constants import (
    DEPTH_BAND_LABELS,
    DEPTH_BANDS,
    MLD_DSIGMA_THRESHOLD,
    MLD_Z_REF_M,
    N2_TOL,
    N2_TOL_SWEEP,
    RHO0_KGM3,
    SIGMA0_TOL,
)
from evalphys.gsw_backend import get_gsw


def _as_profiles_levels(x: np.ndarray) -> np.ndarray:
    """Ensure shape (n_profiles, n_levels)."""
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"expected 2-D array, got shape {a.shape}")
    return a


def to_teos10(
    T: np.ndarray,
    S: np.ndarray,
    depth: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """TEOS-10 SA, CT, pressure [dbar]. Arrays (n_profiles, n_levels)."""
    T = _as_profiles_levels(T)
    S = _as_profiles_levels(S)
    if T.shape != S.shape:
        raise ValueError(f"T shape {T.shape} != S shape {S.shape}")
    n_prof, n_lev = T.shape
    z = np.asarray(depth, dtype=np.float64).reshape(-1)
    if z.shape[0] != n_lev:
        raise ValueError(f"depth length {z.shape[0]} != n_levels {n_lev}")
    gsw = get_gsw()
    lat_v = np.asarray(lat, dtype=np.float64).reshape(n_prof, 1)
    lon_v = np.asarray(lon, dtype=np.float64).reshape(n_prof, 1)
    p = gsw.p_from_z(-np.broadcast_to(z, (n_prof, n_lev)), lat_v)
    sa = gsw.SA_from_SP(S, p, lon_v, lat_v)
    ct = gsw.CT_from_t(sa, T, p)
    return sa, ct, p


def sigma0_profiles(T: np.ndarray, S: np.ndarray, depth: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    gsw = get_gsw()
    sa, ct, _ = to_teos10(T, S, depth, lat, lon)
    return gsw.sigma0(sa, ct)


def _interface_depth_m(depth: np.ndarray) -> np.ndarray:
    z = np.asarray(depth, dtype=np.float64).reshape(-1)
    return 0.5 * (z[:-1] + z[1:])


def _band_mask(depths: np.ndarray, lo: float, hi: float) -> np.ndarray:
    d = np.asarray(depths, dtype=np.float64)
    if np.isfinite(hi):
        return (d >= lo) & (d < hi)
    return d >= lo


def static_stability_violations(
    T: np.ndarray,
    S: np.ndarray,
    depth: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    *,
    n2_tol: float = N2_TOL,
    exclude_top_m: float | None = None,
) -> dict[str, Any]:
    """N² violations via ``gsw.Nsquared``; mid-level depths for stratification."""
    T = _as_profiles_levels(T)
    S = _as_profiles_levels(S)
    n_prof, n_lev = T.shape
    gsw = get_gsw()
    sa, ct, p = to_teos10(T, S, depth, lat, lon)
    lat_v = np.asarray(lat, dtype=np.float64).reshape(n_prof)

    # gsw expects depth-major for vector stacks; loop profiles for clarity (ponytail: n≈4k ok)
    n2_list, pmid_list = [], []
    for i in range(n_prof):
        n2_i, p_mid_i = gsw.Nsquared(sa[i], ct[i], p[i], lat_v[i])
        n2_list.append(n2_i)
        pmid_list.append(p_mid_i)
    n2 = np.stack(n2_list, axis=1)  # (n_interfaces, n_prof)
    p_mid = np.stack(pmid_list, axis=1)
    z_mid = -gsw.z_from_p(p_mid, np.broadcast_to(lat_v, p_mid.shape))

    finite = np.isfinite(n2) & np.isfinite(T[:, :-1].T) & np.isfinite(T[:, 1:].T)
    if exclude_top_m is not None:
        # Keep interfaces at/below exclude_top_m (drop the near-surface band).
        finite &= z_mid >= float(exclude_top_m)

    viol = finite & (n2 < -float(n2_tol))
    prof_viol = np.any(viol, axis=0)
    n_iface = int(finite.sum())
    n_viol = int(viol.sum())

    return {
        "n2_tol": float(n2_tol),
        "exclude_top_m": exclude_top_m,
        "violation_rate_profile": float(np.mean(prof_viol)) if n_prof else 0.0,
        "violation_rate_level": float(n_viol / n_iface) if n_iface else 0.0,
        "n_profiles": n_prof,
        "n_interfaces_checked": n_iface,
        "n_violations": n_viol,
        "interface_depth_m": z_mid[:, 0].tolist(),
        "by_depth_band": {
            "violation_rate_level": _stratify_violation_rate(viol, finite, z_mid[:, 0]),
        },
    }


def _stratify_violation_rate(viol: np.ndarray, finite: np.ndarray, z_iface: np.ndarray) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for label, (lo, hi) in zip(DEPTH_BAND_LABELS, DEPTH_BANDS):
        b = _band_mask(z_iface, lo, hi)
        m = finite & b[:, None]
        den = int(m.sum())
        out[label] = float(viol[m].sum() / den) if den else None
    return out


def sigma0_monotonicity_violations(
    T: np.ndarray,
    S: np.ndarray,
    depth: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    *,
    sigma0_tol: float = SIGMA0_TOL,
    exclude_top_m: float | None = None,
) -> dict[str, Any]:
    """σ₀-space stability: violation iff Δσ₀ < −tol with depth increasing (PLAN §3.2 constraint space)."""
    T = _as_profiles_levels(T)
    S = _as_profiles_levels(S)
    n_prof = T.shape[0]
    z = np.asarray(depth, dtype=np.float64).reshape(-1)
    sig = sigma0_profiles(T, S, depth, lat, lon)
    dsig = np.diff(sig, axis=1)  # (n_prof, n_iface); stable ⇒ dsig >= 0
    z_mid = _interface_depth_m(z)
    finite = np.isfinite(dsig) & np.isfinite(T[:, :-1]) & np.isfinite(T[:, 1:])
    if exclude_top_m is not None:
        finite &= z_mid[None, :] >= float(exclude_top_m)
    viol = finite & (dsig < -float(sigma0_tol))
    prof_viol = np.any(viol, axis=1)
    n_iface = int(finite.sum())
    n_viol = int(viol.sum())
    return {
        "sigma0_tol": float(sigma0_tol),
        "exclude_top_m": exclude_top_m,
        "violation_rate_profile": float(np.mean(prof_viol)) if n_prof else 0.0,
        "violation_rate_level": float(n_viol / n_iface) if n_iface else 0.0,
        "n_profiles": n_prof,
        "n_interfaces_checked": n_iface,
        "n_violations": n_viol,
        "by_depth_band": {
            "violation_rate_level": _stratify_violation_rate(viol.T, finite.T, z_mid),
        },
    }


def static_stability_tolerance_sweep(
    T: np.ndarray,
    S: np.ndarray,
    depth: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    *,
    tolerances: Sequence[float] = N2_TOL_SWEEP,
    exclude_top_m: float | None = None,
) -> dict[str, Any]:
    sweep = {}
    for tol in tolerances:
        key = f"{tol:.0e}" if tol > 0 else "0"
        sweep[key] = static_stability_violations(
            T, S, depth, lat, lon, n2_tol=tol, exclude_top_m=exclude_top_m
        )
    return sweep


def _interp_at_depth(z: np.ndarray, values: np.ndarray, z_query: float) -> np.ndarray:
    """Linear interpolation along depth axis; z shape (n_lev,), values (n_prof, n_lev)."""
    out = np.full(values.shape[0], np.nan, dtype=np.float64)
    for i in range(values.shape[0]):
        vi, zi = values[i], z
        ok = np.isfinite(vi) & np.isfinite(zi)
        if ok.sum() < 2:
            continue
        out[i] = float(np.interp(z_query, zi[ok], vi[ok]))
    return out


def _first_crossing_depth(z: np.ndarray, values: np.ndarray, threshold: np.ndarray, *, greater: bool) -> np.ndarray:
    """Shallowest depth where values crosses threshold (linear interp between levels)."""
    n_prof = values.shape[0]
    out = np.full(n_prof, np.nan, dtype=np.float64)
    for i in range(n_prof):
        vi, zi, ti = values[i], z, threshold[i] if np.ndim(threshold) else threshold
        ok = np.isfinite(vi) & np.isfinite(zi)
        if ok.sum() < 2:
            continue
        diff = vi - ti
        for k in range(int(ok.sum()) - 1):
            idx = np.where(ok)[0]
            a, b = idx[k], idx[k + 1]
            da, db = diff[a], diff[b]
            if greater:
                if da > 0:
                    out[i] = zi[a]
                    break
                if da <= 0 < db:
                    frac = -da / (db - da)
                    out[i] = zi[a] + frac * (zi[b] - zi[a])
                    break
            else:
                if da < 0:
                    out[i] = zi[a]
                    break
                if da >= 0 > db:
                    frac = -da / (db - da)
                    out[i] = zi[a] + frac * (zi[b] - zi[a])
                    break
    return out


def mixed_layer_depth(
    T: np.ndarray,
    S: np.ndarray,
    depth: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    *,
    z_ref_m: float = MLD_Z_REF_M,
    dsigma: float = MLD_DSIGMA_THRESHOLD,
) -> np.ndarray:
    sig = sigma0_profiles(T, S, depth, lat, lon)
    z = np.asarray(depth, dtype=np.float64).reshape(-1)
    sig_ref = _interp_at_depth(z, sig, z_ref_m)
    return _first_crossing_depth(z, sig, sig_ref + dsigma, greater=True)


def isotherm_depth(T: np.ndarray, depth: np.ndarray, temp_c: float) -> tuple[np.ndarray, float]:
    """Depth of isotherm; returns (depths, coverage fraction with crossing)."""
    T = _as_profiles_levels(T)
    z = np.asarray(depth, dtype=np.float64).reshape(-1)
    # Find shallowest crossing of temp_c (from warm surface downward)
    depths = np.full(T.shape[0], np.nan, dtype=np.float64)
    for i in range(T.shape[0]):
        ti, zi = T[i], z
        ok = np.isfinite(ti) & np.isfinite(zi)
        if ok.sum() < 2:
            continue
        for k in range(int(ok.sum()) - 1):
            idx = np.where(ok)[0]
            a, b = idx[k], idx[k + 1]
            ta, tb = ti[a], ti[b]
            if (ta - temp_c) * (tb - temp_c) <= 0 and ta != tb:
                frac = (temp_c - ta) / (tb - ta)
                depths[i] = zi[a] + frac * (zi[b] - zi[a])
                break
    coverage = float(np.isfinite(depths).mean()) if depths.size else 0.0
    return depths, coverage


def drhodz_rmse(
    T_pred: np.ndarray,
    S_pred: np.ndarray,
    T_true: np.ndarray,
    S_true: np.ndarray,
    depth: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
) -> dict[str, Any]:
    z = np.asarray(depth, dtype=np.float64).reshape(-1)
    sig_p = sigma0_profiles(T_pred, S_pred, depth, lat, lon)
    sig_t = sigma0_profiles(T_true, S_true, depth, lat, lon)
    dz = np.diff(z)
    dz = np.where(dz > 0, dz, np.nan)
    dsp_p = np.diff(sig_p, axis=1) / dz
    dsp_t = np.diff(sig_t, axis=1) / dz
    err = dsp_p - dsp_t
    z_mid = _interface_depth_m(z)
    valid = np.isfinite(err)
    overall = float(np.sqrt(np.mean(err[valid] ** 2))) if valid.any() else np.nan
    by_band: dict[str, float | None] = {}
    for label, (lo, hi) in zip(DEPTH_BAND_LABELS, DEPTH_BANDS):
        b = _band_mask(z_mid, lo, hi)
        m = valid[:, b]
        by_band[label] = float(np.sqrt(np.mean(err[:, b][m] ** 2))) if m.any() else None
    return {"rmse_overall": overall, "by_depth_band": by_band}


def ts_rmse_by_band(
    T_pred: np.ndarray,
    S_pred: np.ndarray,
    T_true: np.ndarray,
    S_true: np.ndarray,
    depth: np.ndarray,
) -> dict[str, Any]:
    z = np.asarray(depth, dtype=np.float64).reshape(-1)
    out: dict[str, Any] = {}
    for var, pred, true in (("T", T_pred, T_true), ("S", S_pred, S_true)):
        err2 = (pred - true) ** 2
        bands = {}
        for label, (lo, hi) in zip(DEPTH_BAND_LABELS, DEPTH_BANDS):
            b = _band_mask(z, lo, hi)
            m = np.isfinite(err2[:, b])
            bands[label] = float(np.sqrt(np.mean(err2[:, b][m]))) if m.any() else None
        out[var] = bands
    return out


def steric_height_cm(
    T: np.ndarray,
    S: np.ndarray,
    depth: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    *,
    T_clim: np.ndarray | None = None,
    S_clim: np.ndarray | None = None,
) -> np.ndarray:
    """Steric height anomaly [cm] relative to climatology (or zero if clim absent)."""
    gsw = get_gsw()
    sa, ct, p = to_teos10(T, S, depth, lat, lon)
    rho = gsw.rho(sa, ct, p)
    if T_clim is not None and S_clim is not None:
        sa_c, ct_c, p_c = to_teos10(T_clim, S_clim, depth, lat, lon)
        rho_clim = gsw.rho(sa_c, ct_c, p_c)
    else:
        rho_clim = np.zeros_like(rho)
    z = np.asarray(depth, dtype=np.float64).reshape(1, -1)
    dz = np.diff(z, axis=1)
    rho_mid = 0.5 * (rho[:, :-1] + rho[:, 1:])
    rho_clim_mid = 0.5 * (rho_clim[:, :-1] + rho_clim[:, 1:])
    integrand = (rho_mid - rho_clim_mid) * dz
    # integrate from bottom to surface (positive up)
    eta_m = -np.nansum(integrand, axis=1) / RHO0_KGM3
    return eta_m * 100.0


def _rmse_bias(pred: np.ndarray, true: np.ndarray) -> dict[str, float | None]:
    m = np.isfinite(pred) & np.isfinite(true)
    if not m.any():
        return {"rmse": None, "bias": None, "n": 0}
    d = pred[m] - true[m]
    return {"rmse": float(np.sqrt(np.mean(d**2))), "bias": float(np.mean(d)), "n": int(m.sum())}


def summarize_physical(
    T_pred: np.ndarray,
    S_pred: np.ndarray,
    T_true: np.ndarray,
    S_true: np.ndarray,
    depth: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    *,
    T_clim: np.ndarray | None = None,
    S_clim: np.ndarray | None = None,
) -> dict[str, Any]:
    """Bundle headline physical metrics for one prediction/truth pair."""
    stab = static_stability_tolerance_sweep(T_pred, S_pred, depth, lat, lon)
    stab_excl = static_stability_tolerance_sweep(T_pred, S_pred, depth, lat, lon, exclude_top_m=15.0)
    mld_p = mixed_layer_depth(T_pred, S_pred, depth, lat, lon)
    mld_t = mixed_layer_depth(T_true, S_true, depth, lat, lon)
    d20_p, cov20_p = isotherm_depth(T_pred, depth, 20.0)
    d20_t, cov20_t = isotherm_depth(T_true, depth, 20.0)
    d26_p, cov26_p = isotherm_depth(T_pred, depth, 26.0)
    d26_t, cov26_t = isotherm_depth(T_true, depth, 26.0)
    eta_p = steric_height_cm(T_pred, S_pred, depth, lat, lon, T_clim=T_clim, S_clim=S_clim)
    eta_t = steric_height_cm(T_true, S_true, depth, lat, lon, T_clim=T_clim, S_clim=S_clim)
    eta_m = np.isfinite(eta_p) & np.isfinite(eta_t)
    steric_rms = float(np.sqrt(np.mean((eta_p[eta_m] - eta_t[eta_m]) ** 2))) if eta_m.any() else None

    return {
        "static_stability_pred": stab,
        "static_stability_pred_exclude_top15m": stab_excl,
        "sigma0_monotonicity_pred": sigma0_monotonicity_violations(T_pred, S_pred, depth, lat, lon),
        "ts_rmse": ts_rmse_by_band(T_pred, S_pred, T_true, S_true, depth),
        "drhodz_rmse": drhodz_rmse(T_pred, S_pred, T_true, S_true, depth, lat, lon),
        "mld": {"pred_vs_true": _rmse_bias(mld_p, mld_t)},
        "D20": {
            "coverage_pred": cov20_p,
            "coverage_true": cov20_t,
            "pred_vs_true": _rmse_bias(d20_p, d20_t),
        },
        "D26": {
            "coverage_pred": cov26_p,
            "coverage_true": cov26_t,
            "pred_vs_true": _rmse_bias(d26_p, d26_t),
        },
        "steric_height_cm_rms": steric_rms,
    }
