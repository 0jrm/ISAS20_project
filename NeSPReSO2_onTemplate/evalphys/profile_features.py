"""Named 1 m Savitzky–Golay smoother and T-shape features for Argo, xb, and NeSPReSO."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import savgol_filter

from evalphys.metrics import isotherm_depth, mixed_layer_depth

SMOOTHER_NAME = "interp1m_savgol_w11_p2"
SMOOTHER_WINDOW = 11
SMOOTHER_POLY = 2
SMOOTH_Z_MAX_M = 400.0
SMOOTH_DZ_M = 1.0
RULE_VERSION = "shape-v1"


def fine_depth_grid(z_max_m: float = SMOOTH_Z_MAX_M, dz_m: float = SMOOTH_DZ_M) -> np.ndarray:
    return np.arange(0.0, z_max_m + 0.5 * dz_m, dz_m, dtype=np.float64)


def smooth_temperature(
    depth: np.ndarray,
    temperature: np.ndarray,
    valid: np.ndarray | None = None,
    *,
    z_max_m: float = SMOOTH_Z_MAX_M,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate each profile to 1 m and apply Savitzky–Golay.

    ``temperature`` is (n_profiles, n_levels). Returns ``(z_fine, T_smooth)``.
    """
    z = np.asarray(depth, dtype=np.float64).reshape(-1)
    t = np.asarray(temperature, dtype=np.float64)
    if t.ndim != 2:
        raise ValueError(f"temperature must be 2-D, got {t.shape}")
    if t.shape[1] != z.size:
        raise ValueError(f"temperature levels {t.shape[1]} != depth {z.size}")
    z_fine = fine_depth_grid(z_max_m)
    n = t.shape[0]
    out = np.full((n, z_fine.size), np.nan, dtype=np.float64)
    if valid is None:
        valid = np.isfinite(t)
    else:
        valid = np.asarray(valid, dtype=bool) & np.isfinite(t)
    half = SMOOTHER_WINDOW // 2
    for i in range(n):
        ok = valid[i] & np.isfinite(z)
        if int(ok.sum()) < 4:
            continue
        zi = z[ok]
        ti = t[i, ok]
        order = np.argsort(zi)
        zi = zi[order]
        ti = ti[order]
        if zi[0] >= z_fine[-1] or zi[-1] <= z_fine[0]:
            continue
        t_lin = np.interp(z_fine, zi, ti, left=np.nan, right=np.nan)
        finite = np.isfinite(t_lin)
        if int(finite.sum()) < SMOOTHER_WINDOW:
            continue
        filled = t_lin.copy()
        filled[~finite] = np.interp(z_fine[~finite], z_fine[finite], t_lin[finite])
        smoothed = savgol_filter(filled, SMOOTHER_WINDOW, SMOOTHER_POLY, mode="interp")
        smoothed[~finite] = np.nan
        # Drop a half-window at open ends so the filter does not invent a front.
        first = int(np.argmax(finite))
        last = int(finite.size - 1 - np.argmax(finite[::-1]))
        smoothed[: first + half] = np.nan
        smoothed[last - half + 1 :] = np.nan
        out[i] = smoothed
    return z_fine, out


def peak_dtdz(z_fine: np.ndarray, t_smooth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Peak cooling rate (°C/m, positive) and its depth."""
    dtdz = np.gradient(t_smooth, z_fine, axis=1)
    cooling = -dtdz
    peak = np.full(t_smooth.shape[0], np.nan, dtype=np.float64)
    z_peak = np.full(t_smooth.shape[0], np.nan, dtype=np.float64)
    for i in range(t_smooth.shape[0]):
        row = cooling[i]
        ok = np.isfinite(row) & np.isfinite(t_smooth[i])
        if not ok.any():
            continue
        k = int(np.nanargmax(np.where(ok, row, -np.inf)))
        peak[i] = float(row[k])
        z_peak[i] = float(z_fine[k])
    return peak, z_peak


def shape_features(
    depth: np.ndarray,
    temperature: np.ndarray,
    valid: np.ndarray | None = None,
    *,
    salinity: np.ndarray | None = None,
    lat: np.ndarray | None = None,
    lon: np.ndarray | None = None,
) -> dict[str, np.ndarray | str]:
    z_fine, t_s = smooth_temperature(depth, temperature, valid)
    z20, _ = isotherm_depth(t_s, z_fine, 20.0)
    z15, _ = isotherm_depth(t_s, z_fine, 15.0)
    z25, _ = isotherm_depth(t_s, z_fine, 25.0)
    peak, z_peak = peak_dtdz(z_fine, t_s)
    thickness = z15 - z25
    out: dict[str, Any] = {
        "smoother": SMOOTHER_NAME,
        "z20_m": z20,
        "z15_m": z15,
        "z25_m": z25,
        "thickness_15_25_m": thickness,
        "peak_dtdz_c_per_m": peak,
        "peak_dtdz_z_m": z_peak,
    }
    if salinity is not None and lat is not None and lon is not None:
        s_on_fine = np.full_like(t_s, np.nan)
        z = np.asarray(depth, dtype=np.float64).reshape(-1)
        s = np.asarray(salinity, dtype=np.float64)
        for i in range(s.shape[0]):
            ok = np.isfinite(s[i]) & np.isfinite(z)
            if int(ok.sum()) < 2:
                continue
            s_on_fine[i] = np.interp(z_fine, z[ok], s[i, ok], left=np.nan, right=np.nan)
        out["mld_m"] = mixed_layer_depth(t_s, s_on_fine, z_fine, lat, lon)
    else:
        out["mld_m"] = np.full(t_s.shape[0], np.nan, dtype=np.float64)
    return out


def band_rmse(
    pred: np.ndarray,
    truth: np.ndarray,
    valid: np.ndarray,
    depth: np.ndarray,
    z_lo: float,
    z_hi: float,
) -> np.ndarray:
    z = np.asarray(depth, dtype=np.float64).reshape(-1)
    in_band = (z >= z_lo) & (z < z_hi)
    err2 = (pred - truth) ** 2
    mask = np.asarray(valid, dtype=bool) & np.isfinite(err2) & in_band[None, :]
    n = pred.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        m = mask[i]
        if m.any():
            out[i] = float(np.sqrt(np.mean(err2[i, m])))
    return out
