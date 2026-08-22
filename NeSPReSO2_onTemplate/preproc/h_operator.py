"""TSIS ingest H: 1 m T/S -> HYCOM layer means. No HYCOM I/O.

H is not a fixed z table. Interfaces p_ifc come from the background column:

    dp_m, p_ifc = interfaces_m(thknss_column)
    T_k = layer_sample(z_1m, T_1m, p_ifc)

thknss is pressure thickness (Pa). Divide by ONEM=9806 to get metres.
"""
from __future__ import annotations

import json

import numpy as np

ONEM = 9806.0
KDM = 41
SPVAL = 2.0**100
ZMAX = 1800.0


def interfaces_m(dp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dp_m = np.asarray(dp, dtype=np.float64) / ONEM
    dp_m = np.where(np.abs(dp) > 0.5 * SPVAL, 0.0, dp_m)
    p_ifc = np.concatenate([[0.0], np.cumsum(np.maximum(dp_m, 0.0))])
    return dp_m, p_ifc


def layer_sample(y_z, y_val, p_ifc, zmax: float = ZMAX) -> np.ndarray:
    """Mean of 1 m samples in [z_k, z_{k+1}). Interp to layer midpoint if empty."""
    n = len(p_ifc) - 1
    out = np.full(n, np.nan, dtype=np.float64)
    y_z = np.asarray(y_z, dtype=np.float64).ravel()
    y_val = np.asarray(y_val, dtype=np.float64).ravel()
    nuse = min(y_z.size, y_val.size)
    if nuse == 0:
        return out
    y_z, y_val = y_z[:nuse], y_val[:nuse]
    order = np.argsort(y_z)
    y_z, y_val = y_z[order], y_val[order]
    for k in range(n):
        z0, z1 = float(p_ifc[k]), float(p_ifc[k + 1])
        if z1 <= z0 + 1e-6 or z0 >= zmax:
            continue
        z1c = min(z1, zmax)
        m = (y_z >= z0) & (y_z < z1c)
        if m.any():
            out[k] = float(np.mean(y_val[m]))
        else:
            zmid = 0.5 * (z0 + z1c)
            out[k] = float(np.interp(zmid, y_z, y_val))
    return out


def apply_H(z_1m, T_1m, S_1m, p_ifc, zmax: float = ZMAX):
    return layer_sample(z_1m, T_1m, p_ifc, zmax), layer_sample(z_1m, S_1m, p_ifc, zmax)


def load_interfaces(path) -> dict:
    with open(path) as f:
        return json.load(f)


def p_ifc_for_cast(packet, lat, lon, match_deg=0.05):
    # Radius is in lat/lon degrees as stored on the GDAC columns, not km.
    best, best_d = None, None
    for col in packet.get("columns") or []:
        d = float(np.hypot(float(col["lat"]) - float(lat), float(col["lon"]) - float(lon)))
        if best_d is None or d < best_d:
            best, best_d = col, d
    if best is not None and best_d is not None and best_d <= match_deg:
        p_ifc = np.asarray(best["p_ifc_m"], dtype=np.float64)
        zmid = np.asarray(best.get("zmid_m"), dtype=np.float64)
        if zmid.size != p_ifc.size - 1:
            zmid = 0.5 * (p_ifc[:-1] + p_ifc[1:])
        return p_ifc, zmid, f"column:{best['platform']}"
    p_ifc = np.asarray(packet["scorecard_reference_p_ifc"], dtype=np.float64)
    zmid = np.asarray(packet.get("scorecard_reference_zmid"), dtype=np.float64)
    if zmid.size != p_ifc.size - 1:
        zmid = 0.5 * (p_ifc[:-1] + p_ifc[1:])
    return p_ifc, zmid, "scorecard_reference"
