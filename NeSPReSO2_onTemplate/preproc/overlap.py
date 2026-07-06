"""ISAS ↔ ARGO spatiotemporal overlap for matched evaluation."""

from __future__ import annotations

import hashlib
import pickle
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

_EPOCH_1950_ORDINAL = datetime(1950, 1, 1).toordinal()


def days_since_1950(juld: np.ndarray, *, dataset_tag: str) -> np.ndarray:
    """ISAS JULD is already days since 1950-01-01; ARGO stores MATLAB datenum.

    Vectorized affine transform for the ARGO branch — MATLAB datenum ``d`` maps to
    ``datetime.fromordinal(int(d) - 366) + timedelta(days=d % 1)``, which for
    positive ``d`` collapses to ``date.fromordinal(1) + (d - 367)`` days (since
    ``int(d) + d % 1 == d``). Subtracting the 1950-01-01 ordinal gives days-since-1950
    directly, with no per-element datetime construction. Verified bit-identical
    against the old per-element loop on real caches; introduces at most ~1e-11 day
    of extra precision relative to the original's microsecond-rounded ``timedelta``,
    far below the ``dt_max_days``/``grid_deg`` matching tolerances used downstream.
    """
    if dataset_tag == "isas20":
        return juld.astype(np.float64)
    d = np.asarray(juld, dtype=np.float64)
    return d - 366.0 - _EPOCH_1950_ORDINAL


def load_cache(path: str | Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


def find_matched_pairs(
    isas: dict[str, Any],
    argo: dict[str, Any],
    *,
    dt_max_days: float = 1.0,
    grid_deg: float = 0.5,
) -> list[tuple[int, int, float]]:
    """
    Greedy 1:1 match on rounded lat/lon cell + |Δt| <= dt_max_days.
    Returns (isas_idx, argo_idx, dt_days).
    """
    isas_t = days_since_1950(isas["JULD"], dataset_tag="isas20")
    argo_t = days_since_1950(argo["JULD"], dataset_tag="argo_v2")

    def cell(lat, lon):
        la = np.floor(lat / grid_deg) * grid_deg + grid_deg / 2
        lo = np.floor(lon / grid_deg) * grid_deg + grid_deg / 2
        return la, lo

    isas_cells: dict[tuple[float, float], list[int]] = defaultdict(list)
    argo_cells: dict[tuple[float, float], list[int]] = defaultdict(list)
    for i in range(len(isas["LAT"])):
        isas_cells[cell(isas["LAT"][i], isas["LON"][i])].append(i)
    for j in range(len(argo["LAT"])):
        argo_cells[cell(argo["LAT"][j], argo["LON"][j])].append(j)

    candidates: list[tuple[float, int, int]] = []
    for key in set(isas_cells) & set(argo_cells):
        for i in isas_cells[key]:
            for j in argo_cells[key]:
                dt = abs(isas_t[i] - argo_t[j])
                if dt <= dt_max_days:
                    candidates.append((dt, i, j))
    candidates.sort()

    used_i: set[int] = set()
    used_j: set[int] = set()
    pairs: list[tuple[int, int, float]] = []
    for dt, i, j in candidates:
        if i in used_i or j in used_j:
            continue
        used_i.add(i)
        used_j.add(j)
        pairs.append((i, j, dt))
    return pairs


def holdout_mask(pairs: list[tuple[int, int, float]], isas: dict[str, Any], frac: float = 0.15, seed: int = 42) -> np.ndarray:
    """Deterministic eval holdout on matched pairs (independent of per-cache row order)."""
    isas_t = days_since_1950(isas["JULD"], dataset_tag="isas20")
    keep = np.zeros(len(pairs), dtype=bool)
    for k, (i, _j, _dt) in enumerate(pairs):
        key = f"{isas['LAT'][i]:.4f},{isas['LON'][i]:.4f},{isas_t[i]:.1f},{seed}"
        h = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
        keep[k] = (h % 10_000) < int(frac * 10_000)
    return keep


def depth_grid_m(z_min: float = 0.0, z_max: float = 1800.0, step: float = 10.0) -> np.ndarray:
    return np.arange(z_min, z_max + step, step, dtype=np.float64)


def _interp_col(z_dst: np.ndarray, z_src: np.ndarray, fp: np.ndarray) -> np.ndarray:
    """Linear interp dropping non-finite source samples; NaN outside the valid span."""
    finite = np.isfinite(fp)
    if finite.sum() < 2:
        return np.full(len(z_dst), np.nan)
    zs, vs = z_src[finite], fp[finite]
    order = np.argsort(zs)
    out = np.interp(z_dst, zs[order], vs[order], left=np.nan, right=np.nan)
    return out


def interp_profiles(prof: np.ndarray, z_src: np.ndarray, z_dst: np.ndarray) -> np.ndarray:
    """``prof`` (n_z,) or (n_z, n); ``z_src`` 1d — linear interp, NaN outside data span."""
    z_src = np.asarray(z_src, dtype=np.float64)
    if prof.ndim == 1:
        return _interp_col(z_dst, z_src, np.asarray(prof, dtype=np.float64))
    out = np.empty((len(z_dst), prof.shape[1]), dtype=np.float64)
    for col in range(prof.shape[1]):
        out[:, col] = _interp_col(z_dst, z_src, prof[:, col].astype(np.float64))
    return out


def overlap_summary(isas: dict[str, Any], argo: dict[str, Any]) -> dict[str, Any]:
  pairs = find_matched_pairs(isas, argo)
  isas_t = days_since_1950(isas["JULD"], dataset_tag="isas20")
  argo_t = days_since_1950(argo["JULD"], dataset_tag="argo_v2")
  return {
      "n_isas": int(len(isas["LAT"])),
      "n_argo": int(len(argo["LAT"])),
      "matched_pairs_1d_05deg": len(pairs),
      "isas_time_days1950": [float(isas_t.min()), float(isas_t.max())],
      "argo_time_days1950": [float(argo_t.min()), float(argo_t.max())],
      "holdout_15pct": int(holdout_mask(pairs, isas).sum()),
  }
