"""Landmark vertical registration: mixed-layer + D26 heave, then unwarp.

Canonical knots: surface, MLD0, D26_0, bottom. Physical knots: 0, MLD, D26, bottom.
``T_canon(z) = T_phys(z_phys(z))``; reconstruct by the inverse map.
"""

from __future__ import annotations

import numpy as np
import torch

CANON_MLD_M = 50.0
CANON_D26_M = 120.0
MIN_LAYER_M = 5.0


def _as_1d(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(-1)


def _ordered_knots(mld: float, d26: float, z_bot: float) -> tuple[np.ndarray, np.ndarray]:
    z_bot = max(float(z_bot), CANON_D26_M + MIN_LAYER_M)
    mld = float(np.clip(mld, MIN_LAYER_M, z_bot - 2.0 * MIN_LAYER_M))
    d26 = float(np.clip(d26, mld + MIN_LAYER_M, z_bot - MIN_LAYER_M))
    phys = np.array([0.0, mld, d26, z_bot], dtype=np.float64)
    canon = np.array([0.0, CANON_MLD_M, CANON_D26_M, z_bot], dtype=np.float64)
    return phys, canon


def phys_from_canon(z_canon: np.ndarray, mld: float, d26: float, z_bot: float) -> np.ndarray:
    phys, canon = _ordered_knots(mld, d26, z_bot)
    return np.interp(_as_1d(z_canon), canon, phys)


def canon_from_phys(z_phys: np.ndarray, mld: float, d26: float, z_bot: float) -> np.ndarray:
    phys, canon = _ordered_knots(mld, d26, z_bot)
    return np.interp(_as_1d(z_phys), phys, canon)


def warp_to_canonical(T: np.ndarray, z: np.ndarray, mld: np.ndarray, d26: np.ndarray) -> np.ndarray:
    """Sample physical profiles onto the canonical z-grid (same numeric nodes as ``z``)."""
    T = np.asarray(T, dtype=np.float64)
    z = _as_1d(z)
    mld = _as_1d(mld)
    d26 = _as_1d(d26)
    n, nz = T.shape
    z_bot = float(z[-1])
    out = np.empty_like(T)
    for i in range(n):
        z_p = phys_from_canon(z, float(mld[i]), float(d26[i]), z_bot)
        out[i] = np.interp(z_p, z, T[i], left=np.nan, right=np.nan)
    return out


def unwarp_from_canonical(T_canon: np.ndarray, z: np.ndarray, mld: np.ndarray, d26: np.ndarray) -> np.ndarray:
    """Map canonical-grid profiles back to physical z."""
    T_canon = np.asarray(T_canon, dtype=np.float64)
    z = _as_1d(z)
    mld = _as_1d(mld)
    d26 = _as_1d(d26)
    n, nz = T_canon.shape
    z_bot = float(z[-1])
    out = np.empty_like(T_canon)
    for i in range(n):
        z_c = canon_from_phys(z, float(mld[i]), float(d26[i]), z_bot)
        out[i] = np.interp(z_c, z, T_canon[i], left=np.nan, right=np.nan)
    return out


def _interp1d(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """``x`` (Z,), ``xp`` (K,) increasing, ``fp`` (K,) → (Z,)."""
    idx = torch.searchsorted(xp, x.contiguous(), right=True).clamp(1, xp.numel() - 1)
    x0, x1 = xp[idx - 1], xp[idx]
    f0, f1 = fp[idx - 1], fp[idx]
    w = (x - x0) / (x1 - x0).clamp_min(1e-8)
    return f0 + w * (f1 - f0)


def torch_ordered_knots(mld: torch.Tensor, d26: torch.Tensor, z_bot: float) -> tuple[torch.Tensor, torch.Tensor]:
    z_bot = max(float(z_bot), CANON_D26_M + MIN_LAYER_M)
    mld = mld.clamp(MIN_LAYER_M, z_bot - 2.0 * MIN_LAYER_M)
    d26 = torch.maximum(mld + MIN_LAYER_M, d26).clamp_max(z_bot - MIN_LAYER_M)
    b = mld.shape[0]
    zeros = torch.zeros(b, device=mld.device, dtype=mld.dtype)
    bots = torch.full((b,), z_bot, device=mld.device, dtype=mld.dtype)
    phys = torch.stack([zeros, mld, d26, bots], dim=1)
    canon = torch.tensor(
        [0.0, CANON_MLD_M, CANON_D26_M, z_bot], device=mld.device, dtype=mld.dtype
    ).unsqueeze(0).expand(b, -1)
    return phys, canon


def torch_warp_to_canonical(
    T: torch.Tensor, z: torch.Tensor, mld: torch.Tensor, d26: torch.Tensor
) -> torch.Tensor:
    z = z.reshape(-1)
    z_bot = float(z[-1].item())
    phys, canon = torch_ordered_knots(mld, d26, z_bot)
    # ponytail: per-row 4-knot interp; ceiling = batched searchsorted, upgrade if B*Z dominates step
    out = []
    for i in range(T.shape[0]):
        z_p = _interp1d(z, canon[i], phys[i])
        out.append(_interp1d(z_p, z, T[i]))
    return torch.stack(out, dim=0)


def torch_unwarp_from_canonical(
    T_canon: torch.Tensor, z: torch.Tensor, mld: torch.Tensor, d26: torch.Tensor
) -> torch.Tensor:
    z = z.reshape(-1)
    z_bot = float(z[-1].item())
    phys, canon = torch_ordered_knots(mld, d26, z_bot)
    out = []
    for i in range(T_canon.shape[0]):
        z_c = _interp1d(z, phys[i], canon[i])
        out.append(_interp1d(z_c, z, T_canon[i]))
    return torch.stack(out, dim=0)
