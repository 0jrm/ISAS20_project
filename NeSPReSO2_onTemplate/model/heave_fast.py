"""Batched-warp copy of HeaveResidual. Same numerics as model.heave / warp.py.

The original ``torch_warp_to_canonical`` loops the batch in Python (ponytail
ceiling in model/warp.py). That dominates ARGO steps (B=512, Z=1801, 4 warps
per loss). This file vectorizes the same searchsorted lerp.
"""

from __future__ import annotations

import torch

from model.heave import HeaveResidual
from model.warp import torch_ordered_knots


class HeaveResidualFast(HeaveResidual):
    """Same weights as HeaveResidual. Pair with HeaveResidualFastLoss."""


def torch_warp_to_canonical_batched(T, z, mld, d26, z_bot=None):
    """Same as ``model.warp.torch_warp_to_canonical``, one searchsorted over B."""
    z = z.reshape(-1)
    if z_bot is None:
        z_bot = float(z[-1].item())
    phys, canon = torch_ordered_knots(mld, d26, z_bot)
    z_p = canon_to_phys(z, phys, canon)
    return lerp_along_z(z_p, z, T)


def torch_unwarp_from_canonical_batched(T_canon, z, mld, d26, z_bot=None):
    """Same as ``model.warp.torch_unwarp_from_canonical``, batched."""
    z = z.reshape(-1)
    if z_bot is None:
        z_bot = float(z[-1].item())
    phys, canon = torch_ordered_knots(mld, d26, z_bot)
    z_c = phys_to_canon(z, phys, canon)
    return lerp_along_z(z_c, z, T_canon)


def canon_to_phys(z, phys, canon):
    """Physical z for each canonical node. canon rows are identical."""
    xp = canon[0]
    idx = torch.searchsorted(xp, z.contiguous(), right=True).clamp(1, xp.numel() - 1)
    x0, x1 = xp[idx - 1], xp[idx]
    w = (z - x0) / (x1 - x0).clamp_min(1e-8)
    return phys[:, idx - 1] + w * (phys[:, idx] - phys[:, idx - 1])


def phys_to_canon(z, phys, canon):
    """Canonical z for each physical node. phys knots vary per row."""
    x = z.reshape(-1).unsqueeze(0).expand(phys.shape[0], -1).contiguous()
    idx = torch.searchsorted(phys, x, right=True).clamp(1, phys.shape[-1] - 1)
    x0 = phys.gather(1, idx - 1)
    x1 = phys.gather(1, idx)
    f0 = canon.gather(1, idx - 1)
    f1 = canon.gather(1, idx)
    return f0 + ((x - x0) / (x1 - x0).clamp_min(1e-8)) * (f1 - f0)


def lerp_along_z(x, z, fp):
    """Sample ``fp`` (B, Z) at coordinates ``x`` (B, Z) on shared grid ``z`` (Z,)."""
    idx = torch.searchsorted(z, x.contiguous(), right=True).clamp(1, z.numel() - 1)
    x0, x1 = z[idx - 1], z[idx]
    w = (x - x0) / (x1 - x0).clamp_min(1e-8)
    return fp.gather(1, idx - 1) + w * (fp.gather(1, idx) - fp.gather(1, idx - 1))
