"""Phase 4 probabilistic helpers: heteroscedastic σ and non-crossing quantiles."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from evalphys.constants import SIGMA_MIN_DEFAULT

QUANTILE_TAUS = tuple(float(x) for x in __import__("numpy").linspace(0.05, 0.95, 9))


def softplus_sigma(raw: torch.Tensor, sigma_min: float = SIGMA_MIN_DEFAULT) -> torch.Tensor:
    return F.softplus(raw) + float(sigma_min)


def inv_softplus_sigma(sigma: torch.Tensor, sigma_min: float = SIGMA_MIN_DEFAULT) -> torch.Tensor:
    """Bias such that softplus_sigma(bias) == sigma. sigma must exceed sigma_min."""
    gap = torch.clamp(sigma - float(sigma_min), min=1e-6)
    return torch.log(torch.expm1(gap))


def noncrossing_quantiles(raw: torch.Tensor) -> torch.Tensor:
    """raw (B, D, Q) → non-crossing quantiles via cumsoftplus on Q."""
    if raw.ndim != 3:
        raise ValueError(f"expected (B,D,Q), got {tuple(raw.shape)}")
    q0 = raw[:, :, :1]
    gaps = F.softplus(raw[:, :, 1:])
    return torch.cat([q0, q0 + torch.cumsum(gaps, dim=-1)], dim=-1)


def pinball_loss(quantiles: torch.Tensor, y: torch.Tensor, taus: tuple[float, ...] = QUANTILE_TAUS) -> torch.Tensor:
    """quantiles (B,D,Q), y (B,D) → mean pinball."""
    if len(taus) != quantiles.shape[-1]:
        raise ValueError(f"taus len {len(taus)} != Q={quantiles.shape[-1]}")
    y = y.unsqueeze(-1)
    err = y - quantiles
    t = torch.tensor(taus, device=quantiles.device, dtype=quantiles.dtype).view(1, 1, -1)
    return torch.mean(torch.maximum(t * err, (t - 1.0) * err))


def beta_nll(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    y: torch.Tensor,
    *,
    beta: float = 0.5,
    sigma_min: float = SIGMA_MIN_DEFAULT,
) -> torch.Tensor:
    """Seitzer β-NLL: sg(σ^{2β}) · [(y−μ)²/(2σ²) + ½ log σ²]."""
    s = torch.clamp(sigma, min=float(sigma_min))
    var = s * s
    nll = (y - mu) ** 2 / (2.0 * var) + 0.5 * torch.log(var)
    w = var.detach() ** float(beta)
    return torch.mean(w * nll)


def split_mu_sigma(output: torch.Tensor, d: int) -> tuple[torch.Tensor, torch.Tensor]:
    if output.shape[-1] != 2 * d:
        raise ValueError(f"heteroscedastic output dim {output.shape[-1]} != 2*{d}")
    return output[..., :d], output[..., d:]
