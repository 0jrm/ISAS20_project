"""Heave + warped-residual head: predict (MLD, D26, stretch) and residual PCs on the canonical z-grid."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_model import BaseModel
from model.warp import CANON_D26_M, CANON_MLD_M, MIN_LAYER_M


class HeaveResidual(BaseModel):
    """PatchConvMLP backbone; output layout ``[warp (3), T residual PCs, S residual PCs]``.

    Warp raw → physical depths in :func:`decode_warp`. Residual PCs live on the
    canonical grid — not z-level PCA scores.
    """

    def __init__(
        self,
        input_dim=11,
        output_dim=35,
        n_warp=3,
        dropout_prob=0.2,
        d_model=128,
        head_layers=None,
        patch_shape=None,
        n_enc=8,
        n_sat=3,
        probabilistic=False,
        sigma_min=1e-3,
        n_quantiles=0,
        **kwargs,
    ):
        super().__init__()
        from model.model import PatchConvMLP

        self.n_warp = int(n_warp)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.probabilistic = bool(probabilistic)
        self.backbone = PatchConvMLP(
            input_dim=input_dim,
            output_dim=output_dim,
            dropout_prob=dropout_prob,
            d_model=d_model,
            head_layers=head_layers,
            patch_shape=patch_shape,
            n_enc=n_enc,
            n_sat=n_sat,
            probabilistic=probabilistic,
            sigma_min=sigma_min,
            n_quantiles=n_quantiles,
        )

    def forward(self, x):
        return self.backbone(x)

    def set_sigma_trainable(self, trainable: bool) -> None:
        self.backbone.set_sigma_trainable(trainable)


def decode_warp(raw: torch.Tensor, z_bot: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map unbounded warp logits to (mld, d26, stretch). ``raw`` (B, ≥3) or (B, 3)."""
    mld = CANON_MLD_M + 40.0 * torch.tanh(raw[..., 0])
    mld = mld.clamp(MIN_LAYER_M, z_bot - 2.0 * MIN_LAYER_M)
    d26 = mld + MIN_LAYER_M + F.softplus(raw[..., 1] + 1.5)
    d26 = d26.clamp_max(z_bot - MIN_LAYER_M)
    stretch = 1.0 + 0.3 * torch.tanh(raw[..., 2] if raw.shape[-1] > 2 else raw[..., 1] * 0.0)
    return mld, d26, stretch
