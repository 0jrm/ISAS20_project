"""Heave + warped-residual head: predict (MLD, D26, stretch) and residual PCs on the canonical z-grid."""

from __future__ import annotations

import torch

from base.base_model import BaseModel
from model.warp import CANON_D26_M, CANON_MLD_M, MIN_LAYER_M


_GAP0 = CANON_D26_M - CANON_MLD_M - MIN_LAYER_M


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
        spatial_pool=True,
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
            spatial_pool=spatial_pool,
            **kwargs,
        )
        self._zero_warp_mu()

    def _zero_warp_mu(self):
        """Start at canonical MLD/D26 (raw=0). ponytail: tanh floor had vanishing grad."""
        lin = getattr(self.backbone, "mu_out", None)
        if lin is None:
            return
        with torch.no_grad():
            lin.weight[: self.n_warp].zero_()
            if lin.bias is not None:
                lin.bias[: self.n_warp].zero_()

    def forward(self, x):
        return self.backbone(x)

    def set_sigma_trainable(self, trainable: bool) -> None:
        self.backbone.set_sigma_trainable(trainable)


def decode_warp(raw: torch.Tensor, z_bot: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map unbounded warp logits to (mld, d26, stretch). raw=0 → (50 m, 120 m).

    Exp, not tanh: ``50+40*tanh`` floored at 10 m with vanishing gradient.
    """
    mld = CANON_MLD_M * torch.exp(raw[..., 0])
    mld = mld.clamp(MIN_LAYER_M, z_bot - 2.0 * MIN_LAYER_M)
    d26 = mld + MIN_LAYER_M + _GAP0 * torch.exp(raw[..., 1])
    d26 = d26.clamp_max(z_bot - MIN_LAYER_M)
    stretch_raw = raw[..., 2] if raw.shape[-1] > 2 else raw.new_zeros(raw.shape[:-1])
    stretch = 1.0 + 0.3 * torch.tanh(stretch_raw)
    return mld, d26, stretch


def warp_sigma_meters(raw: torch.Tensor, sigma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """σ in metres: d(a e^x)/dx = a e^x. Clamp ignored in jac."""
    mld = CANON_MLD_M * torch.exp(raw[..., 0])
    gap = _GAP0 * torch.exp(raw[..., 1])
    sig_mld = sigma[..., 0] * mld
    sig_d26 = torch.sqrt(sig_mld ** 2 + (sigma[..., 1] * gap) ** 2)
    return sig_mld, sig_d26
