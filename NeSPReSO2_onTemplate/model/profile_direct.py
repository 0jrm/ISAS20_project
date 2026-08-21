"""Native-z profile heads: learned latent decoder vs direct T(z),S(z) + depth filter."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_model import BaseModel


def binomial_kernel1d(k: int = 5, device=None, dtype=None) -> torch.Tensor:
    """Odd-length binomial smoother. ponytail: fixed FIR; upgrade = learned depthwise conv."""
    if k % 2 == 0:
        raise ValueError("filter kernel must be odd")
    n = k - 1
    row = [1]
    for i in range(n):
        row.append(row[-1] * (n - i) // (i + 1))
    w = torch.tensor(row, dtype=dtype or torch.float32, device=device)
    return (w / w.sum()).view(1, 1, k)


def depth_filter(ts: torch.Tensor, k: int = 5) -> torch.Tensor:
    """``ts`` (B, 2, n_z) → same shape, replicate-pad binomial smooth along z."""
    b, c, n = ts.shape
    w = binomial_kernel1d(k, device=ts.device, dtype=ts.dtype).repeat(c, 1, 1)
    pad = k // 2
    x = F.pad(ts, (pad, pad), mode="replicate")
    return F.conv1d(x, w, groups=c)


class LatentProfileDecoder(BaseModel):
    """PatchConvMLP → 32 latent scores → learned Linear decode to T(z), S(z). Not PCA V."""

    def __init__(
        self,
        input_dim=11,
        output_dim=3602,
        n_z=1801,
        n_latent=32,
        n_t=16,
        n_s=16,
        dropout_prob=0.2,
        d_model=128,
        head_layers=None,
        patch_shape=None,
        n_enc=8,
        n_sat=3,
        **kwargs,
    ):
        super().__init__()
        from model.model import PatchConvMLP

        self.n_z = int(n_z)
        self.n_t = int(n_t)
        self.n_s = int(n_s)
        self.n_latent = int(n_latent)
        if self.n_t + self.n_s != self.n_latent:
            raise ValueError("n_t + n_s must equal n_latent")
        if int(output_dim) != 2 * self.n_z:
            raise ValueError(f"output_dim {output_dim} != 2*n_z {2 * self.n_z}")
        self.backbone = PatchConvMLP(
            input_dim=input_dim,
            output_dim=self.n_latent,
            dropout_prob=dropout_prob,
            d_model=d_model,
            head_layers=head_layers,
            patch_shape=patch_shape,
            n_enc=n_enc,
            n_sat=n_sat,
            probabilistic=False,
        )
        self.t_dec = nn.Linear(self.n_t, self.n_z)
        self.s_dec = nn.Linear(self.n_s, self.n_z)

    def forward(self, x):
        z = self.backbone(x)
        t = self.t_dec(z[:, : self.n_t])
        s = self.s_dec(z[:, self.n_t :])
        return torch.cat([t, s], dim=1)


class ProfileDirect(BaseModel):
    """Predict full T(z), S(z) then a fixed full-profile smoother."""

    def __init__(
        self,
        input_dim=11,
        output_dim=3602,
        n_z=1801,
        filter_k=5,
        dropout_prob=0.2,
        d_model=128,
        head_layers=None,
        patch_shape=None,
        n_enc=8,
        n_sat=3,
        **kwargs,
    ):
        super().__init__()
        from model.model import PatchConvMLP

        self.n_z = int(n_z)
        self.filter_k = int(filter_k)
        if int(output_dim) != 2 * self.n_z:
            raise ValueError(f"output_dim {output_dim} != 2*n_z {2 * self.n_z}")
        self.backbone = PatchConvMLP(
            input_dim=input_dim,
            output_dim=2 * self.n_z,
            dropout_prob=dropout_prob,
            d_model=d_model,
            head_layers=head_layers,
            patch_shape=patch_shape,
            n_enc=n_enc,
            n_sat=n_sat,
            probabilistic=False,
        )

    def forward(self, x):
        raw = self.backbone(x)
        ts = raw.view(raw.size(0), 2, self.n_z)
        ts = depth_filter(ts, k=self.filter_k)
        return ts.reshape(raw.size(0), 2 * self.n_z)
