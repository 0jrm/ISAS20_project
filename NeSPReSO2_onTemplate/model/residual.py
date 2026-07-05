"""Point-anchored residual model over named cube features (Component C)."""

from __future__ import annotations

import torch
import torch.nn as nn

from base.base_model import BaseModel
from model.model import PatchConvMLP, _load_warmstart_state


class PointAnchoredResidual(BaseModel):
    """
    ``y = y_point + gate * delta(x_feat)`` with gate init 0 and frozen point backbone.

    Input layout: ``[x_point (base_dim raw), x_feat (feat_dim z-scored)]``.
    """

    def __init__(
        self,
        input_dim=41,
        output_dim=32,
        base_dim=9,
        feat_dim=32,
        feat_offset=9,
        d_model=128,
        head_hidden=(128, 128),
        dropout_prob=0.1,
        gate_per_pc=True,
        freeze_base=True,
        warmstart_ckpt=None,
        n_enc=6,
        n_sat=3,
        base_head_layers=None,
        **kwargs,
    ):
        super().__init__()
        if base_head_layers is None:
            base_head_layers = [1024, 1024]

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.base_dim = int(base_dim)
        self.feat_dim = int(feat_dim)
        self.feat_offset = int(feat_offset)
        self.gate_per_pc = bool(gate_per_pc)

        self.base = PatchConvMLP(
            input_dim=base_dim,
            output_dim=output_dim,
            dropout_prob=dropout_prob,
            d_model=d_model,
            head_layers=base_head_layers,
            patch_shape=None,
            n_enc=n_enc,
            n_sat=n_sat,
        )

        layers: list[nn.Module] = [nn.LayerNorm(feat_dim)]
        prev = feat_dim
        for width in head_hidden:
            layers.extend([nn.Linear(prev, width), nn.GELU(), nn.Dropout(dropout_prob)])
            prev = width
        self.residual_trunk = nn.Sequential(*layers)
        self.residual_out = nn.Linear(prev, output_dim)

        if gate_per_pc:
            self.gate = nn.Parameter(torch.zeros(output_dim))
        else:
            self.gate = nn.Parameter(torch.zeros(1))

        if warmstart_ckpt:
            _load_warmstart_state(self.base, warmstart_ckpt)

        self.set_freeze_base(freeze_base)

    def set_freeze_base(self, freeze: bool = True) -> None:
        for param in self.base.parameters():
            param.requires_grad = not freeze

    def forward_base(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x[:, : self.base_dim])

    def forward_delta(self, x: torch.Tensor) -> torch.Tensor:
        feat = x[:, self.feat_offset : self.feat_offset + self.feat_dim]
        h = self.residual_trunk(feat)
        return self.residual_out(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.forward_base(x)
        delta = self.forward_delta(x)
        if self.gate_per_pc:
            return base + delta * self.gate
        return base + delta * self.gate.squeeze()

    @property
    def gate_l1(self) -> torch.Tensor:
        return self.gate.abs().sum()
