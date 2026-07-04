"""Module B tests: point / base encoder."""

from __future__ import annotations

from pathlib import Path

import torch

from model.model import PatchConvMLP, ResidualPatchModel, _load_warmstart_state


def test_b1_base_input_completeness(residual_model, synthetic_batch):
    x = synthetic_batch
    assert x.shape[1] >= residual_model.base_dim
    base_slice = x[:, : residual_model.base_dim]
    assert base_slice.shape[1] == 9


def test_b2_warmstart_integrity_if_checkpoint_exists():
    ckpt = Path("saved/models/NeSPReSO2_ARGO_GoM/argo16_scales/model_best.pth")
    if not ckpt.is_file():
        import pytest

        pytest.skip("golden point checkpoint not available")
    standalone = PatchConvMLP(input_dim=9, output_dim=32, n_enc=6, n_sat=3, patch_shape=None)
    _load_warmstart_state(standalone, str(ckpt))
    model = ResidualPatchModel(
        input_dim=534,
        output_dim=32,
        base_dim=9,
        patch_offset=9,
        patch_shape=(3, 7, 5, 5),
        warmstart_ckpt=str(ckpt),
    )
    standalone.eval()
    model.eval()
    x = torch.randn(4, 9)
    with torch.no_grad():
        out_standalone = standalone(x)
        out_warm = model.base(x)
        rmse = torch.sqrt(torch.mean((out_standalone - out_warm) ** 2)).item()
    assert rmse < 1e-5


def test_b3_frozen_base_zero_grad(residual_model, synthetic_batch):
    residual_model.set_freeze_base(True)
    residual_model.train()
    residual_model.gate.data.fill_(1.0)
    residual_model.patch_head_out.weight.data.normal_(0, 0.01)
    out = residual_model(synthetic_batch)
    loss = out.pow(2).mean()
    loss.backward()
    base_grad_norm = sum(p.grad.norm().item() for p in residual_model.base.parameters() if p.grad is not None)
    assert base_grad_norm == 0.0
    patch_grad_norm = sum(
        p.grad.norm().item() for p in residual_model.patch_enc.parameters() if p.grad is not None
    )
    assert patch_grad_norm > 0.0
