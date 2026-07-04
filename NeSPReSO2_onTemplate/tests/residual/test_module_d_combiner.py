"""Module D tests: residual combiner."""

from __future__ import annotations

import torch


def test_d1_output_equals_base_at_init(residual_model, synthetic_batch):
    residual_model.eval()
    with torch.no_grad():
        base = residual_model.forward_base(synthetic_batch)
        out = residual_model(synthetic_batch)
    assert torch.allclose(out, base, atol=1e-8)


def test_d2_residual_norm_near_zero_at_init(residual_model, synthetic_batch):
    residual_model.eval()
    with torch.no_grad():
        delta = residual_model.forward_delta(synthetic_batch)
    norm = delta.norm(dim=1).mean().item()
    assert norm < 1e-6


def test_d3_gate_can_learn_nonzero(residual_model, synthetic_batch):
    residual_model.train()
    residual_model.patch_head_out.weight.data.normal_(0, 0.05)
    opt = torch.optim.Adam(residual_model.parameters(), lr=0.05)
    target = torch.randn_like(residual_model(synthetic_batch))
    for _ in range(30):
        opt.zero_grad()
        out = residual_model(synthetic_batch)
        loss = (out - target).pow(2).mean()
        loss.backward()
        opt.step()
    assert residual_model.gate.detach().abs().max().item() > 1e-6
