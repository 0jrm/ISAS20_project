"""Module E tests: trainer stages (lightweight)."""

from __future__ import annotations

import torch

from model.model import ResidualPatchModel


def test_e1_epoch0_matches_point_baseline(residual_model, synthetic_batch):
    residual_model.eval()
    with torch.no_grad():
        base = residual_model.forward_base(synthetic_batch)
        out = residual_model(synthetic_batch)
    rel_diff = (out - base).abs().mean() / (base.abs().mean() + 1e-8)
    assert rel_diff.item() < 0.001


def test_e2_no_exploding_gradients(residual_model, synthetic_batch):
    residual_model.train()
    opt = torch.optim.Adam(residual_model.parameters(), lr=0.01)
    norms = []
    for _ in range(5):
        opt.zero_grad()
        out = residual_model(synthetic_batch)
        loss = out.pow(2).mean()
        loss.backward()
        total = 0.0
        for p in residual_model.parameters():
            if p.grad is not None:
                total += p.grad.norm().item() ** 2
        norms.append(total ** 0.5)
        opt.step()
    assert max(norms) < 1e4


def test_e3_residual_training_changes_output(residual_model, synthetic_batch):
    residual_model.train()
    opt = torch.optim.Adam(
        [p for p in residual_model.parameters() if p.requires_grad],
        lr=0.05,
    )
    target = torch.randn(synthetic_batch.shape[0], 32)
    with torch.no_grad():
        before = residual_model(synthetic_batch).clone()
    for _ in range(20):
        opt.zero_grad()
        out = residual_model(synthetic_batch)
        loss = (out - target).pow(2).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        after = residual_model(synthetic_batch)
    assert (after - before).abs().mean().item() > 1e-4
