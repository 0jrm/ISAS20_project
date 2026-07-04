"""Module C tests: patch encoder."""

from __future__ import annotations

import numpy as np
import torch


def test_c1_localization_neighbor_perturbation(residual_model, synthetic_batch):
    residual_model.eval()
    x = synthetic_batch.clone()
    with torch.no_grad():
        y0 = residual_model.patch_enc(x[:, residual_model.patch_offset :])
    x2 = x.clone()
    x2[:, residual_model.patch_offset + 1] += 3.0
    with torch.no_grad():
        y1 = residual_model.patch_enc(x2[:, residual_model.patch_offset :])
    diff = (y0 - y1).abs().max().item()
    assert diff > 1e-6


def test_c2_center_invariance_constant_offset(residual_model, synthetic_batch):
    residual_model.eval()
    x = synthetic_batch.clone()
    x[:, residual_model.patch_offset :] = 0.0
    with torch.no_grad():
        y0 = residual_model.forward_delta(x)
    x2 = x.clone()
    x2[:, residual_model.patch_offset :] += 5.0
    with torch.no_grad():
        y1 = residual_model.forward_delta(x2)
    np.testing.assert_allclose(y0.numpy(), y1.numpy(), atol=1e-5)


def test_c3_gradient_sensitivity(residual_model, base_dim, patch_flat, patch_shape):
    c, t, h, w = patch_shape
    n = 2
    x = torch.zeros(n, base_dim + patch_flat)
    flat = x[:, base_dim:]
    vol = flat.view(n, c, t, h, w)
    with torch.no_grad():
        y_flat = residual_model.patch_enc(x[:, base_dim:])
    gx, gy = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij")
    grad_plane = gx + gy
    vol[:, 2, :, :, :] = grad_plane
    x_grad = x.clone()
    x_grad[:, base_dim:] = vol.reshape(n, -1)
    with torch.no_grad():
        y_grad = residual_model.patch_enc(x_grad[:, base_dim:])
    assert (y_flat - y_grad).abs().max().item() > 1e-6
