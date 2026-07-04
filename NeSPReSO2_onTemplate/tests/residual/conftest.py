"""Shared fixtures for residual model tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from model.model import ResidualPatchModel


@pytest.fixture
def spatial_pad():
    return 2


@pytest.fixture
def temporal_pad():
    return 3


@pytest.fixture
def patch_shape():
    return (3, 4, 5, 5)


@pytest.fixture
def base_dim():
    return 9


@pytest.fixture
def patch_flat(patch_shape):
    c, t, h, w = patch_shape
    return c * t * h * w


@pytest.fixture
def residual_model(base_dim, patch_shape, patch_flat):
    return ResidualPatchModel(
        input_dim=base_dim + patch_flat,
        output_dim=32,
        base_dim=base_dim,
        patch_offset=base_dim,
        patch_shape=patch_shape,
        d_model=32,
        conv_channels=[8, 16],
        pool_output=[2, 2, 2],
        head_hidden=32,
        head_depth=1,
        dropout_prob=0.0,
        warmstart_ckpt=None,
    )


@pytest.fixture
def synthetic_batch(base_dim, patch_flat):
    n = 8
    x = torch.randn(n, base_dim + patch_flat)
    return x
