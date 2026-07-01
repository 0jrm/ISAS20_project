#!/usr/bin/env python3
"""Verify ARGO L4 patch input layout and model forward pass."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import numpy as np
import torch

from model.model import PatchMaskConvMLP
from parse_config import validate_config
from base.util import read_json
from preproc.l3_input import sync_arch_with_io
from preproc.preproc_isas_sat import build_argo_l4_input_matrix, resolve_use_mask_channels


def main() -> int:
    cfg_path = _ROOT / "config/argo/config_argo_patch_l4_smoke.json"
    cfg = read_json(str(cfg_path))
    validate_config(cfg)
    dim = sync_arch_with_io(cfg)
    use_mask = resolve_use_mask_channels(cfg)
    spatial_pad = int(cfg["io"]["spatial_pad"])
    temporal_pad = int(cfg["io"]["temporal_pad"])
    n = 4
    juld = np.linspace(100, 200, n, dtype=np.float32)
    lat = np.full(n, 25.0, dtype=np.float32)
    lon = np.full(n, -90.0, dtype=np.float32)
    per = (2 * spatial_pad + 1) ** 2 * (temporal_pad + 1)
    sat_vars = {
        "sss": np.random.randn(n, per).astype(np.float32),
        "sst": np.random.randn(n, per).astype(np.float32) + 300,
        "ssh": np.random.randn(n, per).astype(np.float32),
    }
    basin = {"sss": np.zeros(n), "sst": np.full(n, 300.0), "ssh": np.zeros(n)}
    bathy = np.full(n, 1200.0, dtype=np.float32)
    x = build_argo_l4_input_matrix(
        juld,
        lat,
        lon,
        sat_vars,
        cfg["input_params"],
        basin_means=basin,
        bathy_depth=bathy,
        use_mask_channels=use_mask,
    )
    assert x.shape == (n, dim), x.shape
    model = PatchMaskConvMLP(**cfg["arch"]["args"])
    sat_flat = torch.tensor(x[:, model.n_enc :], dtype=torch.float32)
    sat_vol = model._unpack_sat_channels(sat_flat)
    c, t, h, w = model.patch_shape
    assert sat_vol.shape == (n, c, t, h, w), sat_vol.shape
    if use_mask:
        for ch in (1, 3, 5):
            u = torch.unique(sat_vol[0, ch])
            assert set(u.tolist()) <= {0.0, 1.0}, f"mask channel {ch} not binary: {u[:8].tolist()}"
    y = model(torch.tensor(x))
    assert y.shape == (n, 32), y.shape
    print(
        f"OK argo_l4 layout dim={dim} use_mask_channels={use_mask} "
        f"patch_shape={list(model.patch_shape)} forward={tuple(y.shape)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
