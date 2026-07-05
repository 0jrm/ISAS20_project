"""S0 sanity: residual model at init reproduces point baseline."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import model.model as module_arch
from base.util import read_json
from parse_config import ConfigParser


ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "config/argo/config_argo_residual_cube.json"
GOLDEN_CKPT = ROOT / "saved/models/NeSPReSO2_ARGO_GoM/argo16_scales/model_best.pth"
POINT_CUBE_CKPT = ROOT / "saved/models/NeSPReSO2_ARGO_GoM/point_cube/model_best.pth"
CUBE = ROOT / "data/cube/gom_cube.zarr"

S0B_ATOL = 0.05


@pytest.fixture(scope="module")
def cache_path():
    if not CUBE.exists():
        pytest.skip("cube not built")
    if not GOLDEN_CKPT.is_file():
        pytest.skip("golden point checkpoint missing")
    from preproc.features.export_feature_cache import build_feature_cache

    cfg = read_json(CFG)
    return build_feature_cache(cfg, force=False)


def test_s3b_residual_init_matches_golden_point(cache_path):
    """S3b reference: residual base matches legacy golden point checkpoint (tight atol)."""
    if not GOLDEN_CKPT.is_file():
        pytest.skip("golden point checkpoint missing")

    config = ConfigParser(read_json(CFG))
    config.config["data_loader"]["args"]["cache_path"] = cache_path
    model = config.init_obj("arch", module_arch)
    model.eval()

    import pickle

    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    inputs = torch.tensor(cache["inputs"], dtype=torch.float32)
    base_only = model.base(inputs[:, : model.base_dim])

    ckpt = torch.load(GOLDEN_CKPT, map_location="cpu", weights_only=False)
    point = module_arch.PatchConvMLP(
        input_dim=9,
        output_dim=32,
        n_enc=6,
        n_sat=3,
        patch_shape=None,
        head_layers=[1024, 1024],
    )
    point.load_state_dict(ckpt["state_dict"], strict=True)
    point.eval()
    point_out = point(inputs[:, : model.base_dim])

    assert torch.allclose(base_only, point_out, atol=1e-5, rtol=1e-4)

    with torch.no_grad():
        full = model(inputs)
    assert torch.allclose(full, base_only, atol=1e-5, rtol=1e-4)
    assert float(model.gate_l1) == 0.0


@pytest.mark.s0b_gate
def test_s0b_residual_init_matches_point_cube(cache_path):
    """S0b gate: residual at init matches cube-trained point baseline within prediction atol."""
    if not POINT_CUBE_CKPT.is_file():
        pytest.skip("point_cube checkpoint missing")

    cfg = read_json(CFG)
    cfg["arch"]["args"]["warmstart_ckpt"] = str(POINT_CUBE_CKPT)
    config = ConfigParser(cfg)
    config.config["data_loader"]["args"]["cache_path"] = cache_path
    model = config.init_obj("arch", module_arch)
    model.eval()

    import pickle

    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    inputs = torch.tensor(cache["inputs"], dtype=torch.float32)
    base_only = model.base(inputs[:, : model.base_dim])

    ckpt = torch.load(POINT_CUBE_CKPT, map_location="cpu", weights_only=False)
    point = module_arch.PatchConvMLP(
        input_dim=9,
        output_dim=32,
        n_enc=6,
        n_sat=3,
        patch_shape=None,
        head_layers=[1024, 1024],
    )
    point.load_state_dict(ckpt["state_dict"], strict=True)
    point.eval()
    point_out = point(inputs[:, : model.base_dim])

    assert torch.allclose(base_only, point_out, atol=S0B_ATOL, rtol=0.0)

    with torch.no_grad():
        full = model(inputs)
    assert torch.allclose(full, base_only, atol=S0B_ATOL, rtol=0.0)
    assert float(model.gate_l1) == 0.0
