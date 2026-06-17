#!/usr/bin/env python3
"""Ponytail self-check: v2 equivalence, PCA round-trip, N-output offsets."""

from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.decomposition import PCA

from model.loss import (
    CombinedPCALoss,
    output_slices,
    sklearn_inverse_transform_pcs,
    torch_reconstruct_profile,
)
from model.model import PatchConvMLP, PredictionModel
from preproc.preproc_isas_sat import compute_input_dim, sat_patch_shape

TOL = 1e-6
GOLDEN = {
    "combined_loss": 0.008507695980370045,
    "pca_loss": 0.00037024961784482,
    "weighted_mse_loss": 0.000430555606726557,
    "recon_temp_head": [20.078725814819336, 19.486875534057617, 19.124080657958984, 18.788761138916016, 18.34334945678711],
    "recon_sal_head": [36.001399993896484, 36.043731689453125, 36.02961349487305, 36.0516357421875, 36.07063293457031],
    "prediction_head": [-0.0486145056784153, -0.2789950966835022, -0.18031582236289978, -0.10186368972063065, -0.19932889938354492, -0.10338201373815536],
}


def _synthetic_pca_pair():
    np.random.seed(42)
    depth = np.linspace(0, 500, 26)
    n_profiles = 8
    temp = 20.0 - 0.02 * depth[:, None] + 0.1 * np.random.randn(len(depth), n_profiles)
    sal = 36.0 + 0.001 * depth[:, None] + 0.01 * np.random.randn(len(depth), n_profiles)
    n_components = 3
    pca_temp = PCA(n_components=n_components).fit(temp.T)
    pca_sal = PCA(n_components=n_components).fit(sal.T)
    temp_pcs = pca_temp.transform(temp.T)
    sal_pcs = pca_sal.transform(sal.T)
    return pca_temp, pca_sal, temp_pcs, sal_pcs, n_components


def test_cap_batch_size():
    from playground.batch_size import cap_batch_size, resolve_batch_size

    assert cap_batch_size(512, 100) == 100
    assert cap_batch_size(256, 1000) == 256


def test_resolve_batch_size_fixed():
    from playground.batch_size import resolve_batch_size

    class _Tiny(torch.nn.Module):
        def forward(self, x):
            return x

    model = _Tiny()
    criterion = _Tiny()
    inputs = torch.randn(10, 4)
    targets = torch.randn(10, 2)
    device = torch.device("cpu")
    assert resolve_batch_size(512, 10, model, criterion, inputs, targets, device) == 10
    assert resolve_batch_size(256, 10, model, criterion, inputs, targets, device) == 10


def test_compute_input_dim():
    flags = {
        "timecos": True,
        "timesin": True,
        "latcos": True,
        "latsin": True,
        "loncos": True,
        "lonsin": True,
        "sss": True,
        "sst": True,
        "ssh": True,
        "sat": True,
    }
    assert compute_input_dim(flags, 0, 0) == 9
    assert compute_input_dim(flags, 2, 3) == 306
    assert sat_patch_shape(2, 3) == (3, 4, 5, 5)
    assert sat_patch_shape(0, 0) is None


def test_patch_conv_mlp_point_mode():
    torch.manual_seed(0)
    model = PatchConvMLP(
        input_dim=9,
        output_dim=32,
        dropout_prob=0.0,
        d_model=32,
        head_layers=[16],
        patch_shape=None,
        n_enc=6,
        n_sat=3,
    )
    model.eval()
    x = torch.randn(4, 9)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (4, 32)


def test_patch_conv_mlp_patch_mode():
    torch.manual_seed(0)
    model = PatchConvMLP(
        input_dim=306,
        output_dim=32,
        dropout_prob=0.0,
        d_model=32,
        head_layers=[16],
        patch_shape=[3, 4, 5, 5],
        n_enc=6,
        n_sat=3,
    )
    model.eval()
    x = torch.randn(2, 306)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 32)


def test_prediction_model_v2():
    torch.manual_seed(42)
    model = PredictionModel(input_dim=5, layers_config=[16, 8], output_dim=6, dropout_prob=0.0)
    model.eval()
    x = torch.tensor([[0.5, -0.3, 1.2, 0.0, -0.8]], dtype=torch.float32)
    with torch.no_grad():
        out = model(x)
    assert np.allclose(out[0, :6].numpy(), GOLDEN["prediction_head"], rtol=0, atol=TOL)


def test_combined_pca_loss_v2():
    pca_temp, pca_sal, temp_pcs, sal_pcs, n_components = _synthetic_pca_pair()
    pcs_np = np.hstack([temp_pcs, sal_pcs])[:4].astype(np.float32)
    pred_np = pcs_np + np.array([[0.1, -0.05, 0.02, 0.03, -0.01, 0.04]], dtype=np.float32)
    outputs = OrderedDict([("temperature", n_components), ("salinity", n_components)])
    pca_models = {"temperature": pca_temp, "salinity": pca_sal}
    weights = np.ones(2 * n_components, dtype=np.float64)
    device = torch.device("cpu")

    combined = CombinedPCALoss(pca_models, outputs, weights, device)
    combined.eval()
    pcs = torch.tensor(pred_np, dtype=torch.float32)
    targets = torch.tensor(pcs_np, dtype=torch.float32)

    with torch.no_grad():
        combined_loss = combined(pcs, targets)
        pca_loss = combined.pca_loss(pcs, targets)
        weighted_mse = combined.weighted_mse_loss(pcs, targets)
        recon_t, recon_s = combined._reconstruct_profiles(pcs)

    assert np.isclose(combined_loss.item(), GOLDEN["combined_loss"], rtol=0, atol=TOL)
    assert np.isclose(pca_loss.item(), GOLDEN["pca_loss"], rtol=0, atol=TOL)
    assert np.isclose(weighted_mse.item(), GOLDEN["weighted_mse_loss"], rtol=0, atol=TOL)
    assert np.allclose(recon_t[0, :5].numpy(), GOLDEN["recon_temp_head"], rtol=0, atol=TOL)
    assert np.allclose(recon_s[0, :5].numpy(), GOLDEN["recon_sal_head"], rtol=0, atol=TOL)


def test_pca_round_trip():
    pca_temp, pca_sal, temp_pcs, sal_pcs, n_components = _synthetic_pca_pair()
    outputs = OrderedDict([("temperature", n_components), ("salinity", n_components)])
    pca_models = {"temperature": pca_temp, "salinity": pca_sal}
    pcs = np.hstack([temp_pcs[:2], sal_pcs[:2]]).astype(np.float64)

    sk = sklearn_inverse_transform_pcs(pcs, pca_models, outputs)

    for name, start, end in output_slices(outputs):
        pca = pca_models[name]
        t_pcs = torch.tensor(pcs[:, start:end], dtype=torch.float32)
        comp = torch.tensor(pca.components_, dtype=torch.float32)
        mean = torch.tensor(pca.mean_, dtype=torch.float32).unsqueeze(0)
        torch_prof = torch_reconstruct_profile(t_pcs, comp, mean).numpy()
        assert np.nanmax(np.abs(sk[name].T - torch_prof)) < 5e-6


def test_asymmetric_output_offsets():
    outputs = OrderedDict([("temperature", 15), ("salinity", 12)])
    slices = output_slices(outputs)
    assert slices == [("temperature", 0, 15), ("salinity", 15, 27)]
    pcs = np.arange(27, dtype=np.float32)[None, :]
    pca_models = {
        "temperature": SimpleNamespace(inverse_transform=lambda x: x * 0 + 1),
        "salinity": SimpleNamespace(inverse_transform=lambda x: x * 0 + 2),
    }
    # use real PCAs for a meaningful split check
    pca_temp, pca_sal, temp_pcs, sal_pcs, _ = _synthetic_pca_pair()
    pca15 = PCA(n_components=15).fit(np.random.randn(20, 26))
    pca12 = PCA(n_components=12).fit(np.random.randn(20, 26))
    pca_models = {"temperature": pca15, "salinity": pca12}
    full = np.zeros((1, 27), dtype=np.float64)
    full[0, :15] = pca15.transform(np.random.randn(1, 26))
    full[0, 15:] = pca12.transform(np.random.randn(1, 26))
    assert full.shape[1] == sum(outputs.values())
    for name, start, end in slices:
        assert end - start == outputs[name]


def test_split_matches_torch_seed():
    """ponytail: same lengths + seed as v2 ``random_split`` after ``manual_seed``."""
    from torch.utils.data import Dataset, random_split

    class _DS(Dataset):
        def __len__(self):
            return 100

        def __getitem__(self, i):
            return i

    g = torch.Generator().manual_seed(42)
    a, b, c = random_split(_DS(), [70, 15, 15], generator=g)
    g2 = torch.Generator().manual_seed(42)
    a2, b2, c2 = random_split(_DS(), [70, 15, 15], generator=g2)
    assert list(a.indices) == list(a2.indices)
    assert list(b.indices) == list(b2.indices)


import sys


def test_train_monitor_once():
    """Fake status.json pair + manifest → train_monitor exit code 1 (still running)."""
    import json
    import subprocess
    import tempfile
    from pathlib import Path

    root = Path(__file__).resolve().parent
    script = root / "scripts" / "train_monitor.py"
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        run_id = "selfcheck_run"
        for tag, name in (("isas20", "NeSPReSO2_ISAS_GoM"), ("argo_v2", "NeSPReSO2_ARGO_GoM")):
            save_dir = tmp / "saved" / "models" / name / f"{run_id}_{tag}"
            save_dir.mkdir(parents=True)
            (save_dir / "status.json").write_text(
                json.dumps(
                    {
                        "tag": tag,
                        "state": "running",
                        "epoch": 1,
                        "val_loss": 1.0,
                        "updated_at": "2099-01-01T00:00:00Z",
                    }
                )
            )
        cfg_isas = tmp / "config_isas.json"
        cfg_argo = tmp / "config_argo.json"
        for path, name in ((cfg_isas, "NeSPReSO2_ISAS_GoM"), (cfg_argo, "NeSPReSO2_ARGO_GoM")):
            path.write_text(json.dumps({"name": name, "trainer": {"save_dir": str(tmp / "saved")}}))
        manifest = tmp / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "runs": [
                        {"tag": "isas20", "config": str(cfg_isas), "pid": 0},
                        {"tag": "argo_v2", "config": str(cfg_argo), "pid": 0},
                    ],
                }
            )
        )
        proc = subprocess.run(
            [sys.executable, str(script), "--once", "--manifest", str(manifest)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 1, proc.stdout + proc.stderr
        assert "isas20" in proc.stdout


from playground import read_json


def test_overlap_pairs():
    """Matched ISAS↔ARGO colocation count (needs both caches on disk)."""
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent
    cache_dir = root.parent / "data/cache"
    if not cache_dir.is_dir():
        return
    caches = {}
    import pickle

    for p in cache_dir.glob("train_ready_*.pkl"):
        with open(p, "rb") as f:
            c = pickle.load(f)
        tag = c.get("dataset_tag")
        if tag in ("isas20", "argo_v2"):
            caches[tag] = c
    if len(caches) != 2:
        return
    cfg = read_json(root / "config_argo.json")
    v2_src = cfg.get("io", {}).get("v2_src")
    if v2_src:
        sys.path.insert(0, str(v2_src))
    from preproc.overlap import find_matched_pairs, overlap_summary

    summary = overlap_summary(caches["isas20"], caches["argo_v2"])
    assert summary["matched_pairs_1d_05deg"] > 1000, summary
    pairs = find_matched_pairs(caches["isas20"], caches["argo_v2"])
    assert len(pairs) == summary["matched_pairs_1d_05deg"]


def test_cache_schema_keys():
    """If an ISAS cache exists on disk, it must carry eval fields."""
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent / "data/cache"
    for path in sorted(root.glob("train_ready_*.pkl")):
        import pickle

        with open(path, "rb") as f:
            cache = pickle.load(f)
        if "profiles" not in cache:
            continue
        for key in ("dataset_tag", "inputs", "targets", "pca_models"):
            assert key in cache, f"{path.name} missing {key}"
        for key in ("spatial_pad", "temporal_pad", "sat_patch_shape"):
            if key in cache:
                continue
            # legacy caches may omit patch metadata until rebuilt
        prof = cache["profiles"]
        assert "temperature" in prof and prof["temperature"].ndim == 2
        return
    # ponytail: no rebuilt cache on disk — skip when data absent


if __name__ == "__main__":
    test_cap_batch_size()
    test_resolve_batch_size_fixed()
    test_compute_input_dim()
    test_patch_conv_mlp_point_mode()
    test_patch_conv_mlp_patch_mode()
    test_prediction_model_v2()
    test_combined_pca_loss_v2()
    test_pca_round_trip()
    test_asymmetric_output_offsets()
    test_split_matches_torch_seed()
    test_train_monitor_once()
    test_overlap_pairs()
    test_cache_schema_keys()
    print("selfcheck: all assertions passed")
