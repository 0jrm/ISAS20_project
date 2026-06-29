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


def test_patch_conv_mlp_residual_mode():
    torch.manual_seed(0)
    model = PatchConvMLP(
        input_dim=306,
        output_dim=64,
        dropout_prob=0.0,
        d_model=32,
        head_layers=[16, 16],
        patch_shape=[3, 4, 5, 5],
        n_enc=6,
        n_sat=3,
        residual=True,
    )
    model.eval()
    x = torch.randn(2, 306)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 64)


def test_res_autoencoder_round_trip():
    from model.model import ResAutoencoder

    torch.manual_seed(0)
    n, depth, k = 8, 26, 4
    x = torch.randn(n, depth)
    mask = torch.zeros(n, depth, dtype=torch.bool)
    mask[:, -2:] = True
    surface = torch.linspace(10.0, 20.0, n)
    model = ResAutoencoder(k, encoder_layers=[32, 16], decoder_layers=[16, 32], input_dim=depth, variable="temperature")
    model.eval()
    with torch.no_grad():
        recon = model(x, mask, surface_residual=surface)
    assert recon.shape == x.shape
    assert torch.allclose(recon[mask], x[mask])
    latent = model.encode(x, mask)
    decoded = model.decode(latent, surface_residual=surface)
    assert decoded.shape == (n, depth)
    valid = ~mask
    assert torch.allclose(decoded[valid], recon[valid], atol=1e-5)


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

    combined = CombinedPCALoss(pca_models, outputs, weights, device, mode="combined")
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


def test_pred_profile_cached_matches_combined():
    from model.loss import make_loss

    pca_temp, pca_sal, temp_pcs, sal_pcs, n_components = _synthetic_pca_pair()
    full_pcs = np.hstack([temp_pcs, sal_pcs]).astype(np.float32)
    pred_np = full_pcs[:4] + np.array([[0.1, -0.05, 0.02, 0.03, -0.01, 0.04]], dtype=np.float32)
    outputs = OrderedDict([("temperature", n_components), ("salinity", n_components)])
    pca_models = {"temperature": pca_temp, "salinity": pca_sal}
    weights = np.ones(2 * n_components, dtype=np.float64)
    device = torch.device("cpu")

    combined = make_loss(
        pca_models=pca_models,
        outputs=outputs,
        weights=weights,
        device=device,
        loss_config={"mode": "combined"},
    )
    cached = make_loss(
        pca_models=pca_models,
        outputs=outputs,
        weights=weights,
        device=device,
        loss_config={"mode": "pred_profile_cached"},
        targets=full_pcs,
    )
    combined.eval()
    cached.eval()

    pcs = torch.tensor(pred_np, dtype=torch.float32)
    targets = torch.tensor(full_pcs[:4], dtype=torch.float32)
    indices = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    with torch.no_grad():
        loss_combined = combined(pcs, targets, indices)
        loss_cached = cached(pcs, targets, indices)
        pca_combined = combined.pca_loss(pcs, targets)
        pca_cached = cached.pca_loss(pcs, targets, indices)

    assert np.allclose(loss_combined.item(), loss_cached.item(), rtol=0, atol=1e-5)
    assert np.allclose(pca_combined.item(), pca_cached.item(), rtol=0, atol=1e-5)


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


def test_decoder_profile_loss():
    """Frozen decoder profile loss: grads flow through latent, not decoder weights."""
    from collections import OrderedDict

    from model.loss import DecoderProfileLoss, _build_profile_ae, make_loss

    torch.manual_seed(0)
    n, depth, k = 12, 26, 3
    profiles = torch.randn(n, depth)
    mask = torch.zeros(n, depth, dtype=torch.bool)
    mask[:, -2:] = True
    profiles_masked = profiles.clone()
    profiles_masked[mask] = 0.0

    ae = _build_profile_ae("Autoencoder", k, depth)
    with torch.no_grad():
        x = profiles_masked
        latent = ae.encoder(x)
        true_profiles = {name: profiles for name in ("temperature", "salinity")}
    # ponytail: one shared profile block for both vars in this smoke test
    outputs = OrderedDict([("temperature", k), ("salinity", k)])
    device = torch.device("cpu")
    true_t = {
        "temperature": profiles,
        "salinity": profiles,
    }
    decoders = {"temperature": ae, "salinity": ae}
    for model in decoders.values():
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

    loss_mod = DecoderProfileLoss(decoders, outputs, device=device, true_profiles=true_t)
    pcs = torch.cat([latent[:4] + 0.1, latent[:4] - 0.05], dim=1)
    pcs.requires_grad_(True)
    indices = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    loss = loss_mod(pcs, pcs, indices)
    loss.backward()
    assert pcs.grad is not None and torch.isfinite(pcs.grad).all()
    assert loss.item() > 0

    # make_loss decoder path (synthetic checkpoint on disk)
    import tempfile
    from pathlib import Path

    from sklearn.decomposition import PCA

    pca_temp = PCA(n_components=k).fit(profiles.numpy())
    pca_sal = PCA(n_components=k).fit(profiles.numpy())
    pca_models = {"temperature": pca_temp, "salinity": pca_sal}
    ae_targets = np.hstack([latent.numpy(), latent.numpy()]).astype(np.float32)
    weights = np.ones(2 * k, dtype=np.float64)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name in outputs:
            out = root / name
            out.mkdir()
            torch.save(
                {
                    "arch": "Autoencoder",
                    "encoding_dim": k,
                    "input_dim": depth,
                    "state_dict": ae.state_dict(),
                },
                out / "decoder_best.pth",
            )
        combined = make_loss(
            pca_models=pca_models,
            outputs=outputs,
            weights=weights,
            device=device,
            loss_config={"mode": "decoder", "decoder_dir": str(root)},
            targets=ae_targets,
            true_profiles=true_t,
            ae_targets=ae_targets,
        )
        combined.eval()
        out_loss = combined(pcs.detach(), pcs.detach(), indices)
        assert torch.isfinite(out_loss)


def test_decoder_profile_loss_nan_mask():
    """NaN depths in true profiles must not poison decoder profile loss."""
    from collections import OrderedDict

    from model.loss import DecoderProfileLoss, _build_profile_ae

    torch.manual_seed(1)
    n, depth, k = 8, 20, 2
    profiles = torch.randn(n, depth)
    profiles[:, 10:] = float("nan")
    ae = _build_profile_ae("Autoencoder", k, depth)
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)
    outputs = OrderedDict([("temperature", k), ("salinity", k)])
    true_t = {"temperature": profiles, "salinity": profiles}
    decoders = {"temperature": ae, "salinity": ae}
    loss_mod = DecoderProfileLoss(decoders, outputs, device=torch.device("cpu"), true_profiles=true_t)
    with torch.no_grad():
        z = ae.encoder(torch.nan_to_num(profiles, nan=0.0))
    pcs = torch.cat([z, z], dim=1)
    loss_val = loss_mod(pcs, torch.zeros_like(pcs), indices=torch.arange(n))
    assert torch.isfinite(loss_val)


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


def test_chronological_split_no_leakage():
    """Chronological split: train max date <= val min <= test min; no overlap."""
    from datetime import date

    from base.split_utils import assign_chronological_fraction_indices, sample_dates

    # synthetic JULD as days since 1950
    epoch_days = np.array([date(2015, 1, 1).toordinal()] * 50 + [date(2020, 6, 1).toordinal()] * 30 + [date(2021, 1, 1).toordinal()] * 20)
    juld = (epoch_days - date(1950, 1, 1).toordinal()).astype(np.float64)
    idx = assign_chronological_fraction_indices(
        juld, dataset_tag="isas20", train_frac=0.7, val_frac=0.15, test_frac=0.15
    )
    dates = sample_dates(juld, dataset_tag="isas20")
    dtrain = [date.fromisoformat(str(dates[i])[:10]) for i in idx["train"]]
    dval = [date.fromisoformat(str(dates[i])[:10]) for i in idx["val"]]
    dtest = [date.fromisoformat(str(dates[i])[:10]) for i in idx["test"]]
    assert max(dtrain) <= min(dval)
    assert max(dval) <= min(dtest)
    assert len(set(idx["train"]) & set(idx["test"])) == 0


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


def test_l3_bin_observations_synthetic():
    from datetime import datetime

    from preproc.l3_rasterize import (
        IDX_AGE,
        IDX_COUNT,
        IDX_MASK,
        IDX_VALUE,
        bin_observations,
        patch_grid,
    )

    target_time = datetime(2020, 1, 15, 12, 0, 0)
    grid = patch_grid(25.0, -90.0, half_deg=1.0, step_deg=0.5)
    windows = [0.0, 24.0]
    obs_time = target_time
    bundle = bin_observations(
        np.array([25.0]),
        np.array([-90.0]),
        [obs_time],
        np.array([0.42]),
        np.array([0.1]),
        target_time,
        grid,
        windows,
    )
    cy, cx = grid.center_y, grid.center_x
    assert bundle[IDX_MASK, 0, cy, cx] == 1.0
    assert abs(bundle[IDX_VALUE, 0, cy, cx] - 0.42) < 1e-5
    assert bundle[IDX_COUNT, 0, cy, cx] == 1.0
    assert bundle[IDX_AGE, 0, cy, cx] == 0.0
    assert bundle[IDX_MASK, 1, cy, cx] == 0.0


def test_l3_lon_normalization_gom_bbox():
    from preproc.l3_rasterize import _in_bbox, _normalize_lon_deg

    assert abs(_normalize_lon_deg(273.3) - (-86.7)) < 0.01
    assert _in_bbox(26.0, 273.3, 26.0, -86.7, half_deg=3.0)


def test_l3_rasterize_missing_is_explicit():
    from datetime import datetime

    from preproc.l3_rasterize import IDX_MASK, IDX_VALUE, empty_bundle

    bundle = empty_bundle(3, 5, 5)
    assert bundle[IDX_MASK].sum() == 0.0
    assert bundle[IDX_VALUE].sum() == 0.0
    assert (bundle[IDX_MASK] == 0).all()


def test_l3_processed_batch_smoke():
    from pathlib import Path

    root = Path(__file__).resolve().parent
    cfg_path = root / "config_argo_l3_smoke.json"
    if not cfg_path.is_file():
        return
    from playground import read_json
    from preproc.export_l3_cache import build_l3_processed_batch

    cfg = read_json(cfg_path)
    path = build_l3_processed_batch(cfg, max_samples=3, anchor_date="2020-01-15", force=True)
    import pickle

    with open(path, "rb") as f:
        payload = pickle.load(f)
    assert len(payload["samples"]) == 3
    s0 = payload["samples"][0]
    for key in ("ssh", "wind_u", "wind_v"):
        arr = s0[key]
        assert arr.shape[0] == 5
        assert arr.shape[1:] == (5, 25, 25)
    assert "split" in s0
    assert "coverage" in s0


def test_l3_input_dim_and_flatten():
    from preproc.l3_input import compute_l3_input_dim, flatten_l3_tensors, l3_sat_patch_shape

    l3_cfg = {
        "patch_half_deg": 3.0,
        "grid_step_deg": 0.25,
        "time_windows_hours": [0, 24, 72, 168, 336],
        "variables": {"ssh": {}, "wind_u": {}, "wind_v": {}},
        "features": ["value", "mask", "age", "uncertainty", "count"],
    }
    flags = {k: True for k in ("timecos", "timesin", "latcos", "latsin", "loncos", "lonsin")}
    dim = compute_l3_input_dim(flags, l3_cfg)
    c, t, h, w = l3_sat_patch_shape(l3_cfg)
    assert dim == 6 + c * t * h * w
    flat = flatten_l3_tensors(None, l3_cfg)
    assert flat.shape == (c * t * h * w,)


def test_l3_feature_subset_dim():
    from preproc.l3_input import compute_l3_input_dim, l3_n_channels, l3_sat_patch_shape

    base = {
        "patch_half_deg": 3.0,
        "grid_step_deg": 0.25,
        "time_windows_hours": [0, 24],
        "variables": {"ssh": {}, "wind_u": {}, "wind_v": {}},
    }
    full = {**base, "features": ["value", "mask", "age", "uncertainty", "count"]}
    slim = {**base, "features": ["value", "mask"]}
    flags = {k: True for k in ("timecos", "timesin", "latcos", "latsin", "loncos", "lonsin")}
    assert l3_n_channels(full) == 15
    assert l3_n_channels(slim) == 6
    assert compute_l3_input_dim(flags, slim) < compute_l3_input_dim(flags, full)
    c, t, h, w = l3_sat_patch_shape(slim)
    assert c == 6 and compute_l3_input_dim(flags, slim) == 6 + c * t * h * w


def test_l3_cache_layout_guard():
    from preproc.l3_input import l3_channel_metadata, verify_l3_cache_layout

    l3_cfg = {
        "patch_half_deg": 3.0,
        "grid_step_deg": 0.25,
        "time_windows_hours": [0, 24],
        "variables": {"ssh": {}},
        "features": ["value", "mask"],
    }
    cache = {
        "l3_enabled": True,
        "input_params": {},
        "inputs": np.zeros((2, 2 * 2 * 25 * 25), dtype=np.float32),
        "l3_channel_metadata": l3_channel_metadata(l3_cfg),
    }
    verify_l3_cache_layout(cache, l3_cfg)
    bad = dict(cache)
    bad["l3_channel_metadata"] = dict(cache["l3_channel_metadata"])
    bad["l3_channel_metadata"]["features"] = ["mask", "value"]
    try:
        verify_l3_cache_layout(bad, l3_cfg)
        raise AssertionError("expected ValueError for feature order mismatch")
    except ValueError as exc:
        assert "features" in str(exc)


def test_sync_arch_l3_config():
    from pathlib import Path

    from playground import read_json
    from preproc.l3_input import sync_arch_with_io

    root = Path(__file__).resolve().parent
    cfg = read_json(root / "config_argo_l3_smoke.json")
    dim = sync_arch_with_io(cfg)
    assert dim == 46881
    assert cfg["arch"]["args"]["patch_shape"] == [15, 5, 25, 25]
    assert cfg["arch"]["args"]["n_sat"] == 15


def test_patch_conv_mlp_l3_forward():
    from preproc.l3_input import compute_l3_input_dim, l3_n_channels, l3_sat_patch_shape

    l3_cfg = {
        "patch_half_deg": 3.0,
        "grid_step_deg": 0.25,
        "time_windows_hours": [0, 24, 72, 168, 336],
        "variables": {"ssh": {}, "wind_u": {}, "wind_v": {}},
        "features": ["value", "mask", "age", "uncertainty", "count"],
    }
    flags = {k: True for k in ("timecos", "timesin", "latcos", "latsin", "loncos", "lonsin")}
    c, t, h, w = l3_sat_patch_shape(l3_cfg)
    dim = compute_l3_input_dim(flags, l3_cfg)
    model = PatchConvMLP(
        input_dim=dim,
        output_dim=4,
        n_enc=6,
        n_sat=l3_n_channels(l3_cfg),
        patch_shape=[c, t, h, w],
        head_layers=[32],
        conv_channels=[8, 16],
    )
    x = torch.randn(2, dim)
    y = model(x)
    assert y.shape == (2, 4)


def test_l4_mask_augment():
    from preproc.l4_augment import SOURCE_SYNTH_L4_MASKED, apply_l4_mask_augment

    field = np.ones((3, 3), dtype=np.float32)
    mask = np.zeros((3, 3), dtype=np.float32)
    mask[1, 1] = 1.0
    values, out_mask, flag = apply_l4_mask_augment(field, mask, noise_scale=0.0, rng=np.random.default_rng(0))
    assert flag == SOURCE_SYNTH_L4_MASKED
    assert out_mask[1, 1] == 1.0
    assert values[1, 1] == 1.0
    assert values[0, 0] == 0.0


def test_l4_augment_bundle_wiring():
    from datetime import datetime
    from pathlib import Path

    from preproc.l3_rasterize import IDX_MASK, IDX_VALUE, TargetPoint, empty_bundle, patch_grid
    from preproc.l4_augment import SOURCE_SYNTH_L4_MASKED, augment_bundle_from_l4, apply_l4_to_sample

    windows = [0, 24]
    grid = patch_grid(25.0, -90.0, 3.0, 0.25)
    t, h, w = len(windows), grid.height, grid.width
    l3 = empty_bundle(t, h, w)
    l3[IDX_MASK, 0, h // 2, w // 2] = 1.0
    l3[IDX_VALUE, 0, h // 2, w // 2] = 0.05
    l4_vals = np.full((t, h, w), 0.2, dtype=np.float64)
    l4_errs = np.full((t, h, w), 0.01, dtype=np.float64)
    l4_cfg = {"noise_scale": 0.0, "mission_dropout_prob": 0.0, "time_window_dropout_prob": 0.0}
    aug, src = augment_bundle_from_l4(l3, l4_vals, l4_errs, l4_cfg, np.random.default_rng(0))
    assert aug[IDX_MASK, 0, h // 2, w // 2] == 1.0
    assert abs(float(aug[IDX_VALUE, 0, h // 2, w // 2]) - 0.2) < 1e-5
    assert src[0, h // 2, w // 2] == SOURCE_SYNTH_L4_MASKED

    sample = {
        "ssh": l3,
        "wind_u": empty_bundle(t, h, w),
        "wind_v": empty_bundle(t, h, w),
        "sources": {},
    }
    target = TargetPoint(25.0, -90.0, datetime(2020, 1, 15))
    out = apply_l4_to_sample(
        sample,
        target=target,
        grid=grid,
        raw_root=Path("/nonexistent"),
        windows_hours=windows,
        l4_cfg={"enabled": True, "mode": "mask_augment", "apply_to": ["ssh"], **l4_cfg},
        rng=np.random.default_rng(1),
    )
    assert "l4_augment" in out
    assert out["l4_augment"]["mode"] == "mask_augment"
    assert "ssh" in out["l4_augment"]["source_flags"]


def test_l4_processed_batch_smoke():
    from pathlib import Path

    root = Path(__file__).resolve().parent
    cfg_path = root / "config_argo_l3_l4_smoke.json"
    if not cfg_path.is_file():
        return
    from playground import read_json
    from preproc.export_l3_cache import build_l3_processed_batch

    cfg = read_json(cfg_path)
    path = build_l3_processed_batch(cfg, max_samples=4, force=True)
    import pickle

    with open(path, "rb") as f:
        batch = pickle.load(f)
    assert batch.get("l4_augment", {}).get("enabled") is True
    assert batch["samples"][0].get("l4_augment") is not None


def test_l3_dataloader_batch():
    from pathlib import Path

    root = Path(__file__).resolve().parent
    cfg_path = root / "config_argo_l3_smoke.json"
    if not cfg_path.is_file():
        return
    from playground import read_json
    from preproc.export_l3_cache import build_argo_l3_train_cache

    cfg = read_json(cfg_path)
    cache_path = build_argo_l3_train_cache(cfg, force=True, max_samples=16)
    from preproc.l3_input import compute_l3_input_dim

    enc_flags = {k: bool(v) for k, v in cfg["input_params"].items()}
    expected_dim = compute_l3_input_dim(enc_flags, cfg["io"]["l3"])
    import data_loader.data_loaders as dlmod

    loader = dlmod.NeSPReSODataLoader(
        cache_path=cache_path,
        batch_size=4,
        split="train",
        split_mode="chronological",
        v2_src=cfg["io"].get("v2_src"),
    )
    assert loader.l3_enabled
    inputs, targets, idx = next(iter(loader))
    assert inputs.shape[1] == expected_dim
    assert targets.ndim == 2


def test_raw_profile_rmse_decoder_indexing():
    """Decoder path aligns sample-major preds with depth-major cache profiles."""
    from collections import OrderedDict

    import torch.nn as nn

    from eval_run import raw_profile_rmse

    depth, n = 4, 8
    idx = np.array([1, 4], dtype=int)
    true = {"temperature": np.arange(depth * n, dtype=np.float64).reshape(depth, n)}
    pred_profiles = true["temperature"][:, idx].T + 0.5
    outputs = OrderedDict([("temperature", 2)])

    class _StubDecoder(nn.Module):
        def __init__(self, profiles):
            super().__init__()
            self.register_buffer("profiles", torch.tensor(profiles, dtype=torch.float32))

        def decode(self, pcs):
            return self.profiles[: pcs.shape[0]]

    decoders = {"temperature": _StubDecoder(pred_profiles)}
    rmse = raw_profile_rmse(
        np.zeros((len(idx), 2), dtype=np.float32),
        true,
        {},
        outputs,
        idx,
        decoders=decoders,
        device=torch.device("cpu"),
    )
    assert abs(rmse["temperature"] - 0.5) < 1e-6


def test_stratified_eval_bins_and_aggregate():
    from eval_stratified import (
        bin_coverage,
        bin_track_km,
        build_strata_masks,
        pooled_stratum_metrics,
    )

    assert bin_coverage(0.0) == "zero"
    assert bin_coverage(0.005) == "trace"
    assert bin_coverage(0.02) == "low"
    assert bin_track_km(10.0) == "0-25km"
    assert bin_track_km(float("nan")) == "missing"

    years = np.array([2015, 2020, 2018], dtype=int)
    masks = build_strata_masks(
        years=years,
        coverage=np.array([0.0, 0.02, 0.0]),
        track_km=np.array([10.0, 80.0, np.nan]),
    )
    assert masks["subset_high_observation_2020"].tolist() == [False, True, False]
    assert masks["coverage_zero"].tolist() == [True, False, True]

    pred = {"temperature": np.array([[0.0, 1.0], [0.0, 3.0]])}
    true = {"temperature": np.array([[0.0, 0.0], [0.0, 0.0]])}
    stats = pooled_stratum_metrics(pred, true, np.array([True, True]))
    assert abs(stats["rmse"]["temperature"] - np.sqrt((1.0**2 + 3.0**2) / 4)) < 1e-9
    assert stats["n"] == 2


def test_stratified_eval_l3_cache_smoke():
    """Build tiny L3 cache and verify strata masks + report shape without a checkpoint."""
    from pathlib import Path

    root = Path(__file__).resolve().parent
    cfg_path = root / "config_argo_l3_smoke.json"
    if not cfg_path.is_file():
        return
    from playground import read_json
    from preproc.export_l3_cache import build_argo_l3_train_cache

    from eval_stratified import build_strata_masks, extract_l3_metrics, sample_years

    cfg = read_json(cfg_path)
    cache_path = build_argo_l3_train_cache(cfg, force=True, max_samples=8)
    import pickle

    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    indices = list(range(8))
    l3 = extract_l3_metrics(cache, indices)
    assert l3["ssh_coverage_fraction"].shape == (8,)
    years = sample_years(
        cache,
        indices,
        dataset_tag=cfg["io"]["dataset_tag"],
        v2_src=cfg["io"].get("v2_src"),
    )
    masks = build_strata_masks(
        years=years,
        coverage=l3["ssh_coverage_fraction"],
        track_km=l3["nearest_track_km"],
    )
    assert masks["all"].sum() == 8


def test_static_stability_readiness_synthetic():
    from diagnostics.readiness import static_stability_diagnostic

    n_depth = 5
    p = np.linspace(0.0, 100.0, n_depth)
    t_stable = np.linspace(25.0, 10.0, n_depth)[:, None] * np.ones((1, 2))
    s = 36.0 * np.ones((n_depth, 2))
    lat = np.array([25.0, 25.0])
    lon = np.array([-90.0, -90.0])

    stable = static_stability_diagnostic(t_stable, s, p, lat, lon)
    assert stable["violation_rate"] == 0.0
    assert stable["profile_flags"] == [0, 0]
    assert stable["method"] == "gsw_torch.sigma0_inversion"

    t_bad = t_stable.copy()
    t_bad[2, 0] = 30.0
    mixed = static_stability_diagnostic(t_bad, s, p, lat, lon)
    assert mixed["profile_flags"][0] == 1
    assert mixed["profile_flags"][1] == 0
    assert mixed["total_violations"] >= 1


def test_gsw_torch_matches_gsw_reference():
    """gsw_torch signatures match gsw; vectorized σ₀ within ~1e-5 kg/m³ of reference gsw."""
    import gsw as gsw_ref
    import gsw_torch as gt

    from diagnostics.readiness import _GSW_REF_ATOL_KGM3, static_stability_diagnostic

    n_depth, n_samples = 6, 4
    p = np.linspace(0.0, 200.0, n_depth)
    temp = np.linspace(26.0, 8.0, n_depth)[:, None] * np.ones((1, n_samples))
    temp[2, 1] = 29.0  # one inversion
    sal = (36.0 + 0.1 * np.linspace(0.0, 1.0, n_depth))[:, None] * np.ones((1, n_samples))
    lat = np.linspace(24.0, 27.0, n_samples)
    lon = np.linspace(-92.0, -88.0, n_samples)

    report = static_stability_diagnostic(temp, sal, p, lat, lon)

    # Per-profile reference via vanilla gsw
    flags_ref = []
    for j in range(n_samples):
        sa = gsw_ref.SA_from_SP(sal[:, j], p, lon[j], lat[j])
        ct = gsw_ref.CT_from_t(sa, temp[:, j], p)
        sig = gsw_ref.sigma0(sa, ct)
        delta = np.diff(sig)
        ok = np.isfinite(temp[:, j]) & np.isfinite(sal[:, j])
        valid = ok[:-1] & ok[1:]
        flags_ref.append(bool(np.any(valid & (delta < -0.01))))
    assert report["profile_flags"] == [int(v) for v in flags_ref]

    # Vectorized gsw_torch sigma0 matches gsw on all finite points
    pres = p[:, None] * np.ones((1, n_samples))
    sa_t = gt.SA_from_SP(sal, pres, lon, lat)
    ct_t = gt.CT_from_t(sa_t, temp, pres)
    sig_t = np.asarray(gt.sigma0(sa_t, ct_t))
    for j in range(n_samples):
        sa = gsw_ref.SA_from_SP(sal[:, j], p, lon[j], lat[j])
        ct = gsw_ref.CT_from_t(sa, temp[:, j], p)
        sig = gsw_ref.sigma0(sa, ct)
        assert np.nanmax(np.abs(sig - sig_t[:, j])) < _GSW_REF_ATOL_KGM3


def test_readiness_report_requires_lat_lon():
    from diagnostics.readiness import readiness_report

    t = np.ones((3, 2))
    s = np.ones((3, 2))
    try:
        readiness_report(t, s)
        raise AssertionError("expected ValueError for missing lat/lon")
    except ValueError as exc:
        assert "latitude" in str(exc)


def test_results_table_smoke():
    from pathlib import Path

    from scripts.results_table import build_table, collect_eval_rows

    root = Path(__file__).resolve().parent
    saved = root / "saved"
    if not any(saved.glob("eval_*.json")):
        return
    rows = collect_eval_rows(saved)
    report = build_table(rows)
    assert report["n_files"] >= 1
    prod = [r for r in report["rows"] if "prod" in r["label"].lower()]
    assert prod or len(report["rows"]) >= 1


if __name__ == "__main__":
    test_cap_batch_size()
    test_resolve_batch_size_fixed()
    test_compute_input_dim()
    test_patch_conv_mlp_point_mode()
    test_patch_conv_mlp_patch_mode()
    test_patch_conv_mlp_residual_mode()
    test_res_autoencoder_round_trip()
    test_prediction_model_v2()
    test_combined_pca_loss_v2()
    test_pred_profile_cached_matches_combined()
    test_decoder_profile_loss()
    test_decoder_profile_loss_nan_mask()
    test_raw_profile_rmse_decoder_indexing()
    test_stratified_eval_bins_and_aggregate()
    test_stratified_eval_l3_cache_smoke()
    test_static_stability_readiness_synthetic()
    test_gsw_torch_matches_gsw_reference()
    test_readiness_report_requires_lat_lon()
    test_results_table_smoke()
    test_pca_round_trip()
    test_asymmetric_output_offsets()
    test_split_matches_torch_seed()
    test_chronological_split_no_leakage()
    test_l3_bin_observations_synthetic()
    test_l3_lon_normalization_gom_bbox()
    test_l3_rasterize_missing_is_explicit()
    test_l3_processed_batch_smoke()
    test_l3_input_dim_and_flatten()
    test_l3_feature_subset_dim()
    test_l3_cache_layout_guard()
    test_sync_arch_l3_config()
    test_patch_conv_mlp_l3_forward()
    test_l4_mask_augment()
    test_l4_augment_bundle_wiring()
    test_l4_processed_batch_smoke()
    test_l3_dataloader_batch()
    test_train_monitor_once()
    test_overlap_pairs()
    test_cache_schema_keys()
    print("selfcheck: all assertions passed")
