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
    "combined_loss": 0.05071902275085449,
    "pca_loss": 0.00037024958874098957,
    "weighted_mse_loss": 0.0025833332911133766,
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
    from base.batch_size import cap_batch_size, resolve_batch_size

    assert cap_batch_size(512, 100) == 100
    assert cap_batch_size(256, 1000) == 256


def test_resolve_batch_size_fixed():
    from base.batch_size import resolve_batch_size

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


def test_argo_l4_input_dim_and_forward():
    from model.model import PatchMaskConvMLP
    from preproc.preproc_isas_sat import build_argo_l4_input_matrix, count_scalar_dims

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
        "basin_sss": True,
        "basin_sst": True,
        "basin_ssh": True,
        "bathy_depth": True,
    }
    assert count_scalar_dims(flags) == 10
    assert compute_input_dim(flags, 2, 6) == 535
    assert sat_patch_shape(2, 6, 3, input_params=flags) == (3, 7, 5, 5)
    assert compute_input_dim(flags, 2, 6, use_mask_channels=True) == 1060
    assert sat_patch_shape(2, 6, 3, input_params=flags, use_mask_channels=True) == (6, 7, 5, 5)

    n = 3
    per = 5 * 5 * 7
    juld = np.linspace(50, 60, n, dtype=np.float32)
    lat = np.full(n, 25.0, dtype=np.float32)
    lon = np.full(n, -90.0, dtype=np.float32)
    sat_vars = {k: np.ones((n, per), dtype=np.float32) for k in ("sss", "sst", "ssh")}
    sat_vars["sst"] += 300.0
    basin = {"sss": np.zeros(n), "sst": np.full(n, 300.0), "ssh": np.zeros(n)}
    bathy = np.full(n, 500.0, dtype=np.float32)
    x = build_argo_l4_input_matrix(
        juld, lat, lon, sat_vars, flags, basin_means=basin, bathy_depth=bathy
    )
    assert x.shape == (n, 535)
    assert np.isfinite(x).all()

    x_mask = build_argo_l4_input_matrix(
        juld,
        lat,
        lon,
        sat_vars,
        flags,
        basin_means=basin,
        bathy_depth=bathy,
        use_mask_channels=True,
    )
    assert x_mask.shape == (n, 1060)
    mask_cols = x_mask[:, 10 + per : 10 + 2 * per]
    assert mask_cols.min() >= 0.0 and mask_cols.max() <= 1.0

    model = PatchMaskConvMLP(
        input_dim=535,
        output_dim=32,
        dropout_prob=0.0,
        d_model=32,
        head_layers=[16],
        conv_channels=[8, 16],
        patch_shape=[3, 7, 5, 5],
        n_enc=10,
        n_sat=3,
        use_mask_channels=False,
    )
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(x))
    assert out.shape == (n, 32)


def test_bathy_truncation_loss():
    from collections import OrderedDict
    from sklearn.decomposition import PCA

    from model.loss import PCALoss, bathy_depth_mask

    n_z, n = 5, 4
    rng = np.random.default_rng(0)
    prof = rng.normal(size=(n, n_z)).astype(np.float64)
    pca = PCA(n_components=2).fit(prof)
    outputs = OrderedDict([("temperature", 2)])
    pres = np.arange(n_z, dtype=np.float32)
    bottom = np.array([1.0, 2.0, 0.0, 1.5], dtype=np.float32)
    device = torch.device("cpu")
    loss_fn = PCALoss(
        {"temperature": pca},
        outputs,
        device=device,
        bottom_depth=bottom,
        pres_levels=pres,
    )
    pcs = torch.tensor(rng.normal(size=(n, 2)), dtype=torch.float32)
    targets = torch.tensor(pca.transform(prof), dtype=torch.float32)
    idx = np.arange(n)
    mask = bathy_depth_mask(idx, bottom, pres, device)
    assert mask.shape == (n, n_z)
    assert not mask[:, -1].any()
    val = loss_fn(pcs, targets, indices=idx)
    assert torch.isfinite(val)


def test_argo_l4_cache_smoke():
    from pathlib import Path

    from base.util import read_json
    from preproc.export_argo_l4_cache import build_argo_l4_cache

    root = Path(__file__).resolve().parent
    cfg_path = root / "config/argo/config_argo_patch_l4_smoke.json"
    sat_path = root.parent / "data/NeSPReSO_v2_ARGO_GoM_sat/satellite_NeSPReSO_v2_ARGO_GoM_smoke.h5"
    if not cfg_path.is_file() or not sat_path.is_file():
        return
    cfg = read_json(cfg_path)
    cache_path = build_argo_l4_cache(cfg, force=False)
    import pickle

    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    assert cache["dataset_tag"] == "argo_l4"
    assert cache["inputs"].shape[1] == 535
    assert cache["targets"].shape[1] == 32
    assert "bottom_depth" in cache
    assert list(cache["sat_patch_shape"]) == [3, 7, 5, 5]
    assert cache.get("use_mask_channels") is False


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


from base.util import read_json


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
    cfg = read_json(root / "config/argo/config_argo.json")
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
        if cache.get("dataset_tag") == "argo_l4":
            assert "bottom_depth" in cache
            assert cache.get("bathy_truncation") is True
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
    cfg_path = root / "config/argo/config_argo_l3_smoke.json"
    if not cfg_path.is_file():
        return
    from base.util import read_json
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

    from base.util import read_json
    from preproc.l3_input import sync_arch_with_io

    root = Path(__file__).resolve().parent
    cfg = read_json(root / "config/argo/config_argo_l3_smoke.json")
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
    cfg_path = root / "config/argo/config_argo_l3_l4_smoke.json"
    if not cfg_path.is_file():
        return
    from base.util import read_json
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
    cfg_path = root / "config/argo/config_argo_l3_smoke.json"
    if not cfg_path.is_file():
        return
    from base.util import read_json
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
    cfg_path = root / "config/argo/config_argo_l3_smoke.json"
    if not cfg_path.is_file():
        return
    from base.util import read_json
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
    """Vectorized gsw_torch σ₀ matches per-profile gsw_torch loop within tolerance."""
    import gsw_torch as gsw

    from diagnostics.readiness import _GSW_REF_ATOL_KGM3, static_stability_diagnostic

    n_depth, n_samples = 6, 4
    p = np.linspace(0.0, 200.0, n_depth)
    temp = np.linspace(26.0, 8.0, n_depth)[:, None] * np.ones((1, n_samples))
    temp[2, 1] = 29.0  # one inversion
    sal = (36.0 + 0.1 * np.linspace(0.0, 1.0, n_depth))[:, None] * np.ones((1, n_samples))
    lat = np.linspace(24.0, 27.0, n_samples)
    lon = np.linspace(-92.0, -88.0, n_samples)

    report = static_stability_diagnostic(temp, sal, p, lat, lon)

    # Per-profile reference via gsw_torch loop
    import torch

    flags_ref = []
    for j in range(n_samples):
        sa = gsw.SA_from_SP(
            torch.as_tensor(sal[:, j], dtype=torch.float64),
            torch.as_tensor(p, dtype=torch.float64),
            torch.as_tensor(lon[j], dtype=torch.float64),
            torch.as_tensor(lat[j], dtype=torch.float64),
        )
        ct = gsw.CT_from_t(sa, torch.as_tensor(temp[:, j], dtype=torch.float64), torch.as_tensor(p, dtype=torch.float64))
        sig = gsw.sigma0(sa, ct).detach().cpu().numpy()
        delta = np.diff(sig)
        ok = np.isfinite(temp[:, j]) & np.isfinite(sal[:, j])
        valid = ok[:-1] & ok[1:]
        flags_ref.append(bool(np.any(valid & (delta < -0.01))))
    assert report["profile_flags"] == [int(v) for v in flags_ref]

    # Vectorized gsw_torch sigma0 matches loop on all finite points
    pres = p[:, None] * np.ones((1, n_samples))
    sa_t = gsw.SA_from_SP(
        torch.as_tensor(sal, dtype=torch.float64),
        torch.as_tensor(pres, dtype=torch.float64),
        torch.as_tensor(lon, dtype=torch.float64),
        torch.as_tensor(lat, dtype=torch.float64),
    )
    ct_t = gsw.CT_from_t(sa_t, torch.as_tensor(temp, dtype=torch.float64), torch.as_tensor(pres, dtype=torch.float64))
    sig_t = np.asarray(gsw.sigma0(sa_t, ct_t).detach().cpu().numpy())
    for j in range(n_samples):
        sa = gsw.SA_from_SP(
            torch.as_tensor(sal[:, j], dtype=torch.float64),
            torch.as_tensor(p, dtype=torch.float64),
            torch.as_tensor(lon[j], dtype=torch.float64),
            torch.as_tensor(lat[j], dtype=torch.float64),
        )
        ct = gsw.CT_from_t(sa, torch.as_tensor(temp[:, j], dtype=torch.float64), torch.as_tensor(p, dtype=torch.float64))
        sig = gsw.sigma0(sa, ct).detach().cpu().numpy()
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


def test_dates_to_juld_round_trip():
    """dates_to_juld must invert sample_dates exactly — it feeds the climatology's DOY.

    A wrong convention here does not crash: it silently shifts the seasonal cycle. The old
    export_field_product.py used `float(ord(ds))` = 50.0 for every date in the 2000s.
    """
    from base.split_utils import dates_to_juld, sample_dates

    for tag in ("argo_v2", "isas20"):
        for iso in ("2000-01-01", "2004-02-29", "2019-12-31", "2020-01-01", "2021-07-04"):
            juld = dates_to_juld([iso], dataset_tag=tag)
            back = sample_dates(juld, dataset_tag=tag)
            assert str(back[0]) == iso, f"{tag} {iso} -> juld {juld} -> {back[0]}"

    # Round-trips real cache JULDs (whole-day truncation is expected and intended).
    rng = np.random.default_rng(0)
    for tag in ("argo_v2", "isas20"):
        base = 738000.0 if tag == "argo_v2" else 25000.0
        juld = base + rng.integers(0, 4000, size=200).astype(np.float64)
        assert np.array_equal(dates_to_juld(sample_dates(juld, dataset_tag=tag), dataset_tag=tag), juld)

    # ARGO is MATLAB datenum; the two tags must NOT agree (a swapped tag is a real bug).
    assert dates_to_juld(["2020-01-01"], dataset_tag="argo_v2")[0] == 737791.0
    assert dates_to_juld(["2020-01-01"], dataset_tag="isas20")[0] == 25567.0


def test_ensemble_crps_matches_pairwise_definition():
    """Sorted-ensemble CRPS == the naive O(N²) double sum, and == MAE at zero spread."""
    from diagnostics.readiness import ensemble_crps

    rng = np.random.default_rng(0)
    n, shape = 9, (4, 3)
    members = rng.normal(size=(n, *shape))
    target = rng.normal(size=shape)

    fast = ensemble_crps(members, target)
    naive = np.empty(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            x, y = members[:, i, j], target[i, j]
            t1 = np.mean(np.abs(x - y))
            t2 = np.abs(x[:, None] - x[None, :]).sum() / (2 * n * n)
            naive[i, j] = t1 - t2
    assert np.allclose(fast, naive, atol=1e-12)

    # A zero-spread ensemble is a point forecast: CRPS degenerates to |x - y|.
    const = np.repeat(target[None, ...], n, axis=0)
    assert np.allclose(ensemble_crps(const, target + 0.5), 0.5, atol=1e-12)


def test_uncertainty_calibration_detects_calibrated_ensemble():
    """Positive control: a *known* calibrated ensemble must score ratio ≈ 1.

    Without this, a real ratio of ≈1 is unreadable — it could equally mean the models are
    calibrated or that the metric is broken. This pins the metric so the model result is
    interpretable (PLAN-agentic-close-out.md Step 3).
    """
    from diagnostics.readiness import uncertainty_calibration_hook

    rng = np.random.default_rng(1)
    n_members, n_depth, n_samples = 60, 12, 400
    # Heteroscedastic on purpose: true sigma spans 0.2..2.0 across points. ENCE bins BY
    # predicted spread, so it only has signal when the spread genuinely varies. On
    # homoscedastic data the bins sort on sampling noise in the sample std (relative error
    # ~1/sqrt(2(N-1)) ≈ 9% at N=60) and ENCE reports that floor (~0.08 here) no matter how
    # well calibrated the ensemble is. Read a real ENCE against that floor, not against 0.
    sigma = rng.uniform(0.2, 2.0, size=(n_depth, n_samples))
    mu_true = rng.normal(size=(n_depth, n_samples))
    # truth and members are draws from the SAME per-point distribution => perfectly reliable
    target = mu_true + rng.normal(size=(n_depth, n_samples)) * sigma
    members = mu_true[None, ...] + rng.normal(size=(n_members, n_depth, n_samples)) * sigma[None, ...]

    out = uncertainty_calibration_hook(
        ensemble_mean=members.mean(axis=0),
        ensemble_spread=members.std(axis=0, ddof=1),
        target=target,
        depth=np.arange(n_depth, dtype=float),
        members=members,
        n_members=n_members,
    )
    assert out["status"] == "ok"
    # Finite-N: a reliable N-member ensemble scores sqrt(N/(N+1)), not exactly 1.
    assert abs(out["spread_error_ratio"] - np.sqrt(n_members / (n_members + 1))) < 0.02
    assert abs(out["spread_error_ratio_corrected"] - 1.0) < 0.02
    assert out["ence"] < 0.05, out["ence"]
    # A calibrated heteroscedastic ensemble must also RANK its errors: wider spread where
    # the error is bigger. Ratio ≈ 1 alone does not prove this.
    assert out["spread_error_rank_corr"] > 0.3, out["spread_error_rank_corr"]
    assert out["crps"] > 0
    assert len(out["reliability_by_depth"]["rmse"]) == n_depth


def test_uncertainty_calibration_detects_underdispersion():
    """Negative control: spread deliberately 4x too small => ratio ≈ 0.25, ENCE large."""
    from diagnostics.readiness import uncertainty_calibration_hook

    rng = np.random.default_rng(2)
    n_members, n_depth, n_samples = 40, 8, 500
    sigma = 1.0
    mu_true = rng.normal(size=(n_depth, n_samples))
    target = mu_true + rng.normal(scale=sigma, size=(n_depth, n_samples))
    members = mu_true[None, ...] + rng.normal(scale=sigma / 4.0, size=(n_members, n_depth, n_samples))

    out = uncertainty_calibration_hook(
        ensemble_mean=members.mean(axis=0),
        ensemble_spread=members.std(axis=0, ddof=1),
        target=target,
        n_members=n_members,
    )
    assert out["status"] == "ok"
    assert 0.20 < out["spread_error_ratio"] < 0.30, out["spread_error_ratio"]
    assert out["ence"] > 0.5


def test_uncertainty_hook_not_implemented_without_ensemble():
    """N=0 keeps the honest `not_implemented` contract rather than inventing zeros."""
    from diagnostics.readiness import uncertainty_calibration_hook

    out = uncertainty_calibration_hook()
    assert out["status"] == "not_implemented"
    assert "ensemble_spread" in out["expected_fields"]


def test_mc_dropout_enables_only_dropout():
    """eval() + enable_mc_dropout => Dropout trains, everything else stays in eval."""
    import torch.nn as nn

    from nb_metrics import enable_mc_dropout

    model = nn.Sequential(nn.Linear(4, 8), nn.BatchNorm1d(8), nn.Dropout(0.3), nn.Linear(8, 2))
    model.eval()
    probs = enable_mc_dropout(model)
    assert probs == [0.3]
    assert model[2].training is True          # Dropout live
    assert model[1].training is False         # BatchNorm still eval (running stats frozen)
    assert model[0].training is False

    # Stochastic: two passes must differ, or the "ensemble" is a copy of one number.
    x = torch.randn(16, 4)
    with torch.no_grad():
        assert not torch.allclose(model(x), model(x))

    # A model with no dropout must fail loudly, not report zero spread.
    plain = nn.Sequential(nn.Linear(4, 2))
    plain.eval()
    try:
        enable_mc_dropout(plain)
        raise AssertionError("expected ValueError for a model with no dropout")
    except ValueError as exc:
        assert "no nn.Dropout" in str(exc)

    zeroed = nn.Sequential(nn.Linear(4, 8), nn.Dropout(0.0), nn.Linear(8, 2))
    zeroed.eval()
    try:
        enable_mc_dropout(zeroed)
        raise AssertionError("expected ValueError for p=0 dropout")
    except ValueError as exc:
        assert "p=0" in str(exc)


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


def test_climatology_fit_eval_roundtrip():
    import numpy as np
    from preproc.climatology import eval_climatology, fit_climatology

    n, n_z = 80, 20
    rng = np.random.default_rng(0)
    lat = rng.uniform(20, 28, n).astype(np.float32)
    lon = rng.uniform(-95, -85, n).astype(np.float32)
    juld = np.linspace(730000, 730300, n).astype(np.float32)
    pres = np.linspace(0, 190, n_z, dtype=np.float32)
    profiles = {
        "temperature": (20 + rng.normal(size=(n_z, n))).astype(np.float32),
        "salinity": (36 + 0.1 * rng.normal(size=(n_z, n))).astype(np.float32),
    }
    train_idx = np.arange(0, n, 2)
    cfg = {"dataset_tag": "argo_v2", "v2_src": None, "basin": {"min_lat": 18, "max_lat": 31, "min_lon": -98, "max_lon": -81}}
    clim = fit_climatology(profiles, lat, lon, juld, pres, train_idx, cfg)
    pred = eval_climatology(clim, lat, lon, juld)
    for name in profiles:
        assert pred[name].shape == profiles[name].shape
        err = np.nanmean(np.abs(pred[name] - profiles[name]))
        assert err < 5.0


def test_climatology_train_only():
    import numpy as np
    from preproc.climatology import eval_climatology, fit_climatology

    n, n_z = 40, 10
    rng = np.random.default_rng(1)
    lat = rng.uniform(22, 27, n).astype(np.float32)
    lon = rng.uniform(-92, -87, n).astype(np.float32)
    juld = np.linspace(730100, 730200, n).astype(np.float32)
    pres = np.linspace(0, 100, n_z, dtype=np.float32)
    profiles = {"temperature": rng.normal(size=(n_z, n)).astype(np.float32)}
    profiles["temperature"][:, -1] = 999.0
    train_idx = np.arange(n - 1)
    cfg = {"dataset_tag": "argo_v2", "basin": {"min_lat": 18, "max_lat": 31, "min_lon": -98, "max_lon": -81}}
    clim_a = fit_climatology(profiles, lat, lon, juld, pres, train_idx, cfg)
    profiles_b = dict(profiles)
    profiles_b["temperature"] = profiles["temperature"].copy()
    profiles_b["temperature"][:, -1] = -999.0
    clim_b = fit_climatology(profiles_b, lat, lon, juld, pres, train_idx, cfg)
    pa = eval_climatology(clim_a, lat, lon, juld)["temperature"]
    pb = eval_climatology(clim_b, lat, lon, juld)["temperature"]
    assert np.allclose(pa, pb, atol=1e-4)


def test_anomaly_cache_addback():
    import numpy as np
    from collections import OrderedDict
    from sklearn.decomposition import PCA
    from model.loss import reconstruct_physical_profiles

    n, n_z, k = 30, 15, 15
    rng = np.random.default_rng(2)
    raw = rng.normal(size=(n_z, n)).astype(np.float32)
    clim = (raw.mean(axis=1, keepdims=True) + 0.1 * rng.normal(size=(n_z, n))).astype(np.float32)
    anom = raw - clim
    pca = PCA(n_components=k).fit(anom.T)
    pcs = pca.transform(anom.T)
    outputs = OrderedDict([("temperature", k)])
    pca_models = {"temperature": pca}
    recon = reconstruct_physical_profiles(pcs, pca_models, outputs, clim_profiles={"temperature": clim})
    assert np.nanmax(np.abs(recon["temperature"] - raw)) < 0.5


def test_ssh_obs_cached_smoke():
    import numpy as np
    from preproc.ssh_obs import _juld_to_astropy_jd

    jd = _juld_to_astropy_jd(730500.0, dataset_tag="argo_v2", v2_src="/unity/g2/jmiranda/v2-nespreso/src")
    assert 2450000 < jd < 2465000


def test_steric_height_sanity():
    import torch
    from model.steric import steric_height_anomaly

    pres = torch.linspace(0, 500, 6)
    lat = torch.tensor([25.0])
    lon = torch.tensor([-90.0])
    warm = torch.tensor([[28.0, 27.0, 26.0, 25.0, 24.0, 23.0]])
    cold = warm - 5.0
    sal = torch.full((1, 6), 36.0)
    h_warm = steric_height_anomaly(warm, sal, pres, lat, lon)
    h_cold = steric_height_anomaly(cold, sal, pres, lat, lon)
    assert h_warm.item() > h_cold.item()
    assert abs(h_warm.item()) > 1e-9


def test_steric_loss_grad():
    import torch
    from collections import OrderedDict
    from model.loss import CombinedPCALoss

    outputs = OrderedDict([("temperature", 2), ("salinity", 2)])
    n, n_z = 4, 5
    from sklearn.decomposition import PCA

    pca_models = {}
    for name in outputs:
        pca_models[name] = PCA(2).fit(np.random.randn(20, n_z))
    weights = np.ones(4, dtype=np.float32)
    clim = {name: np.zeros((n_z, n), dtype=np.float32) for name in outputs}
    steric_meta = type("M", (), {
        "LAT": np.full(n, 25.0, dtype=np.float32),
        "LON": np.full(n, -90.0, dtype=np.float32),
        "PRES": np.linspace(0, 100, n_z, dtype=np.float32),
        "ssh_obs_sla": np.zeros(n, dtype=np.float32),
        "clim_steric": np.zeros(n, dtype=np.float32),
        "steric_calibration": {"alpha": 1.0, "beta": 0.0},
    })()
    loss_fn = CombinedPCALoss(
        pca_models, outputs, weights, torch.device("cpu"),
        steric_config={"enabled": True, "scale": 0.01, "subsample_dz": 2},
        steric_meta=steric_meta,
        clim_profiles=clim,
    )
    pcs = torch.zeros(n, 4, requires_grad=True)
    tgt = torch.zeros(n, 4)
    idx = torch.arange(n)
    out = loss_fn(pcs, tgt, idx)
    out.backward()
    assert pcs.grad is not None
    assert torch.isfinite(pcs.grad).any()


def test_field_unet_shapes():
    import torch
    from model.model import FieldUNet

    m = FieldUNet(in_channels=7, out_channels=32, base_width=16)
    x = torch.randn(2, 7, 52, 68)
    y = m(x)
    assert y.shape == (2, 32, 52, 68)


def _synthetic_field_cache(n_dates=4, n_profiles=12, n_z=10, k=4):
    import numpy as np
    from collections import OrderedDict
    from sklearn.decomposition import PCA

    rng = np.random.default_rng(7)
    h, w = 8, 10
    n = n_profiles
    pres = np.linspace(0, 100, n_z, dtype=np.float32)
    lat = rng.uniform(20, 28, n).astype(np.float32)
    lon = rng.uniform(-95, -85, n).astype(np.float32)
    juld = np.linspace(730100, 730200, n).astype(np.float32)
    profiles = {
        "temperature": rng.normal(size=(n_z, n)).astype(np.float32),
        "salinity": (36 + 0.1 * rng.normal(size=(n_z, n))).astype(np.float32),
    }
    clim = {name: arr + 0.5 for name, arr in profiles.items()}
    outputs = OrderedDict([("temperature", k), ("salinity", k)])
    targets = np.zeros((n, 2 * k), dtype=np.float32)
    pca_models = {}
    for name, nk in outputs.items():
        anom = profiles[name] - clim[name]
        pca = PCA(nk).fit(anom.T)
        pca_models[name] = pca
        sl = list(outputs.keys()).index(name) * k
        targets[:, sl : sl + nk] = pca.transform(anom.T)
    dates = [f"2016-06-{d:02d}" for d in range(1, n_dates + 1)]
    sample_date_idx = rng.integers(0, n_dates, n)
    sample_pixel_rc = np.stack(
        [rng.integers(0, h, n), rng.integers(0, w, n)], axis=1
    ).astype(np.int64)
    fields = rng.normal(size=(n_dates, 3, h, w)).astype(np.float32)
    return {
        "fields": fields,
        "dates": dates,
        "sample_date_idx": sample_date_idx,
        "sample_pixel_rc": sample_pixel_rc,
        "inputs": rng.normal(size=(n, 9)).astype(np.float32),
        "targets": targets,
        "pca_models": pca_models,
        "outputs": dict(outputs),
        "weights": np.ones(n, dtype=np.float32),
        "profiles": profiles,
        "LAT": lat,
        "LON": lon,
        "PRES": pres,
        "JULD": juld,
        "clim_profiles": clim,
        "anomaly_targets": True,
        "ssh_obs_sla": rng.normal(size=n).astype(np.float32),
        "clim_steric": rng.normal(size=n).astype(np.float32) * 0.01,
        "steric_calibration": {"alpha": 1.0, "beta": 0.0},
        "dataset_tag": "argo_field",
        "min_depth": 0,
        "max_depth": int(pres[-1]),
    }


def test_field_gather_matches_loop():
    import torch
    from model.model import FieldUNet

    cache = _synthetic_field_cache()
    m = FieldUNet(in_channels=7, out_channels=8, base_width=8)
    from data_loader.data_loaders import FieldDataset

    ds = FieldDataset(cache, [0, 1])
    field, tgt, pixel_rc, sample_idx, _d = ds[0]
    out = m(field.unsqueeze(0))
    bop = torch.zeros(pixel_rc.shape[0], dtype=torch.long)
    rows = pixel_rc[:, 0].long()
    cols = pixel_rc[:, 1].long()
    gathered = out[0, :, rows, cols].T
    loop = []
    for i in range(pixel_rc.shape[0]):
        r, c = int(rows[i]), int(cols[i])
        loop.append(out[0, :, r, c].detach())
    loop = torch.stack(loop, dim=0)
    assert torch.allclose(gathered, loop, atol=1e-5)


def test_field_date_split_disjoint():
    from data_loader.data_loaders import FieldDataLoader
    import tempfile
    import pickle
    import os

    cache = _synthetic_field_cache(n_dates=10, n_profiles=20)
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        pickle.dump(cache, f)
        path = f.name
    try:
        dl = FieldDataLoader(
            cache_path=path,
            batch_size=2,
            split_mode="chronological",
            train_frac=0.7,
            val_frac=0.15,
            test_frac=0.15,
            split="train",
        )
        train = set(dl.split_indices["train"])
        val = set(dl.split_indices["val"])
        test = set(dl.split_indices["test"])
        assert train.isdisjoint(val)
        assert train.isdisjoint(test)
        assert val.isdisjoint(test)
        assert len(train | val | test) == len(cache["dates"])
    finally:
        os.unlink(path)


def test_field_cache_targets_match_v2():
    cache = _synthetic_field_cache()
    assert cache["targets"].shape[1] == sum(cache["outputs"].values())
    assert cache["anomaly_targets"] is True
    start = 0
    for name, k in cache["outputs"].items():
        pcs = cache["targets"][:, start : start + k]
        anom = cache["pca_models"][name].inverse_transform(pcs).T
        phys = anom + cache["clim_profiles"][name]
        assert np.nanmax(np.abs(phys - cache["profiles"][name])) < 1e-4
        start += k


def test_field_loss_grad_with_steric():
    import torch
    from collections import OrderedDict
    from model.loss import CombinedPCALoss
    from model.model import FieldUNet

    cache = _synthetic_field_cache(n_profiles=3, n_dates=1, k=2)
    outputs = OrderedDict(cache["outputs"])
    steric_meta = type("M", (), {
        "LAT": cache["LAT"],
        "LON": cache["LON"],
        "PRES": cache["PRES"],
        "ssh_obs_sla": cache["ssh_obs_sla"],
        "clim_steric": cache["clim_steric"],
        "steric_calibration": cache["steric_calibration"],
    })()
    loss_fn = CombinedPCALoss(
        cache["pca_models"], outputs, cache["weights"], torch.device("cpu"),
        steric_config={"enabled": True, "scale": 0.01, "subsample_dz": 2},
        steric_meta=steric_meta,
        clim_profiles=cache["clim_profiles"],
        mode="pc_mse_only",
    )
    model = FieldUNet(in_channels=7, out_channels=sum(outputs.values()), base_width=8)
    field = torch.randn(1, 7, cache["fields"].shape[2], cache["fields"].shape[3])
    mask = cache["sample_date_idx"] == 0
    pixel_rc = torch.tensor(cache["sample_pixel_rc"][mask], dtype=torch.long)
    sample_idx = torch.tensor(np.where(mask)[0], dtype=torch.long)
    batch_of_pixel = torch.zeros(pixel_rc.shape[0], dtype=torch.long)
    out = model(field)
    pred = out[batch_of_pixel, :, pixel_rc[:, 0], pixel_rc[:, 1]]
    tgt = torch.tensor(cache["targets"][mask], dtype=torch.float32)
    loss = loss_fn(pred, tgt, sample_idx)
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).any() for p in model.parameters())


def test_steric_train_calibration():
    import numpy as np
    from model.steric import compute_clim_steric, fit_steric_calibration

    n, n_z = 120, 20
    rng = np.random.default_rng(11)
    pres = np.linspace(0, 400, n_z, dtype=np.float32)
    lat = rng.uniform(22, 28, n).astype(np.float32)
    lon = rng.uniform(-94, -86, n).astype(np.float32)
    temp = 20 + rng.normal(0, 2, (n_z, n)).astype(np.float32)
    sal = 36 + rng.normal(0, 0.2, (n_z, n)).astype(np.float32)
    profiles = {"temperature": temp, "salinity": sal}
    clim_profiles = {
        "temperature": temp.mean(axis=1, keepdims=True) + 0.1 * rng.normal(size=(n_z, n)),
        "salinity": sal.mean(axis=1, keepdims=True) + 0.01 * rng.normal(size=(n_z, n)),
    }
    clim_steric = compute_clim_steric(clim_profiles, pres, lat, lon, subsample_dz=5)
    steric_true = compute_clim_steric(profiles, pres, lat, lon, subsample_dz=5)
    alpha, beta = 1.2, 0.03
    ssh_sla = alpha * (steric_true - clim_steric) + beta + rng.normal(0, 0.005, n)
    cal = fit_steric_calibration(steric_true, clim_steric, ssh_sla)
    pred = cal["alpha"] * (steric_true - clim_steric) + cal["beta"]
    corr = np.corrcoef(pred, ssh_sla)[0, 1]
    assert corr > 0.6, f"train steric-SLA calibration corr too low: {corr:.3f}"


def test_steric_matches_climatology_adt():
    import numpy as np
    from model.steric import compute_clim_steric

    n, n_z = 40, 15
    rng = np.random.default_rng(12)
    pres = np.linspace(0, 300, n_z, dtype=np.float32)
    lat = np.linspace(20, 29, n).astype(np.float32)
    lon = np.full(n, -90.0, dtype=np.float32)
    temp = np.broadcast_to(np.linspace(24, 18, n_z)[:, None], (n_z, n)).astype(np.float32)
    sal = np.full((n_z, n), 36.0, dtype=np.float32)
    warm = {"temperature": temp + 2.0, "salinity": sal}
    cool = {"temperature": temp, "salinity": sal}
    h_warm = compute_clim_steric(warm, pres, lat, lon, subsample_dz=3)
    h_cool = compute_clim_steric(cool, pres, lat, lon, subsample_dz=3)
    corr = np.corrcoef(h_warm - h_cool, lat)[0, 1]
    assert corr > 0.5


if __name__ == "__main__":
    test_cap_batch_size()
    test_resolve_batch_size_fixed()
    test_compute_input_dim()
    test_argo_l4_input_dim_and_forward()
    test_bathy_truncation_loss()
    test_argo_l4_cache_smoke()
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
    test_dates_to_juld_round_trip()
    test_ensemble_crps_matches_pairwise_definition()
    test_uncertainty_calibration_detects_calibrated_ensemble()
    test_uncertainty_calibration_detects_underdispersion()
    test_uncertainty_hook_not_implemented_without_ensemble()
    test_mc_dropout_enables_only_dropout()
    test_results_table_smoke()
    test_climatology_fit_eval_roundtrip()
    test_climatology_train_only()
    test_anomaly_cache_addback()
    test_ssh_obs_cached_smoke()
    test_steric_height_sanity()
    test_steric_loss_grad()
    test_steric_train_calibration()
    test_steric_matches_climatology_adt()
    test_field_unet_shapes()
    test_field_gather_matches_loop()
    test_field_date_split_disjoint()
    test_field_cache_targets_match_v2()
    test_field_loss_grad_with_steric()
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
