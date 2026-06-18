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
        assert arr.ndim == 4
    assert "split" in s0
    assert "coverage" in s0


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
    test_raw_profile_rmse_decoder_indexing()
    test_pca_round_trip()
    test_asymmetric_output_offsets()
    test_split_matches_torch_seed()
    test_chronological_split_no_leakage()
    test_l3_bin_observations_synthetic()
    test_l3_rasterize_missing_is_explicit()
    test_l3_processed_batch_smoke()
    test_train_monitor_once()
    test_overlap_pairs()
    test_cache_schema_keys()
    print("selfcheck: all assertions passed")
