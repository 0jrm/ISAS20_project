#!/usr/bin/env python3
"""Analytic physical CRPS/ENCE + val-only σ recalib (ENCE(T) picker)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

ENCE_MAX = 0.20


def _predict(config, model, loss_fn, split: str, device):
    import data_loader.data_loaders as module_data

    dl_args = dict(config["data_loader"]["args"])
    dl_args["split"] = split
    dl_args["shuffle"] = False
    data_loader = getattr(module_data, config["data_loader"]["type"])(**dl_args)
    mus, sigs, ys = [], [], []
    with torch.no_grad():
        for data, target, indices in data_loader:
            data = data.to(device)
            target = target.to(device)
            indices = indices.to(device)
            out = model(data)
            if hasattr(loss_fn, "phys_mu_sigma"):
                mu, sigma = loss_fn.phys_mu_sigma(out, indices)
                y = loss_fn.physical_targets(indices)
            else:
                mu = loss_fn._mu_from_raw(out[:, : loss_fn.d])
                sigma = loss_fn._sigma_target_space(out[:, loss_fn.d :])
                y = loss_fn.decode_targets(target)
            mus.append(mu.cpu())
            sigs.append(sigma.cpu())
            ys.append(y.cpu())
    mu = torch.cat(mus).numpy()
    sigma = torch.cat(sigs).numpy()
    y = torch.cat(ys).numpy()
    z = np.asarray(data_loader.cache["PRES"], dtype=np.float64).reshape(-1)
    return mu, sigma, y, z, int(mu.shape[0])


def _nt(mu):
    return int(mu.shape[1] // 2)


def _pack(mu, sigma, y, z=None):
    from evalphys.calibration import ence, gaussian_crps
    from evalphys.constants import DEPTH_BAND_LABELS, DEPTH_BANDS

    nt = _nt(mu)
    out = {
        "overall_concat": {
            "crps_mean": float(np.nanmean(gaussian_crps(mu, sigma, y))),
            "ence": ence(mu, sigma, y).get("ence"),
        },
        "temperature": {
            "crps_mean": float(np.nanmean(gaussian_crps(mu[:, :nt], sigma[:, :nt], y[:, :nt]))),
            "ence": ence(mu[:, :nt], sigma[:, :nt], y[:, :nt]).get("ence"),
        },
        "salinity": {
            "crps_mean": float(np.nanmean(gaussian_crps(mu[:, nt:], sigma[:, nt:], y[:, nt:]))),
            "ence": ence(mu[:, nt:], sigma[:, nt:], y[:, nt:]).get("ence"),
        },
    }
    if z is not None and z.size == nt:
        bands = {}
        for label, (lo, hi) in zip(DEPTH_BAND_LABELS, DEPTH_BANDS):
            m = (z >= lo) & (z < hi) if np.isfinite(hi) else (z >= lo)
            if not np.any(m):
                continue
            bands[label] = {
                "crps_T": float(np.nanmean(gaussian_crps(mu[:, :nt][:, m], sigma[:, :nt][:, m], y[:, :nt][:, m]))),
                "ence_T": ence(mu[:, :nt][:, m], sigma[:, :nt][:, m], y[:, :nt][:, m]).get("ence"),
            }
        out["temperature_by_band"] = bands
    return out


def _alpha_global_var(mu, sigma, y):
    nt = _nt(mu)
    a = np.ones(mu.shape[1], dtype=np.float64)
    for sl in (slice(0, nt), slice(nt, None)):
        e = (mu[:, sl] - y[:, sl]).ravel()
        s = sigma[:, sl].ravel()
        m = np.isfinite(e) & np.isfinite(s) & (s > 0)
        rmse = float(np.sqrt(np.mean(e[m] ** 2)))
        rmv = float(np.sqrt(np.mean(s[m] ** 2)))
        a[sl] = np.clip(rmse / rmv, 0.05, 20.0) if rmv > 0 else 1.0
    return a


def _alpha_band_var(mu, sigma, y, z):
    from evalphys.constants import DEPTH_BANDS

    nt = _nt(mu)
    a = np.ones(mu.shape[1], dtype=np.float64)
    for offset, sl in ((0, slice(0, nt)), (nt, slice(nt, None))):
        for lo, hi in DEPTH_BANDS:
            cols = np.where((z >= lo) & (z < hi) if np.isfinite(hi) else (z >= lo))[0]
            if cols.size == 0:
                continue
            e = (mu[:, sl][:, cols] - y[:, sl][:, cols]).ravel()
            s = sigma[:, sl][:, cols].ravel()
            m = np.isfinite(e) & np.isfinite(s) & (s > 0)
            rmse = float(np.sqrt(np.mean(e[m] ** 2)))
            rmv = float(np.sqrt(np.mean(s[m] ** 2)))
            val = np.clip(rmse / rmv, 0.05, 20.0) if rmv > 0 else 1.0
            a[offset + cols] = val
    return a


def main() -> int:
    from collections import OrderedDict

    import model.model as module_arch
    from model.loss import HEAVE_LOSS_MODES, make_loss
    from parse_config import ConfigParser, validate_config
    from train import ensure_cache, set_seed
    from base.util import prepare_device, read_json

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("-r", "--checkpoint", required=True)
    ap.add_argument("--out", default="../reports/eval_acrps_phys_pca32_b_cal.json")
    args = ap.parse_args()

    cfg_dict = read_json(args.config)
    validate_config(cfg_dict)
    config = ConfigParser(cfg_dict, run_id="")
    set_seed(config.config.get("seed", 42))
    ensure_cache(config)
    device, _ = prepare_device(config["n_gpu"])
    model = config.init_obj("arch", module_arch).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)))
    model.eval()

    import data_loader.data_loaders as module_data

    dl0 = dict(config["data_loader"]["args"])
    dl0["split"] = "val"
    dl0["shuffle"] = False
    data_loader = getattr(module_data, config["data_loader"]["type"])(**dl0)
    cache = data_loader.cache
    outputs = OrderedDict(config["outputs"])
    from base.split_utils import build_split_indices

    train_idx = build_split_indices(
        cache["inputs"].shape[0],
        cache.get("JULD"),
        dl0,
        dataset_tag=cache.get("dataset_tag", "unknown"),
        v2_src=dl0.get("v2_src"),
    )["train"]
    loss_cfg = config.config.get("loss_config") or {}
    loss_fn = make_loss(
        pca_models=cache["pca_models"],
        outputs=outputs,
        weights=cache["weights"],
        device=device,
        loss_scales=config.config.get("loss_scales"),
        loss_config=loss_cfg,
        pres_levels=cache.get("PRES"),
        true_profiles=cache.get("profiles") if loss_cfg.get("mode") in HEAVE_LOSS_MODES else None,
        train_idx=train_idx,
        lat=cache.get("LAT"),
        lon=cache.get("LON"),
        clim_profiles=cache.get("clim_profiles"),
    )

    mu_v, sg_v, y_v, z, n_val = _predict(config, model, loss_fn, "val", device)
    mu_t, sg_t, y_t, _, n_test = _predict(config, model, loss_fn, "test", device)

    recipes = {
        "none": np.ones(mu_v.shape[1], dtype=np.float64),
        "global_var": _alpha_global_var(mu_v, sg_v, y_v),
        "depth_band_var": _alpha_band_var(mu_v, sg_v, y_v, z),
    }
    val_rows = {}
    for name, a in recipes.items():
        pack = _pack(mu_v, sg_v * a, y_v, z)
        val_rows[name] = {
            "ence_T": pack["temperature"]["ence"],
            "ence_S": pack["salinity"]["ence"],
            "crps_T": pack["temperature"]["crps_mean"],
            "alpha_mean": float(np.mean(a)),
            **pack,
        }
    best = min(recipes, key=lambda n: (val_rows[n]["ence_T"] is None, val_rows[n]["ence_T"] or 9e9))
    a = recipes[best]
    test_raw = _pack(mu_t, sg_t, y_t, z)
    test_cal = _pack(mu_t, sg_t * a, y_t, z)
    payload = {
        "checkpoint": str(args.checkpoint),
        "n_val": n_val,
        "n_test": n_test,
        "n_z": _nt(mu_t),
        "ence_max": ENCE_MAX,
        "best_recipe": best,
        "val": val_rows,
        "test_raw": test_raw,
        "test_recalib": test_cal,
        "test_ence_T_raw": test_raw["temperature"]["ence"],
        "test_ence_T_recalib": test_cal["temperature"]["ence"],
        "test_ence_T_pass_recalib": (
            test_cal["temperature"]["ence"] is not None
            and test_cal["temperature"]["ence"] < ENCE_MAX
        ),
        "note": "α fitted on val; picker = val ENCE(T). per-level α omitted.",
    }
    out = Path(args.out)
    if not out.is_absolute():
        out = _ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: payload[k] for k in (
        "best_recipe", "test_ence_T_raw", "test_ence_T_recalib",
        "test_ence_T_pass_recalib", "n_test",
    )}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
