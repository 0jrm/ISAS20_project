#!/usr/bin/env python3
"""Export heave-residual decode for TSIS later: warp, steric, σ_η — no insertion."""

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

from evalphys.constants import STERIC_LC_RMS_CM
from evalphys.metrics import steric_vs_adt
from model.heave import HeaveResidual, decode_warp, warp_sigma_meters
from model.prob_head import split_mu_sigma, softplus_sigma


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-r", "--resume", required=True, help="heave-residual checkpoint")
    parser.add_argument("--out", required=True, help="npz output path")
    parser.add_argument("--split", default="test")
    args = parser.parse_args(argv)

    from base.util import read_json
    from data_loader.data_loaders import NeSPReSODataLoader
    from parse_config import validate_config
    from preproc.export_v2_cache import build_argo_cache

    cfg = read_json(args.config)
    validate_config(cfg)
    cache_path = cfg["data_loader"]["args"].get("cache_path") or build_argo_cache(cfg)
    cfg["data_loader"]["args"]["cache_path"] = cache_path
    cfg["data_loader"]["args"]["split"] = args.split
    cfg["data_loader"]["args"]["input_params"] = cfg.get("input_params")
    dl = NeSPReSODataLoader(**cfg["data_loader"]["args"])
    cache = dl.cache

    ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
    arch = (ckpt.get("config") or cfg)["arch"]["args"]
    model = HeaveResidual(**arch)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    xs, ids = [], []
    for batch in dl:
        xb, _tb, ib = batch[:3]
        xs.append(xb)
        ids.append(ib)
    x = torch.cat(xs, dim=0)
    local = torch.cat(ids, dim=0).numpy()
    with torch.no_grad():
        out = model(x)
    d = int(sum(cfg["outputs"].values()))
    if out.shape[-1] == 2 * d:
        mu, raw_s = split_mu_sigma(out, d)
        sigma = softplus_sigma(raw_s)
    else:
        mu, sigma = out, None
    z_bot = float(np.asarray(cache["PRES"]).ravel()[-1])
    mld, d26, stretch = decode_warp(mu[:, :3], z_bot)
    payload = {
        "indices": local,
        "mld_m": mld.numpy(),
        "d26_m": d26.numpy(),
        "stretch": stretch.numpy(),
        "mu": mu.numpy(),
    }
    if sigma is not None:
        payload["sigma"] = sigma.numpy()
        _sig_mld, sig_d26 = warp_sigma_meters(mu[:, :3], sigma[:, :3])
        payload["sigma_d26"] = sig_d26.numpy()
        payload["sigma_mld"] = _sig_mld.numpy()
        payload["sigma_eta"] = sigma[:, :3].numpy()
        from evalphys.calibration import ence, season_from_juld
        from evalphys.metrics import isotherm_depth
        from scipy.stats import spearmanr

        n = cache["LAT"].shape[0]
        T = np.asarray(cache["profiles"]["temperature"])
        if T.shape[0] != n:
            T = T.T
        z = np.asarray(cache["PRES"]).ravel()
        d26_true, _ = isotherm_depth(T[local], z, 26.0)
        payload["ence_d26"] = np.array([json.dumps(ence(d26.numpy(), sig_d26.numpy(), d26_true))])
        juld = np.asarray(cache["JULD"])[local]
        season = season_from_juld(juld, dataset_tag=cache.get("dataset_tag", "argo_v2"))
        jja = season == "JJA"
        if jja.any():
            payload["ence_d26_jja"] = np.array(
                [json.dumps(ence(d26.numpy()[jja], sig_d26.numpy()[jja], d26_true[jja]))]
            )
            rho, _p = spearmanr(sig_d26.numpy()[jja], np.abs(d26.numpy()[jja] - d26_true[jja]))
            payload["spearman_sigma_d26_jja"] = np.array(
                [float(rho) if np.isfinite(rho) else np.nan]
            )
    sla = cache.get("ssh_obs_sla")
    T = np.asarray(cache["profiles"]["temperature"])
    S = np.asarray(cache["profiles"]["salinity"])
    n_all = cache["LAT"].shape[0]
    if T.shape[0] != n_all:
        T, S = T.T, S.T
    lat = cache["LAT"][local]
    lon = cache["LON"][local]
    z = np.asarray(cache["PRES"]).ravel()
    if sla is not None:
        st = steric_vs_adt(T[local], S[local], z, lat, lon, np.asarray(sla)[local])
        payload["steric_json"] = np.array([json.dumps(st)])
        payload["promote"] = np.array([bool(st.get("lc_pass"))])
        if st.get("rms_cm_lc") is not None and st["rms_cm_lc"] > STERIC_LC_RMS_CM:
            print(f"PROMOTION FALSE: LC steric RMS {st['rms_cm_lc']:.2f} cm > {STERIC_LC_RMS_CM} cm")
    np.savez_compressed(args.out, **payload)
    print(f"wrote {args.out} n={len(local)} (no TSIS insertion)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
