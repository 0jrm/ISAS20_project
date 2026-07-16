#!/usr/bin/env python3
"""Phase 3 acceptance: softplus+spice truth projection RMSE + inversion fidelity."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evalphys.gsw_backend import get_gsw
from evalphys.inversion import sigma0_spice_from_ts, ts_from_sigma0_spice
from evalphys.metrics import sigma0_monotonicity_violations, summarize_physical
from model.density_spice import (
    decode_sigma0_ctrl,
    encode_a_from_sigma0_ctrl,
    normalized_dz,
    upsample_pchip,
)
import torch


def inversion_fidelity(T, S, T_hat, S_hat, converged, depth, lat, lon) -> dict:
    """First-class Phase 3 metric: round-trip / Newton quality."""
    dT = np.abs(T_hat - T)
    dS = np.abs(S_hat - S)
    s0 = sigma0_monotonicity_violations(T_hat, S_hat, depth, lat, lon)
    return {
        "newton_fail_rate": float(1.0 - np.mean(converged)),
        "max_abs_dT": float(np.nanmax(dT)),
        "max_abs_dS": float(np.nanmax(dS)),
        "mean_abs_dT": float(np.nanmean(dT)),
        "mean_abs_dS": float(np.nanmean(dS)),
        "sigma0_violation_rate_profile_post": s0["violation_rate_profile"],
        "sigma0_violation_rate_level_post": s0["violation_rate_level"],
    }


def project_softplus_spice(cache: dict, n_test: int | None = None) -> dict:
    meta = cache["density_spice_meta"]
    z_ctrl = np.asarray(meta["z_ctrl"], dtype=np.float64)
    dz = np.asarray(meta["dz_tilde"], dtype=np.float64)
    mu_t = np.asarray(meta["spice_mean"], dtype=np.float64)
    sd_t = np.asarray(meta["spice_std"], dtype=np.float64)
    pca = cache["pca_models"]["spice"]
    depth = np.asarray(cache["PRES"], dtype=np.float64)
    lat = np.asarray(cache["LAT"], dtype=np.float64)
    lon = np.asarray(cache["LON"], dtype=np.float64)
    T = np.asarray(cache["true_profiles"]["temperature"], dtype=np.float64)
    S = np.asarray(cache["true_profiles"]["salinity"], dtype=np.float64)
    if T.shape[0] != lat.shape[0]:
        T, S = T.T, S.T
    n = lat.shape[0]
    # chronological test tail
    n_te = n_test if n_test is not None else max(1, int(round(0.15 * n)))
    sl = slice(n - n_te, n)
    T, S, lat, lon = T[sl], S[sl], lat[sl], lon[sl]
    n_te = T.shape[0]

    gsw = get_gsw()
    p = gsw.p_from_z(-np.broadcast_to(depth, (n_te, depth.size)), lat[:, None])
    sig, tau = sigma0_spice_from_ts(T, S, p, lon[:, None], lat[:, None])

    sig_ctrl = np.vstack([np.interp(z_ctrl, depth, sig[i]) for i in range(n_te)])
    a = encode_a_from_sigma0_ctrl(sig_ctrl, dz)
    with torch.no_grad():
        sig_hat_c = decode_sigma0_ctrl(torch.from_numpy(a), torch.from_numpy(dz)).numpy()
    # PCHIP to native
    sig_hat = upsample_pchip(sig_hat_c, z_ctrl, depth)

    tau_z = (tau - mu_t) / sd_t
    tau_z = np.nan_to_num(tau_z, nan=0.0)
    tau_hat = pca.inverse_transform(pca.transform(tau_z)) * sd_t + mu_t

    T_hat, S_hat, ok = ts_from_sigma0_spice(sig_hat, tau_hat, p, lon[:, None], lat[:, None])
    phys = summarize_physical(T_hat, S_hat, T, S, depth, lat, lon)
    fid = inversion_fidelity(T, S, T_hat, S_hat, ok, depth, lat, lon)
    dsig = np.diff(sig_hat, axis=1)
    pre_level = float((dsig < -1e-12).mean())
    return {
        "n_test": n_te,
        "ts_rmse": phys["ts_rmse"],
        "static_stability_pred": phys["static_stability_pred"]["1e-08"],
        "sigma0_monotonicity_pred": phys["sigma0_monotonicity_pred"],
        "pre_inversion_sigma0_level_violation": pre_level,
        "inversion_fidelity": fid,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True, help="density_spice train_ready_*.pkl")
    p.add_argument("--out", default="../reports/phase3_proj_cost.json")
    p.add_argument("--n-test", type=int, default=None)
    args = p.parse_args(argv)
    with open(args.cache, "rb") as f:
        cache = pickle.load(f)
    if cache.get("representation") != "density_spice" and "density_spice_meta" not in cache:
        raise SystemExit("cache is not density_spice")
    out = project_softplus_spice(cache, n_test=args.n_test)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
