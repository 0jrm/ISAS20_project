#!/usr/bin/env python3
"""Decompose T1-E deep-band (>800 m) RMSE: grid vs softplus vs spice.

Ablations (shared spice PCA-16 unless noted):
  A_grid_pchip     — linear interp → ctrl → PCHIP (no softplus, no isotonic)
  B_softplus_rt    — linear interp → softplus encode/decode → PCHIP  (= current E density path)
  C_isotonic_pchip — linear interp → isotonic → PCHIP  (= D-style density; E's make_control_grid)
  D_iso_softplus   — linear interp → isotonic → softplus encode/decode → PCHIP  (proposed fix)
  E_full           — B_softplus_rt + spice PCA (current E)
  F_iso_soft_spice — D_iso_softplus + spice PCA (proposed E')

Also reports per-depth softplus raw-increment stats and ctrl-grid spacing below 800 m.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.isotonic import IsotonicRegression

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from base.split_utils import build_split_indices
from evalphys.gsw_backend import get_gsw, set_config_backend
from evalphys.inversion import ts_from_sigma0_spice
from evalphys.metrics import to_teos10, ts_rmse_by_band
from model.density_spice import (
    decode_sigma0_ctrl,
    encode_a_from_sigma0_ctrl,
    make_control_grid,
    normalized_dz,
    upsample_pchip,
)


def _sigma0_spice(T, S, depth, lat, lon):
    gsw = get_gsw()
    sa, ct, p = to_teos10(T, S, depth, lat, lon)
    return gsw.sigma0(sa, ct), gsw.spiciness0(sa, ct)


def _interp_ctrl(sig, depth, z_ctrl):
    out = np.empty((sig.shape[0], z_ctrl.size), dtype=np.float64)
    for i in range(sig.shape[0]):
        ok = np.isfinite(sig[i]) & np.isfinite(depth)
        out[i] = np.interp(z_ctrl, depth[ok], sig[i, ok]) if ok.sum() >= 2 else np.nan
    return out


def _isotonic_ctrl(sig_ctrl, z_ctrl):
    out = np.empty_like(sig_ctrl)
    for i in range(sig_ctrl.shape[0]):
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
        out[i] = iso.fit_transform(z_ctrl, sig_ctrl[i])
    return out


def _softplus_rt(sig_ctrl, dz_tilde, z_ctrl, *, monotone: bool):
    a = encode_a_from_sigma0_ctrl(sig_ctrl, dz_tilde, z_ctrl, monotone=monotone)
    with torch.no_grad():
        return decode_sigma0_ctrl(torch.from_numpy(a), torch.from_numpy(dz_tilde)).numpy()


def _invert(sig_hat, tau_hat, depth, lat, lon):
    gsw = get_gsw()
    n_prof, n_lev = sig_hat.shape[0], depth.size
    p = gsw.p_from_z(-np.broadcast_to(depth, (n_prof, n_lev)), lat[:, None])
    T_hat, S_hat, ok = ts_from_sigma0_spice(sig_hat, tau_hat, p, lon[:, None], lat[:, None])
    return T_hat, S_hat, float(1.0 - ok.mean())


def _band_rmse(T_hat, S_hat, T, S, depth):
    return ts_rmse_by_band(T_hat, S_hat, T, S, depth)


def _deep_sig_rmse(sig_hat, sig, depth, z0=800.0):
    m = depth >= z0
    return float(np.sqrt(np.nanmean((sig_hat[:, m] - sig[:, m]) ** 2)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "-c",
        "--cache",
        default="/unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project/data/cache/train_ready_4411c65ee518.pkl",
    )
    ap.add_argument("--gsw-backend", default="gsw")
    ap.add_argument(
        "--out",
        type=Path,
        default=_ROOT.parent / "reports" / "e_deep_band_diagnostic.md",
    )
    args = ap.parse_args()
    set_config_backend(args.gsw_backend)

    with open(args.cache, "rb") as f:
        cache = pickle.load(f)
    T_all = np.asarray(cache["profiles"]["temperature"], dtype=np.float64).T
    S_all = np.asarray(cache["profiles"]["salinity"], dtype=np.float64).T
    depth = np.asarray(cache["PRES"], dtype=np.float64)
    lat = np.asarray(cache["LAT"], dtype=np.float64)
    lon = np.asarray(cache["LON"], dtype=np.float64)
    n = T_all.shape[0]
    dl_cfg = {
        "split_mode": "chronological",
        "split_config": None,
        "train_frac": 0.7,
        "val_frac": 0.15,
        "test_frac": 0.15,
        "split_seed": 42,
        "unassigned": "exclude",
    }
    splits = build_split_indices(n, cache["JULD"], dl_cfg, dataset_tag=cache.get("dataset_tag", "argo_v2"))
    tr = np.asarray(splits["train"], dtype=int)
    te = np.asarray(splits["test"], dtype=int)
    T_te, S_te = T_all[te], S_all[te]
    lat_te, lon_te = lat[te], lon[te]
    T_tr, S_tr = T_all[tr], S_all[tr]
    lat_tr, lon_tr = lat[tr], lon[tr]

    z_ctrl = make_control_grid(depth, K=64)
    dz = normalized_dz(z_ctrl)
    deep_ctrl = z_ctrl >= 800.0
    dz_phys = np.diff(z_ctrl, prepend=z_ctrl[0])

    sig_tr, tau_tr = _sigma0_spice(T_tr, S_tr, depth, lat_tr, lon_tr)
    sig_te, tau_te = _sigma0_spice(T_te, S_te, depth, lat_te, lon_te)
    tm = np.nanmean(tau_tr, axis=0)
    ts = np.maximum(np.nanstd(tau_tr, axis=0), 1e-8)
    pca_tau = PCA(n_components=16).fit((tau_tr - tm) / ts)
    tau_hat = pca_tau.inverse_transform(pca_tau.transform((tau_te - tm) / ts)) * ts + tm
    tau_truth = tau_te  # oracle spice (isolates density path)

    sig_ctrl = _interp_ctrl(sig_te, depth, z_ctrl)
    sig_iso = _isotonic_ctrl(sig_ctrl, z_ctrl)
    sig_sp = _softplus_rt(sig_ctrl, dz, z_ctrl, monotone=False)  # pathology path
    sig_iso_sp = _softplus_rt(sig_iso, dz, z_ctrl, monotone=False)  # already monotone
    sig_sp_fixed = _softplus_rt(sig_ctrl, dz, z_ctrl, monotone=True)  # default encode

    # Softplus raw-increment stats on linear-interp ctrl (pre-isotonic)
    raw = np.diff(sig_ctrl, axis=1) / np.maximum(dz[1:], 1e-12)
    raw_deep = raw[:, deep_ctrl[1:]]
    floor_hit = float((raw_deep < 1e-12).mean())
    neg_frac = float((raw_deep < 0).mean())

    variants = {
        "A_grid_pchip": upsample_pchip(sig_ctrl, z_ctrl, depth),
        "B_softplus_rt": upsample_pchip(sig_sp, z_ctrl, depth),
        "C_isotonic_pchip": upsample_pchip(sig_iso, z_ctrl, depth),
        "D_iso_softplus": upsample_pchip(sig_iso_sp, z_ctrl, depth),
        "E_encode_monotone": upsample_pchip(sig_sp_fixed, z_ctrl, depth),
    }

    rows = []
    for name, sig_hat in variants.items():
        # density-only with oracle spice
        T_o, S_o, fail_o = _invert(sig_hat, tau_truth, depth, lat_te, lon_te)
        # with spice PCA (full pipeline)
        T_p, S_p, fail_p = _invert(sig_hat, tau_hat, depth, lat_te, lon_te)
        rmse_o = _band_rmse(T_o, S_o, T_te, S_te, depth)
        rmse_p = _band_rmse(T_p, S_p, T_te, S_te, depth)
        rows.append(
            {
                "name": name,
                "sig_rmse_gt800": _deep_sig_rmse(sig_hat, sig_te, depth),
                "T_gt800_oracle_spice": rmse_o["T"][">800"],
                "T_gt800_spice_pca": rmse_p["T"][">800"],
                "S_gt800_spice_pca": rmse_p["S"][">800"],
                "T_bands_spice_pca": {k: rmse_p["T"][k] for k in rmse_p["T"]},
                "newton_fail_spice_pca": fail_p,
                "pre_inv_neg_dsig": int((np.diff(sig_hat, axis=1) < -1e-12).sum()),
            }
        )

    # Roundtrip error: softplus vs identity on ctrl
    rt_err = float(np.nanmax(np.abs(sig_sp - sig_ctrl)))
    rt_err_iso = float(np.nanmax(np.abs(sig_iso_sp - sig_iso)))
    rt_deep = float(np.nanmax(np.abs(sig_sp[:, deep_ctrl] - sig_ctrl[:, deep_ctrl])))

    payload = {
        "cache": str(args.cache),
        "n_test": int(te.size),
        "z_ctrl_gt800": z_ctrl[deep_ctrl].tolist(),
        "dz_phys_gt800": dz_phys[deep_ctrl].tolist(),
        "dz_tilde_gt800": dz[deep_ctrl].tolist(),
        "raw_incr_deep_neg_frac": neg_frac,
        "raw_incr_deep_floor_hit_frac": floor_hit,
        "softplus_rt_max_abs_err": rt_err,
        "softplus_rt_max_abs_err_deep": rt_deep,
        "softplus_rt_max_abs_err_after_iso": rt_err_iso,
        "variants": rows,
    }

    # Markdown
    lines = [
        "# E deep-band diagnostic (>800 m)",
        "",
        f"Cache: `{args.cache}`  |  test n={te.size}  |  gsw=`{args.gsw_backend}`",
        "",
        "## Control grid below 800 m",
        "",
        f"| z_ctrl | Δz phys | Δz̃ |",
        f"|--------|---------|-----|",
    ]
    for z, dp, dt in zip(z_ctrl[deep_ctrl], dz_phys[deep_ctrl], dz[deep_ctrl]):
        lines.append(f"| {z:.1f} | {dp:.1f} | {dt:.3f} |")
    lines += [
        "",
        "## Softplus increment pathology (linear-interp ctrl, deep levels)",
        "",
        f"- Fraction of deep raw increments `< 0`: **{neg_frac:.4f}**",
        f"- Fraction hitting softplus floor (`raw < 1e-12`): **{floor_hit:.4f}**",
        f"- Max |encode∘decode − id| on ctrl: {rt_err:.3e} (deep-only {rt_deep:.3e})",
        f"- Max |encode∘decode − id| after isotonic: {rt_err_iso:.3e}",
        "",
        "## Ablation (T/S via Newton; spice = PCA-16 unless oracle)",
        "",
        "| variant | σ₀ RMSE>800 | T>800 oracle-τ | T>800 spice-PCA | S>800 spice-PCA | pre-inv Δσ₀<0 |",
        "|---------|-------------|----------------|-----------------|-----------------|---------------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['name']} | {r['sig_rmse_gt800']:.5f} | {r['T_gt800_oracle_spice']:.4f} | "
            f"{r['T_gt800_spice_pca']:.4f} | {r['S_gt800_spice_pca']:.4f} | {r['pre_inv_neg_dsig']} |"
        )
    lines += [
        "",
        "## Interpretation keys",
        "",
        "- If `A ≈ B` and both ≫ `C`: softplus roundtrip is fine; **linear interp without isotonic** is the deep cost.",
        "- If `B ≫ A`: softplus floor / negative-increment clamp is the cost.",
        "- If oracle-τ ≪ spice-PCA for all: spice PCA (not density) drives deep T error.",
        "- `D_iso_softplus` ≈ `C` ⇒ isotonic-before-encode is the fix for the Phase-3 path.",
        "",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    args.out.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
