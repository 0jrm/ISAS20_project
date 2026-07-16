#!/usr/bin/env python3
"""T1 — joint vs separate basis reconstruction stability test (PLAN-v2-recovery Phase 1)."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import gsw
import numpy as np
from scipy.interpolate import PchipInterpolator
from sklearn.decomposition import PCA
from sklearn.isotonic import IsotonicRegression

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from base.split_utils import build_split_indices
from evalphys.inversion import ts_from_sigma0_spice
from evalphys.metrics import summarize_physical, to_teos10


def _level_zscore(profiles: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (profiles - mean) / np.maximum(std, 1e-8)


def _fit_level_stats(profiles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """profiles (n_prof, n_lev)."""
    mean = np.nanmean(profiles, axis=0)
    std = np.nanstd(profiles, axis=0)
    return mean, std


def _reconstruct_separate_pca(
    T: np.ndarray,
    S: np.ndarray,
    *,
    n_comp: int,
    T_mean: np.ndarray,
    T_std: np.ndarray,
    S_mean: np.ndarray,
    S_std: np.ndarray,
    pca_t: PCA,
    pca_s: PCA,
) -> tuple[np.ndarray, np.ndarray]:
    Tz = _level_zscore(T, T_mean, T_std)
    Sz = _level_zscore(S, S_mean, S_std)
    T_hat = pca_t.inverse_transform(pca_t.transform(Tz)) * T_std + T_mean
    S_hat = pca_s.inverse_transform(pca_s.transform(Sz)) * S_std + S_mean
    return T_hat, S_hat


def _fit_separate_pca(T_tr: np.ndarray, S_tr: np.ndarray, n_comp: int):
    T_mean, T_std = _fit_level_stats(T_tr)
    S_mean, S_std = _fit_level_stats(S_tr)
    pca_t = PCA(n_components=n_comp).fit(_level_zscore(T_tr, T_mean, T_std))
    pca_s = PCA(n_components=n_comp).fit(_level_zscore(S_tr, S_mean, S_std))
    return (T_mean, T_std, S_mean, S_std, pca_t, pca_s)


def _fit_joint_eof(T_tr: np.ndarray, S_tr: np.ndarray, n_comp: int):
    T_mean, T_std = _fit_level_stats(T_tr)
    S_mean, S_std = _fit_level_stats(S_tr)
    joint = np.hstack([_level_zscore(T_tr, T_mean, T_std), _level_zscore(S_tr, S_mean, S_std)])
    pca = PCA(n_components=n_comp).fit(joint)
    return T_mean, T_std, S_mean, S_std, pca


def _reconstruct_joint_eof(
    T: np.ndarray,
    S: np.ndarray,
    *,
    T_mean,
    T_std,
    S_mean,
    S_std,
    pca: PCA,
    n_lev: int,
):
    joint = np.hstack([_level_zscore(T, T_mean, T_std), _level_zscore(S, S_mean, S_std)])
    rec = pca.inverse_transform(pca.transform(joint))
    return rec[:, :n_lev] * T_std + T_mean, rec[:, n_lev:] * S_std + S_mean


def _sigma0_spice(T: np.ndarray, S: np.ndarray, depth: np.ndarray, lat: np.ndarray, lon: np.ndarray):
    sa, ct, p = to_teos10(T, S, depth, lat, lon)
    return gsw.sigma0(sa, ct), gsw.spiciness0(sa, ct)


def _fit_density_spice_pca(T_tr, S_tr, depth, lat, lon, n_comp: int):
    sig, tau = _sigma0_spice(T_tr, S_tr, depth, lat, lon)
    sm, ss = _fit_level_stats(sig)
    tm, ts = _fit_level_stats(tau)
    pca_sig = PCA(n_components=n_comp).fit(_level_zscore(sig, sm, ss))
    pca_tau = PCA(n_components=n_comp).fit(_level_zscore(tau, tm, ts))
    return sm, ss, tm, ts, pca_sig, pca_tau


def _reconstruct_density_spice(
    T,
    S,
    depth,
    lat,
    lon,
    *,
    sm,
    ss,
    tm,
    ts,
    pca_sig,
    pca_tau,
):
    sig, tau = _sigma0_spice(T, S, depth, lat, lon)
    sig_hat = pca_sig.inverse_transform(pca_sig.transform(_level_zscore(sig, sm, ss))) * ss + sm
    tau_hat = pca_tau.inverse_transform(pca_tau.transform(_level_zscore(tau, tm, ts))) * ts + tm
    n_prof = T.shape[0]
    n_lev = T.shape[1]
    p = gsw.p_from_z(-np.broadcast_to(depth, (n_prof, n_lev)), lat[:, None])
    T_hat, S_hat, _ = ts_from_sigma0_spice(sig_hat, tau_hat, p, lon[:, None], lat[:, None])
    return T_hat, S_hat


def _control_grid(depth: np.ndarray, K: int = 64) -> np.ndarray:
    z_max = float(np.nanmax(depth))
    z0 = max(float(depth[1]) if depth.size > 1 else 1.0, 1.0)
    return np.logspace(np.log10(z0), np.log10(max(z_max, z0 + 1)), K)


def _monotone_sigma0_profile(sig: np.ndarray, depth: np.ndarray, z_ctrl: np.ndarray) -> np.ndarray:
    ok = np.isfinite(sig) & np.isfinite(depth)
    if ok.sum() < 2:
        return sig.copy()
    sig_c = np.interp(z_ctrl, depth[ok], sig[ok])
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    sig_m = iso.fit_transform(z_ctrl, sig_c)
    return PchipInterpolator(z_ctrl, sig_m, extrapolate=True)(depth)


def _reconstruct_monotone_density(
    T,
    S,
    depth,
    lat,
    lon,
    *,
    tm,
    ts,
    pca_tau,
    z_ctrl,
):
    sig, tau = _sigma0_spice(T, S, depth, lat, lon)
    n_prof = T.shape[0]
    sig_hat = np.empty_like(sig)
    for i in range(n_prof):
        sig_hat[i] = _monotone_sigma0_profile(sig[i], depth, z_ctrl)
    tau_hat = pca_tau.inverse_transform(pca_tau.transform(_level_zscore(tau, tm, ts))) * ts + tm
    n_lev = T.shape[1]
    p = gsw.p_from_z(-np.broadcast_to(depth, (n_prof, n_lev)), lat[:, None])
    T_hat, S_hat, _ = ts_from_sigma0_spice(sig_hat, tau_hat, p, lon[:, None], lat[:, None])
    return T_hat, S_hat


def _headline_rmse(T_hat, S_hat, T, S, depth):
    from evalphys.metrics import ts_rmse_by_band

    bands = ts_rmse_by_band(T_hat, S_hat, T, S, depth)
    t_vals = [v for v in bands["T"].values() if v is not None]
    s_vals = [v for v in bands["S"].values() if v is not None]
    return float(np.mean(t_vals + s_vals))


def _variant_metrics(name, T_hat, S_hat, T, S, depth, lat, lon):
    phys = summarize_physical(T_hat, S_hat, T, S, depth, lat, lon)
    stab = phys["static_stability_pred"]["1e-08"]
    return {
        "name": name,
        "violation_rate_profile": stab["violation_rate_profile"],
        "violation_rate_level": stab["violation_rate_level"],
        "mean_rmse_ts": _headline_rmse(T_hat, S_hat, T, S, depth),
        "drhodz_rmse": phys["drhodz_rmse"]["rmse_overall"],
        "mld_rmse": phys["mld"]["pred_vs_true"]["rmse"],
        "summary": phys,
    }


def run_t1(cache_path: Path, *, n_comp: int = 16, joint_comp: int = 32) -> dict:
    with open(cache_path, "rb") as f:
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
    T_tr, S_tr = T_all[tr], S_all[tr]
    T_te, S_te = T_all[te], S_all[te]
    lat_tr, lon_tr = lat[tr], lon[tr]
    lat_te, lon_te = lat[te], lon[te]

    # A — separate PCA
    stats_a = _fit_separate_pca(T_tr, S_tr, n_comp)
    T_a, S_a = _reconstruct_separate_pca(
        T_te,
        S_te,
        n_comp=n_comp,
        T_mean=stats_a[0],
        T_std=stats_a[1],
        S_mean=stats_a[2],
        S_std=stats_a[3],
        pca_t=stats_a[4],
        pca_s=stats_a[5],
    )

    # B — joint EOF
    T_mean, T_std, S_mean, S_std, pca_j = _fit_joint_eof(T_tr, S_tr, joint_comp)
    T_b, S_b = _reconstruct_joint_eof(T_te, S_te, T_mean=T_mean, T_std=T_std, S_mean=S_mean, S_std=S_std, pca=pca_j, n_lev=depth.size)

    # C — density/spice PCA
    sm, ss, tm, ts, pca_sig, pca_tau = _fit_density_spice_pca(T_tr, S_tr, depth, lat_tr, lon_tr, n_comp)
    T_c, S_c = _reconstruct_density_spice(T_te, S_te, depth, lat_te, lon_te, sm=sm, ss=ss, tm=tm, ts=ts, pca_sig=pca_sig, pca_tau=pca_tau)

    # D — monotone σ₀ + spice PCA
    z_ctrl = _control_grid(depth)
    T_d, S_d = _reconstruct_monotone_density(T_te, S_te, depth, lat_te, lon_te, tm=tm, ts=ts, pca_tau=pca_tau, z_ctrl=z_ctrl)

    results = {
        "A_separate_pca": _variant_metrics("A_separate_pca", T_a, S_a, T_te, S_te, depth, lat_te, lon_te),
        "B_joint_eof": _variant_metrics("B_joint_eof", T_b, S_b, T_te, S_te, depth, lat_te, lon_te),
        "C_density_spice_pca": _variant_metrics("C_density_spice_pca", T_c, S_c, T_te, S_te, depth, lat_te, lon_te),
        "D_monotone_density": _variant_metrics("D_monotone_density", T_d, S_d, T_te, S_te, depth, lat_te, lon_te),
    }

    a_lvl = results["A_separate_pca"]["violation_rate_level"]
    a_prof = results["A_separate_pca"]["violation_rate_profile"]
    decisions = []
    for label in ("B_joint_eof", "C_density_spice_pca"):
        r = results[label]
        if a_lvl > 0 and r["violation_rate_level"] <= a_lvl / 5 and r["mean_rmse_ts"] <= results["A_separate_pca"]["mean_rmse_ts"] * 1.10:
            decisions.append(f"CONFIRMED: {label} cuts level violations ≥5× vs A at ≤10% RMSE cost")
        elif np.isclose(r["violation_rate_level"], a_lvl, rtol=0.05):
            decisions.append(f"ESCALATE: {label} ≈ A — violations may not be basis-induced")
    d = results["D_monotone_density"]
    if d["violation_rate_level"] == 0.0:
        decisions.append(
            f"D monotone: violation_rate_level ≡ 0; RMSE cost vs truth = {d['mean_rmse_ts']:.4f}"
        )
    elif a_lvl > 0:
        ratio = a_lvl / max(d["violation_rate_level"], 1e-12)
        decisions.append(
            f"D monotone: level violation {d['violation_rate_level']:.4f} vs A {a_lvl:.4f} "
            f"({ratio:.1f}× reduction); profile {d['violation_rate_profile']:.3f} vs A {a_prof:.3f}; "
            f"RMSE {d['mean_rmse_ts']:.4f} vs A {results['A_separate_pca']['mean_rmse_ts']:.4f}"
        )
    if not any("CONFIRMED" in x for x in decisions):
        decisions.append(
            "GATE: B/C did not meet ≥5× level-violation cut — review before Phase 3; "
            "D monotone shows partial stability gain (see table)."
        )

    return {
        "cache": str(cache_path),
        "n_train": int(tr.size),
        "n_test": int(te.size),
        "variants": results,
        "decision_rules": decisions,
    }


def _to_md(data: dict) -> str:
    lines = ["# Phase 1 decisive tests — T1 basis stability", ""]
    lines.append(f"Cache: `{data['cache']}`  |  train n={data['n_train']}  test n={data['n_test']}")
    lines.append("")
    lines.append("| variant | viol_rate_profile | viol_rate_level | mean T/S RMSE | dρ/dz RMSE | MLD RMSE |")
    lines.append("|---------|-------------------|-----------------|---------------|------------|----------|")
    for key, r in data["variants"].items():
        lines.append(
            f"| {r['name']} | {r['violation_rate_profile']:.4f} | {r['violation_rate_level']:.4f} | "
            f"{r['mean_rmse_ts']:.4f} | {r['drhodz_rmse']:.4f} | {r['mld_rmse']:.4f} |"
        )
    lines.append("")
    lines.append("## Decision rules")
    for d in data["decision_rules"]:
        lines.append(f"- {d}")
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "-c",
        "--cache",
        default="/unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project/data/cache/train_ready_4411c65ee518.pkl",
    )
    ap.add_argument("--out-json", type=Path, default=_ROOT.parent / "reports" / "t1_basis_stability.json")
    ap.add_argument("--out-md", type=Path, default=None)
    args = ap.parse_args()
    data = run_t1(Path(args.cache))
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    # slim JSON — drop full summarize blobs
    slim = {**data, "variants": {k: {kk: vv for kk, vv in v.items() if kk != "summary"} for k, v in data["variants"].items()}}
    args.out_json.write_text(json.dumps(slim, indent=2) + "\n")
    md_path = args.out_md or args.out_json.with_suffix(".md")
    md_path.write_text(_to_md(slim))
    print(f"wrote {args.out_json} and {md_path}")


if __name__ == "__main__":
    main()
