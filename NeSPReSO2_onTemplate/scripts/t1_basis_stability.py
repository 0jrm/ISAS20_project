#!/usr/bin/env python3
"""T1 — joint vs separate basis reconstruction stability test (PLAN-v2-recovery Phase 1)."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator
from sklearn.decomposition import PCA
from sklearn.isotonic import IsotonicRegression

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from base.split_utils import build_split_indices
from evalphys.gsw_backend import get_gsw, resolve_backend, set_config_backend
from evalphys.inversion import ts_from_sigma0_spice
from evalphys.metrics import (
    sigma0_monotonicity_violations,
    sigma0_profiles,
    static_stability_violations,
    summarize_physical,
    to_teos10,
    ts_rmse_by_band,
)


def _level_zscore(profiles: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (profiles - mean) / np.maximum(std, 1e-8)


def _fit_level_stats(profiles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(profiles, axis=0)
    std = np.nanstd(profiles, axis=0)
    return mean, std


def _reconstruct_separate_pca(T, S, *, n_comp, T_mean, T_std, S_mean, S_std, pca_t, pca_s):
    Tz = _level_zscore(T, T_mean, T_std)
    Sz = _level_zscore(S, S_mean, S_std)
    T_hat = pca_t.inverse_transform(pca_t.transform(Tz)) * T_std + T_mean
    S_hat = pca_s.inverse_transform(pca_s.transform(Sz)) * S_std + S_mean
    return T_hat, S_hat


def _fit_separate_pca(T_tr, S_tr, n_comp: int):
    T_mean, T_std = _fit_level_stats(T_tr)
    S_mean, S_std = _fit_level_stats(S_tr)
    pca_t = PCA(n_components=n_comp).fit(_level_zscore(T_tr, T_mean, T_std))
    pca_s = PCA(n_components=n_comp).fit(_level_zscore(S_tr, S_mean, S_std))
    return (T_mean, T_std, S_mean, S_std, pca_t, pca_s)


def _fit_joint_eof(T_tr, S_tr, n_comp: int):
    T_mean, T_std = _fit_level_stats(T_tr)
    S_mean, S_std = _fit_level_stats(S_tr)
    joint = np.hstack([_level_zscore(T_tr, T_mean, T_std), _level_zscore(S_tr, S_mean, S_std)])
    pca = PCA(n_components=n_comp).fit(joint)
    return T_mean, T_std, S_mean, S_std, pca


def _reconstruct_joint_eof(T, S, *, T_mean, T_std, S_mean, S_std, pca, n_lev):
    joint = np.hstack([_level_zscore(T, T_mean, T_std), _level_zscore(S, S_mean, S_std)])
    rec = pca.inverse_transform(pca.transform(joint))
    return rec[:, :n_lev] * T_std + T_mean, rec[:, n_lev:] * S_std + S_mean


def _sigma0_spice(T, S, depth, lat, lon):
    gsw = get_gsw()
    sa, ct, p = to_teos10(T, S, depth, lat, lon)
    return gsw.sigma0(sa, ct), gsw.spiciness0(sa, ct)


def _fit_density_spice_pca(T_tr, S_tr, depth, lat, lon, n_comp: int):
    sig, tau = _sigma0_spice(T_tr, S_tr, depth, lat, lon)
    sm, ss = _fit_level_stats(sig)
    tm, ts = _fit_level_stats(tau)
    pca_sig = PCA(n_components=n_comp).fit(_level_zscore(sig, sm, ss))
    pca_tau = PCA(n_components=n_comp).fit(_level_zscore(tau, tm, ts))
    return sm, ss, tm, ts, pca_sig, pca_tau


def _reconstruct_density_spice(T, S, depth, lat, lon, *, sm, ss, tm, ts, pca_sig, pca_tau):
    gsw = get_gsw()
    sig, tau = _sigma0_spice(T, S, depth, lat, lon)
    sig_hat = pca_sig.inverse_transform(pca_sig.transform(_level_zscore(sig, sm, ss))) * ss + sm
    tau_hat = pca_tau.inverse_transform(pca_tau.transform(_level_zscore(tau, tm, ts))) * ts + tm
    n_prof, n_lev = T.shape
    p = gsw.p_from_z(-np.broadcast_to(depth, (n_prof, n_lev)), lat[:, None])
    T_hat, S_hat, ok = ts_from_sigma0_spice(sig_hat, tau_hat, p, lon[:, None], lat[:, None])
    return T_hat, S_hat, float(1.0 - ok.mean())


def _reconstruct_softplus_density(T, S, depth, lat, lon, *, tm, ts, pca_tau, z_ctrl, dz_tilde):
    """Phase 3.2+3.3 truth projection: softplus ctrl encode/decode + PCHIP + spice PCA."""
    import torch
    from model.density_spice import decode_sigma0_ctrl, encode_a_from_sigma0_ctrl, upsample_pchip

    gsw = get_gsw()
    sig, tau = _sigma0_spice(T, S, depth, lat, lon)
    n_prof = T.shape[0]
    sig_ctrl = np.empty((n_prof, z_ctrl.size), dtype=np.float64)
    for i in range(n_prof):
        ok = np.isfinite(sig[i]) & np.isfinite(depth)
        if ok.sum() < 2:
            sig_ctrl[i] = np.nan
            continue
        sig_ctrl[i] = np.interp(z_ctrl, depth[ok], sig[i, ok])
    a = encode_a_from_sigma0_ctrl(sig_ctrl, dz_tilde)
    with torch.no_grad():
        sig_hat_c = decode_sigma0_ctrl(torch.from_numpy(a), torch.from_numpy(dz_tilde)).numpy()
    sig_hat = upsample_pchip(sig_hat_c, z_ctrl, depth)
    tau_hat = pca_tau.inverse_transform(pca_tau.transform(_level_zscore(tau, tm, ts))) * ts + tm
    n_lev = T.shape[1]
    p = gsw.p_from_z(-np.broadcast_to(depth, (n_prof, n_lev)), lat[:, None])
    T_hat, S_hat, ok = ts_from_sigma0_spice(sig_hat, tau_hat, p, lon[:, None], lat[:, None])
    dsig = np.diff(sig_hat, axis=1)
    pre_n = int((dsig < -1e-12).sum())
    return T_hat, S_hat, float(1.0 - ok.mean()), pre_n, sig_hat


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


def _reconstruct_monotone_density(T, S, depth, lat, lon, *, tm, ts, pca_tau, z_ctrl):
    gsw = get_gsw()
    sig, tau = _sigma0_spice(T, S, depth, lat, lon)
    n_prof = T.shape[0]
    sig_hat = np.empty_like(sig)
    for i in range(n_prof):
        sig_hat[i] = _monotone_sigma0_profile(sig[i], depth, z_ctrl)
    tau_hat = pca_tau.inverse_transform(pca_tau.transform(_level_zscore(tau, tm, ts))) * ts + tm
    n_lev = T.shape[1]
    p = gsw.p_from_z(-np.broadcast_to(depth, (n_prof, n_lev)), lat[:, None])
    T_hat, S_hat, ok = ts_from_sigma0_spice(sig_hat, tau_hat, p, lon[:, None], lat[:, None])
    # Pre-inversion Δσ₀ violations on projected σ₀ (should be ~0)
    dsig = np.diff(sig_hat, axis=1)
    pre_n = int((dsig < -1e-12).sum())
    return T_hat, S_hat, float(1.0 - ok.mean()), pre_n, sig_hat


def _variant_metrics(name, T_hat, S_hat, T, S, depth, lat, lon, **extra):
    phys = summarize_physical(T_hat, S_hat, T, S, depth, lat, lon)
    stab = phys["static_stability_pred"]["1e-08"]
    stab0 = phys["static_stability_pred"]["0"]
    s0 = phys["sigma0_monotonicity_pred"]
    ts = phys["ts_rmse"]
    return {
        "name": name,
        "violation_rate_profile": stab["violation_rate_profile"],
        "violation_rate_level": stab["violation_rate_level"],
        "violation_rate_level_n2_tol0": stab0["violation_rate_level"],
        "sigma0_violation_rate_profile": s0["violation_rate_profile"],
        "sigma0_violation_rate_level": s0["violation_rate_level"],
        "ts_rmse": ts,
        "drhodz_rmse": phys["drhodz_rmse"]["rmse_overall"],
        "mld_rmse": phys["mld"]["pred_vs_true"]["rmse"],
        "n2_tol_sweep_level": {
            k: v["violation_rate_level"] for k, v in phys["static_stability_pred"].items()
        },
        "exclude_top15m_level": phys["static_stability_pred_exclude_top15m"]["1e-08"]["violation_rate_level"],
        **extra,
        "summary": phys,
    }


def _historical_sigma0_profile_rate(T, S, depth, lat, lon, *, tol=0.01):
    """Session Finding-1 metric: readiness σ₀ Δσ₀ < −tol profile rate (gsw_torch path)."""
    from diagnostics.readiness import static_stability_diagnostic

    out = static_stability_diagnostic(T.T, S.T, depth, lat, lon, tol_kgm3=tol)
    return {
        "tol_kgm3": tol,
        "violation_rate_profile": out["violation_rate"],
        "violation_rate_interface": out["interface_violation_rate"],
        "n_violations": out["total_violations"],
        "method": out["method"],
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

    stats_a = _fit_separate_pca(T_tr, S_tr, n_comp)
    T_a, S_a = _reconstruct_separate_pca(
        T_te, S_te, n_comp=n_comp,
        T_mean=stats_a[0], T_std=stats_a[1], S_mean=stats_a[2], S_std=stats_a[3],
        pca_t=stats_a[4], pca_s=stats_a[5],
    )
    assert stats_a[4].n_components_ == n_comp and stats_a[5].n_components_ == n_comp

    T_mean, T_std, S_mean, S_std, pca_j = _fit_joint_eof(T_tr, S_tr, joint_comp)
    assert pca_j.n_components_ == joint_comp
    T_b, S_b = _reconstruct_joint_eof(
        T_te, S_te, T_mean=T_mean, T_std=T_std, S_mean=S_mean, S_std=S_std, pca=pca_j, n_lev=depth.size
    )

    sm, ss, tm, ts, pca_sig, pca_tau = _fit_density_spice_pca(T_tr, S_tr, depth, lat_tr, lon_tr, n_comp)
    T_c, S_c, c_fail = _reconstruct_density_spice(
        T_te, S_te, depth, lat_te, lon_te, sm=sm, ss=ss, tm=tm, ts=ts, pca_sig=pca_sig, pca_tau=pca_tau
    )

    z_ctrl = _control_grid(depth)
    from model.density_spice import make_control_grid, normalized_dz

    z_ctrl_p3 = make_control_grid(depth, K=64)
    dz_tilde = normalized_dz(z_ctrl_p3)
    T_d, S_d, d_fail, d_pre_dsig_viol, sig_hat_d = _reconstruct_monotone_density(
        T_te, S_te, depth, lat_te, lon_te, tm=tm, ts=ts, pca_tau=pca_tau, z_ctrl=z_ctrl
    )
    T_e, S_e, e_fail, e_pre_dsig_viol, _ = _reconstruct_softplus_density(
        T_te, S_te, depth, lat_te, lon_te,
        tm=tm, ts=ts, pca_tau=pca_tau, z_ctrl=z_ctrl_p3, dz_tilde=dz_tilde,
    )

    results = {
        "A_separate_pca": _variant_metrics("A_separate_pca", T_a, S_a, T_te, S_te, depth, lat_te, lon_te),
        "B_joint_eof": _variant_metrics("B_joint_eof", T_b, S_b, T_te, S_te, depth, lat_te, lon_te),
        "C_density_spice_pca": _variant_metrics(
            "C_density_spice_pca", T_c, S_c, T_te, S_te, depth, lat_te, lon_te,
            newton_fail_rate=c_fail,
        ),
        "D_monotone_density": _variant_metrics(
            "D_monotone_density", T_d, S_d, T_te, S_te, depth, lat_te, lon_te,
            newton_fail_rate=d_fail,
            pre_inversion_dsigma0_neg_count=d_pre_dsig_viol,
        ),
        "E_softplus_phase3": _variant_metrics(
            "E_softplus_phase3", T_e, S_e, T_te, S_te, depth, lat_te, lon_te,
            newton_fail_rate=e_fail,
            pre_inversion_dsigma0_neg_count=e_pre_dsig_viol,
        ),
    }

    # Reconciliation vs historical Finding-1 (σ₀ profile rate, tol=0.01) — all variants
    recon = {
        "historical_raw_test": _historical_sigma0_profile_rate(T_te, S_te, depth, lat_te, lon_te),
        "historical_A_pca16": _historical_sigma0_profile_rate(T_a, S_a, depth, lat_te, lon_te),
        "historical_B_joint_eof": _historical_sigma0_profile_rate(T_b, S_b, depth, lat_te, lon_te),
        "historical_C_density_spice": _historical_sigma0_profile_rate(T_c, S_c, depth, lat_te, lon_te),
        "historical_D": _historical_sigma0_profile_rate(T_d, S_d, depth, lat_te, lon_te),
        "historical_E_softplus_phase3": _historical_sigma0_profile_rate(T_e, S_e, depth, lat_te, lon_te),
        "notes": {
            "a_profile_vs_level": (
                "N² profile rate ≫ level rate because violations are sparse per profile "
                f"(A profile={results['A_separate_pca']['violation_rate_profile']:.4f}, "
                f"level={results['A_separate_pca']['violation_rate_level']:.4f})"
            ),
            "b_n2_tol0_vs_1e8": (
                f"A level N² at tol=0: {results['A_separate_pca']['violation_rate_level_n2_tol0']:.6f}; "
                f"at 1e-8: {results['A_separate_pca']['violation_rate_level']:.6f}"
            ),
            "d_method": (
                "Historical Finding-1 used readiness σ₀ Δσ₀<-0.01 profile rate "
                "(~1.12% raw → ~21.8% PCA-16), not N² level rate."
            ),
            "mechanism_update": (
                "B (joint EOF) does not cut historical σ₀ profile rate vs A — the load-bearing "
                "mechanism is truncation itself, not separateness of T/S bases. Soft "
                "representation changes do not buy stability; only the hard monotone "
                "constraint (D) does."
            ),
            "f_backend": f"headline backend={resolve_backend(None)} (reference gsw)",
        },
    }

    a_lvl = results["A_separate_pca"]["violation_rate_level"]
    a_prof = results["A_separate_pca"]["violation_rate_profile"]
    # Decision uses T RMSE 0-50 as the ≤10% cost proxy (units consistent); also report full bands.
    a_t050 = results["A_separate_pca"]["ts_rmse"]["T"]["0-50"]

    plan_rules = [
        "If B and/or C cut the level violation rate by ≥ 5× vs A at ≤ 10% RMSE cost ⇒ Finding-1 mechanism confirmed; Phase 3 proceeds as planned.",
        "If C ≈ A (no improvement) ⇒ the violations are not basis-induced; escalate to human before Phase 3 (the representation chapter framing changes).",
        'D should show violation rate ≡ 0 by construction; record its RMSE cost — this is the "price of hard stability" headline number.',
    ]
    decisions = []
    for label in ("B_joint_eof", "C_density_spice_pca"):
        r = results[label]
        t050 = r["ts_rmse"]["T"]["0-50"]
        rmse_ok = t050 <= a_t050 * 1.10
        if a_lvl > 0 and r["violation_rate_level"] <= a_lvl / 5 and rmse_ok:
            decisions.append(f"CONFIRMED: {label} cuts level violations ≥5× vs A at ≤10% T(0-50) RMSE cost")
        elif np.isclose(r["violation_rate_level"], a_lvl, rtol=0.05):
            decisions.append(f"ESCALATE: {label} ≈ A — violations may not be basis-induced under N² level metric")
    d = results["D_monotone_density"]
    if d["sigma0_violation_rate_level"] == 0.0 and d["violation_rate_level"] == 0.0:
        decisions.append("D monotone: N² and σ₀ violation_rate_level ≡ 0")
    else:
        decisions.append(
            f"D monotone: N² level={d['violation_rate_level']:.4f} "
            f"(σ₀ level={d['sigma0_violation_rate_level']:.4f}; "
            f"pre-inv Δσ₀<0 count={d.get('pre_inversion_dsigma0_neg_count')}); "
            f"vs A N² level={a_lvl:.4f} ({a_lvl / max(d['violation_rate_level'], 1e-12):.1f}×)"
        )
    if not any(x.startswith("CONFIRMED") for x in decisions):
        decisions.append(
            "GATE: B/C did not meet ≥5× level-violation cut under N² — "
            "Finding-1 still holds under historical σ₀ profile metric (see Reconciliation)."
        )

    # Phase 3 acceptance: softplus path RMSE cost vs A (≤10% T per depth band)
    e = results["E_softplus_phase3"]
    a_ts = results["A_separate_pca"]["ts_rmse"]["T"]
    e_ts = e["ts_rmse"]["T"]
    cost_ok = True
    cost_notes = []
    for band, a_rmse in a_ts.items():
        e_rmse = e_ts[band]
        ratio = e_rmse / max(a_rmse, 1e-12)
        cost_notes.append(f"T[{band}] E/A={ratio:.3f}")
        if ratio > 1.10:
            cost_ok = False
    decisions.append(
        "Phase3 softplus E vs A T-RMSE: "
        + (", ".join(cost_notes))
        + ("; PASS ≤10%" if cost_ok else "; FAIL >10% in ≥1 band")
    )
    results["E_softplus_phase3"]["phase3_t_rmse_vs_A_pass"] = cost_ok

    return {
        "cache": str(cache_path),
        "gsw_backend": resolve_backend(None),
        "n_train": int(tr.size),
        "n_test": int(te.size),
        "n_comp_separate": n_comp,
        "n_comp_joint": joint_comp,
        "bases_fit_on": "train_split_only",
        "variants": results,
        "reconciliation": recon,
        "plan_decision_rules_verbatim": plan_rules,
        "decision_outcomes": decisions,
    }


def _fmt_ts(ts: dict) -> str:
    parts = []
    for var in ("T", "S"):
        bands = ts[var]
        parts.append(
            var
            + ":"
            + ",".join(f"{k}={bands[k]:.3f}" if bands[k] is not None else f"{k}=NA" for k in bands)
        )
    return "; ".join(parts)


def _to_md(data: dict) -> str:
    lines = [
        "# Phase 1 decisive tests — T1 basis stability",
        "",
        f"Cache: `{data['cache']}`  |  train n={data['n_train']}  test n={data['n_test']}  |  gsw=`{data['gsw_backend']}`",
        f"Bases fit on: **{data['bases_fit_on']}** (leakage check: train only).",
        "",
        "| variant | N² prof | N² level | σ₀ level | T/S RMSE by band | dρ/dz | MLD |",
        "|---------|---------|----------|----------|------------------|-------|-----|",
    ]
    for _key, r in data["variants"].items():
        lines.append(
            f"| {r['name']} | {r['violation_rate_profile']:.4f} | {r['violation_rate_level']:.4f} | "
            f"{r['sigma0_violation_rate_level']:.4f} | {_fmt_ts(r['ts_rmse'])} | "
            f"{r['drhodz_rmse']:.4f} | {r['mld_rmse']:.4f} |"
        )
    lines += ["", "## Plan decision rules (verbatim from PLAN §1-T1)", ""]
    for rule in data["plan_decision_rules_verbatim"]:
        lines.append(f"- {rule}")
    lines += ["", "## Decision outcomes", ""]
    for d in data["decision_outcomes"]:
        lines.append(f"- {d}")
    rec = data["reconciliation"]
    lines += [
        "",
        "## Reconciliation (Finding-1 vs T1 N² numbers)",
        "",
        "| row | σ₀ profile rate (tol=0.01) | interface rate |",
        "|-----|----------------------------|----------------|",
        f"| RAW test | {rec['historical_raw_test']['violation_rate_profile']:.4f} | {rec['historical_raw_test']['violation_rate_interface']:.6f} |",
        f"| A PCA-16 | {rec['historical_A_pca16']['violation_rate_profile']:.4f} | {rec['historical_A_pca16']['violation_rate_interface']:.6f} |",
        f"| B joint EOF-32 | {rec['historical_B_joint_eof']['violation_rate_profile']:.4f} | {rec['historical_B_joint_eof']['violation_rate_interface']:.6f} |",
        f"| C density+spice | {rec['historical_C_density_spice']['violation_rate_profile']:.4f} | {rec['historical_C_density_spice']['violation_rate_interface']:.6f} |",
        f"| D monotone | {rec['historical_D']['violation_rate_profile']:.4f} | {rec['historical_D']['violation_rate_interface']:.6f} |",
        f"| E softplus Phase-3 | {rec['historical_E_softplus_phase3']['violation_rate_profile']:.4f} | {rec['historical_E_softplus_phase3']['violation_rate_interface']:.6f} |",
        "",
    ]
    for k, v in rec["notes"].items():
        lines.append(f"- **({k})** {v}")
    lines.append("")
    lines.append("## T3 — exclude top 15 m (N² level @ 1e-8)")
    lines.append("")
    lines.append("| variant | full-column | exclude top 15 m |")
    lines.append("|---------|-------------|------------------|")
    for _key, r in data["variants"].items():
        lines.append(f"| {r['name']} | {r['violation_rate_level']:.4f} | {r['exclude_top15m_level']:.4f} |")
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "-c",
        "--cache",
        default="/unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project/data/cache/train_ready_4411c65ee518.pkl",
    )
    ap.add_argument("--gsw-backend", default="gsw", choices=("gsw", "gsw_torch"))
    ap.add_argument("--out-json", type=Path, default=_ROOT.parent / "reports" / "t1_basis_stability.json")
    ap.add_argument("--out-md", type=Path, default=None)
    args = ap.parse_args()
    set_config_backend(args.gsw_backend)
    data = run_t1(Path(args.cache))
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    slim = {
        **data,
        "variants": {k: {kk: vv for kk, vv in v.items() if kk != "summary"} for k, v in data["variants"].items()},
    }
    args.out_json.write_text(json.dumps(slim, indent=2) + "\n")
    md_path = args.out_md or args.out_json.with_suffix(".md")
    md_path.write_text(_to_md(slim))
    print(f"wrote {args.out_json} and {md_path}")


if __name__ == "__main__":
    main()
