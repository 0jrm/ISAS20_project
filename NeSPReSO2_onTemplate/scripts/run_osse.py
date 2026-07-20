#!/usr/bin/env python3
"""Phase 6 toy OSSE runner (PLAN §6 / osse_preregistration.md).

Column-wise OI:
  x_a = x_b + B Hᵀ (H B Hᵀ + R)⁻¹ (y − H x_b)

Modes
------
cast_column (default, v1)
  Truth = ARGO profiles at real cast locations (2021 by default).
  Map-level ISAS20 grid + L_h spread deferred (no 2021 ISAS year on disk).
  Still exercises E0–E5 R_fixed / R_cal / QC claims at cast columns.

``--selfcheck`` validates OI + A-path R_cal shapes without cache I/O.
"""

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

E_TABLE = ("E0", "E1", "E2", "E3", "E4", "E5")
L_V_M = 150.0
L_H_KM = 100.0
FLOOR = 1e-8
# OSSE scoring bands (prereg §3) — distinct from evalphys DEPTH_BANDS
OSSE_BANDS = ((0.0, 100.0), (100.0, 300.0), (300.0, 700.0), (700.0, np.inf))
OSSE_BAND_LABELS = ("0-100", "100-300", "300-700", ">700")


def gaussian_corr(z: np.ndarray, L: float) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    d = z[:, None] - z[None, :]
    return np.exp(-0.5 * (d / float(L)) ** 2)


def vertical_B(z: np.ndarray, sigma_clim: np.ndarray, *, L_v: float = L_V_M) -> np.ndarray:
    sig = np.asarray(sigma_clim, dtype=np.float64).reshape(-1)
    c = gaussian_corr(z, L_v)
    return (sig[:, None] * c) * sig[None, :]


def oi_update(
    x_b: np.ndarray,
    y: np.ndarray,
    B: np.ndarray,
    R: np.ndarray,
    *,
    H: np.ndarray | None = None,
) -> np.ndarray:
    xb = np.asarray(x_b, dtype=np.float64).reshape(-1)
    yy = np.asarray(y, dtype=np.float64).reshape(-1)
    Bb = np.asarray(B, dtype=np.float64)
    Rr = np.asarray(R, dtype=np.float64)
    n = xb.size
    Hm = np.eye(n) if H is None else np.asarray(H, dtype=np.float64)
    if Rr.ndim == 1:
        Rr = np.diag(Rr)
    K = Bb @ Hm.T @ np.linalg.solve(Hm @ Bb @ Hm.T + Rr, np.eye(Hm.shape[0]))
    return xb + K @ (yy - Hm @ xb)


def r_fixed_from_rmse(rmse: np.ndarray, *, floor: float = FLOOR) -> np.ndarray:
    return np.maximum(np.asarray(rmse, dtype=np.float64).reshape(-1) ** 2, float(floor))


def subsample_levels(n_z: int, n_keep: int = 60) -> np.ndarray:
    if n_keep >= n_z:
        return np.arange(n_z)
    return np.unique(np.linspace(0, n_z - 1, int(n_keep)).astype(int))


def band_rmse(pred: np.ndarray, truth: np.ndarray, depth: np.ndarray) -> dict[str, float]:
    err2 = (np.asarray(pred) - np.asarray(truth)) ** 2
    out = {"overall": float(np.sqrt(np.nanmean(err2)))}
    z = np.asarray(depth).reshape(-1)
    for lab, (lo, hi) in zip(OSSE_BAND_LABELS, OSSE_BANDS):
        m = (z >= lo) & (z < hi) if np.isfinite(hi) else (z >= lo)
        if np.any(m):
            out[lab] = float(np.sqrt(np.nanmean(err2[:, m])))
    return out


def monthly_clim(T: np.ndarray, months: np.ndarray) -> dict[int, np.ndarray]:
    """month 1..12 → mean profile (n_z,)."""
    out = {}
    for m in range(1, 13):
        sel = months == m
        out[m] = np.nanmean(T[sel], axis=0) if np.any(sel) else np.nanmean(T, axis=0)
    return out


def selfcheck() -> None:
    from dacov import assert_psd, export_ts_covariance_pca, localize_covariance
    from sklearn.decomposition import PCA

    rng = np.random.default_rng(0)
    z = np.linspace(0, 500, 20)
    B = vertical_B(z, 0.5 + 0.01 * z / 100.0, L_v=L_V_M)
    assert_psd(B, tol=-1e-8)
    x_true = np.sin(z / 80.0)
    x_b = np.zeros_like(x_true)
    R = r_fixed_from_rmse(np.full(z.size, 0.2))
    x_a = oi_update(x_b, x_true, B, R)
    assert float(np.sqrt(np.mean((x_a - x_true) ** 2))) < float(
        np.sqrt(np.mean((x_b - x_true) ** 2))
    )
    pca_T = PCA(n_components=4).fit(rng.normal(size=(40, z.size)))
    pca_S = PCA(n_components=4).fit(rng.normal(size=(40, z.size)))
    pack = export_ts_covariance_pca(np.ones(8) * 0.2, pca_T, pca_S, n_T=4, floor=FLOOR)
    assert_psd(pack["Sigma_T"])
    _ = oi_update(x_b, x_true, B, pack["Sigma_T"])
    # Structured (localized full) R_cal: PSD, diag preserved, OI still improves.
    R_loc = localize_covariance(pack["Sigma_T"], z, L_loc=L_V_M, floor=FLOOR)
    assert_psd(R_loc)
    assert np.allclose(np.diag(R_loc), np.diag(pack["Sigma_T"]) + FLOOR), "localize changed diag"
    x_a_loc = oi_update(x_b, x_true, B, R_loc)
    assert float(np.sqrt(np.mean((x_a_loc - x_true) ** 2))) < float(
        np.sqrt(np.mean((x_b - x_true) ** 2))
    ), "localized full R_cal did not improve on background"
    # E5 threshold rule
    sigbar = np.array([0.1, 0.2, 0.3, 0.4])
    tau = float(np.median(sigbar))
    assert tau == 0.25
    print(f"run_osse selfcheck OK (E-table={E_TABLE}; mode=cast_column)")


def _load_winner(cfg_path: Path, ckpt: Path):
    from scripts.eval_phase4_crps import predict_mu_sigma
    from scripts.phase5_physical_space_score import _decode_pcs_to_ts, _load_alphas

    cfg = json.loads(cfg_path.read_text())
    pack = predict_mu_sigma(cfg, ckpt, split="test")
    # also need val for E5 τ
    pack_val = predict_mu_sigma(cfg, ckpt, split="val")
    cache = pack["cache"]
    paths = {
        "alpha": ckpt.parent / "sigma_recalib_per_dim.json",
        "ence": None,
    }
    alphas, recipe = _load_alphas(paths, pack["mu"].shape[1])
    return cfg, cache, pack, pack_val, alphas, recipe


def run_cast_column(
    *,
    cfg_path: Path,
    ckpt: Path,
    experiments: list[str],
    cast_year: int = 2021,
    n_levels: int = 60,
    max_casts: int | None = None,
    out_json: Path | None = None,
    rcal_form: str = "full",
    rcal_loc_m: float = L_V_M,
) -> dict:
    from base.split_utils import build_split_indices, sample_dates
    from dacov import export_ts_covariance_pca, localize_covariance
    from model.joint_eof import fit_joint_eof, reconstruct_joint_eof, transform_joint_eof
    from scripts.isop_modas_baseline import design_matrix, fit_ridge, per_level_rmse, predict
    from scripts.phase5_physical_space_score import _decode_pcs_to_ts

    cfg, cache, pack_te, pack_va, alphas, recipe = _load_winner(cfg_path, ckpt)
    depth_full = np.asarray(cache["PRES"], dtype=np.float64).reshape(-1)
    lev = subsample_levels(depth_full.size, n_levels)
    depth = depth_full[lev]

    dates = sample_dates(cache["JULD"], dataset_tag=cache.get("dataset_tag", "argo_v2"))
    months = dates.astype("datetime64[M]").astype(int) % 12 + 1
    years = dates.astype("datetime64[Y]").astype(int) + 1970

    dl = cfg["data_loader"]["args"]
    splits = build_split_indices(
        int(cache["inputs"].shape[0]),
        cache.get("JULD"),
        {
            "split_mode": dl.get("split_mode", "chronological"),
            "train_frac": float(dl.get("train_frac", 0.7)),
            "val_frac": float(dl.get("val_frac", 0.15)),
            "test_frac": float(dl.get("test_frac", 0.15)),
            "split_seed": int(dl.get("split_seed", 42)),
            "unassigned": dl.get("unassigned", "exclude"),
        },
        dataset_tag=cache.get("dataset_tag", "argo_v2"),
        v2_src=cfg.get("io", {}).get("v2_src"),
    )
    tr = np.asarray(splits["train"], dtype=int)
    va = np.asarray(splits["val"], dtype=int)

    T_all = np.asarray(cache["profiles"]["temperature"], dtype=np.float64).T  # (N,z)
    S_all = np.asarray(cache["profiles"]["salinity"], dtype=np.float64).T
    T_tr, S_tr = T_all[tr][:, lev], S_all[tr][:, lev]
    clim = monthly_clim(T_tr, months[tr])
    sigma_clim = np.nanstd(T_tr, axis=0)
    sigma_clim = np.maximum(sigma_clim, 1e-3)
    B = vertical_B(depth, sigma_clim, L_v=L_V_M)

    # cast set: calendar year (prereg) ∩ available profiles
    cast_idx = np.where(years == int(cast_year))[0]
    if cast_idx.size == 0:
        # fallback: chronological test profiles
        cast_idx = np.asarray(splits["test"], dtype=int)
        cast_year_note = f"no year={cast_year} in cache; using chrono test n={cast_idx.size}"
    else:
        cast_year_note = f"year={cast_year} n={cast_idx.size}"
    if max_casts is not None:
        cast_idx = cast_idx[: int(max_casts)]

    T_true = T_all[cast_idx][:, lev]
    months_c = months[cast_idx]
    x_b = np.stack([clim[int(m)] for m in months_c], axis=0)

    # --- E2 ISOP ---
    meta = fit_joint_eof(T_all[tr][:, lev], S_all[tr][:, lev], n_comp=min(32, lev.size))
    pcs_tr = transform_joint_eof(T_all[tr][:, lev], S_all[tr][:, lev], meta, meta["pca"])
    # inputs: last-3 = sss,sst,ssh (config input_params order)
    X_tr = design_matrix(cache["inputs"][tr, 8], cache["inputs"][tr, 7], cache["JULD"][tr])
    coef = fit_ridge(X_tr, pcs_tr, lam=1.0)
    X_c = design_matrix(cache["inputs"][cast_idx, 8], cache["inputs"][cast_idx, 7], cache["JULD"][cast_idx])
    T_isop, _S_isop = reconstruct_joint_eof(predict(X_c, coef), meta, meta["pca"])

    # --- E3/E4 NeSPReSO on cast indices (forward all, then slice) ---
    # Re-forward on cast_idx directly for μ/σ
    import torch
    from model.model import PatchConvMLP
    from model.prob_head import softplus_sigma, split_mu_sigma

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arch = dict(cfg["arch"]["args"])
    arch["probabilistic"] = True
    arch.setdefault("n_quantiles", 0)
    model = PatchConvMLP(**arch).to(device)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state.get("model_state_dict", state.get("state_dict", state)), strict=False)
    model.eval()
    d = int(sum(cfg["outputs"].values()))
    with torch.no_grad():
        out = model(torch.tensor(cache["inputs"][cast_idx], dtype=torch.float32, device=device))
        if isinstance(out, (tuple, list)):
            out = out[0]
        mu, raw = split_mu_sigma(out, d)
        sigma = softplus_sigma(raw).cpu().numpy() * alphas[None, :]
        mu = mu.cpu().numpy()
    T_nes, S_nes = _decode_pcs_to_ts(mu, cfg, cache)
    T_nes, S_nes = T_nes[:, lev], S_nes[:, lev]

    pca_T, pca_S = cache["pca_models"]["temperature"], cache["pca_models"]["salinity"]
    n_T = int(cfg["outputs"]["temperature"])
    sigbar_val = []
    diag_mean_val = []
    T_va = T_all[va][:, lev]
    with torch.no_grad():
        out_v = model(torch.tensor(cache["inputs"][va], dtype=torch.float32, device=device))
        if isinstance(out_v, (tuple, list)):
            out_v = out_v[0]
        mu_v, raw_v = split_mu_sigma(out_v, d)
        sig_v = softplus_sigma(raw_v).cpu().numpy() * alphas[None, :]
        mu_v = mu_v.cpu().numpy()
    T_nes_v, _ = _decode_pcs_to_ts(mu_v, cfg, cache)
    T_nes_v = T_nes_v[:, lev]
    for i in range(sig_v.shape[0]):
        Sig_s = export_ts_covariance_pca(
            sig_v[i], pca_T, pca_S, n_T=n_T, floor=FLOOR, level_idx=lev
        )["Sigma_T"]
        diag_mean_val.append(float(np.mean(np.diag(Sig_s))))
        sigbar_val.append(float(np.mean(np.sqrt(np.maximum(np.diag(Sig_s), FLOOR)))))
    rmse2_mean_val = float(np.mean(per_level_rmse(T_nes_v, T_va) ** 2))
    # Val-only global variance inflation so mean diag(Σ) matches mean RMSE²
    # (Σ alone is ~10× underdispersive → OI unstable). Same spirit as α recalib.
    infl = float(rmse2_mean_val / max(float(np.mean(diag_mean_val)), FLOOR))
    infl = float(np.clip(infl, 1.0, 1e4))
    tau = float(np.median(sigbar_val)) * float(np.sqrt(infl))  # σ̄ after inflation

    # R_fixed from cast-set RMSE (Dai / prereg: once per source)
    R_clim = r_fixed_from_rmse(per_level_rmse(x_b, T_true))
    R_isop = r_fixed_from_rmse(per_level_rmse(T_isop, T_true))
    R_nes = r_fixed_from_rmse(per_level_rmse(T_nes, T_true))

    results = {
        "mode": "cast_column",
        "cast_year_note": cast_year_note,
        "n_casts": int(cast_idx.size),
        "n_levels": int(lev.size),
        "L_v_m": L_V_M,
        "alpha_recipe": recipe,
        "e5_tau": tau,
        "r_cal_inflation": infl,
        "r_cal_form": rcal_form,
        "r_cal_loc_m": float(rcal_loc_m) if rcal_form == "full" else None,
        "winner_ckpt": str(ckpt),
        "experiments": {},
    }

    def score_analyses(X_a: np.ndarray, retained: np.ndarray | None = None) -> dict:
        if retained is not None:
            X_a = X_a[retained]
            truth = T_true[retained]
        else:
            truth = T_true
        return band_rmse(X_a, truth, depth)

    for exp in experiments:
        if exp == "E0":
            X_a = x_b.copy()
            retained = None
            R_note = "none"
        elif exp == "E1":
            # y = clim = x_b → OI increment ≈ 0 (E0≡E1 by construction when x_b is clim)
            X_a = np.stack([oi_update(x_b[i], x_b[i], B, R_clim) for i in range(len(cast_idx))])
            retained = None
            R_note = "R_fixed_clim"
        elif exp == "E2":
            X_a = np.stack([oi_update(x_b[i], T_isop[i], B, R_isop) for i in range(len(cast_idx))])
            retained = None
            R_note = "R_fixed_isop"
        elif exp == "E3":
            X_a = np.stack([oi_update(x_b[i], T_nes[i], B, R_nes) for i in range(len(cast_idx))])
            retained = None
            R_note = "R_fixed_nespreso"
        elif exp in ("E4", "E5"):
            X_a = np.empty_like(T_true)
            sigbar = np.empty(len(cast_idx))
            for i in range(len(cast_idx)):
                Sig_s = export_ts_covariance_pca(
                    sigma[i], pca_T, pca_S, n_T=n_T, floor=FLOOR, level_idx=lev
                )["Sigma_T"]
                Sig_s = infl * Sig_s
                # Structured R_cal = full CRPS-head Σ = V diag((ασ)²) Vᵀ, Schur-
                # localized in depth (L_loc) so the rank-n_T low-rank Σ stays PSD
                # and full-rank for column OI. diag(Σ∘ρ)=diag(Σ), so σ̄/τ below are
                # localization-invariant. `diag` reproduces the v1 diagonal fallback.
                if rcal_form == "full":
                    R_cal = localize_covariance(Sig_s, depth, L_loc=rcal_loc_m, floor=FLOOR)
                else:
                    R_cal = np.maximum(np.diag(Sig_s), FLOOR)
                sigbar[i] = float(np.mean(np.sqrt(np.maximum(np.diag(Sig_s), FLOOR))))
                X_a[i] = oi_update(x_b[i], T_nes[i], B, R_cal)
            if exp == "E5":
                keep = sigbar <= tau
                # dropped casts → background (no obs)
                X_a = np.where(keep[:, None], X_a, x_b)
                R_note = f"R_cal_QC_tau={tau:.4f}_keep={float(keep.mean()):.3f}"
                retention = float(keep.mean())
            else:
                R_note = "R_cal"
                retention = None
            retained = None
        else:
            raise ValueError(exp)

        if exp not in ("E4", "E5"):
            retention = None if retained is None else float(np.mean(retained))

        sc = score_analyses(X_a, retained)
        results["experiments"][exp] = {
            "R": R_note,
            "rmse": sc,
            "retention": retention if exp in ("E4", "E5") else (
                None if retained is None else float(np.mean(retained))
            ),
        }

    # claim checks
    e = results["experiments"]
    claims = {}
    if "E2" in e and "E3" in e:
        claims["E3_gt_E2"] = e["E3"]["rmse"]["overall"] < e["E2"]["rmse"]["overall"]
    if "E3" in e and "E4" in e:
        claims["E4_ge_E3"] = e["E4"]["rmse"]["overall"] <= e["E3"]["rmse"]["overall"]
    results["claims"] = claims

    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(results, indent=2) + "\n")
    return results


def write_results_md(results: dict, path: Path) -> None:
    lines = [
        "# Phase 6 — Toy OSSE results (cast-column v1)",
        "",
        f"**Mode:** `{results['mode']}` — {results['cast_year_note']}",
        f"**Casts:** {results['n_casts']} · levels={results['n_levels']} · L_v={results['L_v_m']} m",
        f"**Winner ckpt:** `{results['winner_ckpt']}`",
        f"**E5 τ (val P50 σ̄):** {results['e5_tau']:.4f}",
        f"**R_cal val inflation:** {results.get('r_cal_inflation', 1.0):.3f}× (mean diag(Σ) → mean RMSE²)",
        (
            f"**R_cal form:** full Σ_T = V diag((ασ)²) Vᵀ, Schur-localized "
            f"(Gaussian L_loc={results.get('r_cal_loc_m')} m; diag preserved)"
            if results.get("r_cal_form", "full") == "full"
            else "**R_cal form:** diagonal of Σ_T only (v1 fallback; off-diagonals dropped)"
        ),
        "",
        "> Mode `cast_column`: truth = ARGO at cast columns. Map-level ISAS20 + L_h not wired (no 2021 ISAS year on disk).",
        "> E0≡E1 when background is monthly clim and E1 casts are the same clim.",
        "",
        "## E-table (overall T RMSE at cast columns)",
        "",
        "| E | R | overall T RMSE | retention |",
        "|---|---|----------------|-----------|",
    ]
    for exp in E_TABLE:
        if exp not in results["experiments"]:
            continue
        r = results["experiments"][exp]
        ret = "—" if r["retention"] is None else f"{r['retention']:.3f}"
        lines.append(
            f"| {exp} | {r['R']} | {r['rmse']['overall']:.4f} | {ret} |"
        )
    lines += ["", "## Claims", ""]
    for k, v in results.get("claims", {}).items():
        lines.append(f"- `{k}`: **{'PASS' if v else 'FAIL'}**")
    lines += ["", "## By depth band", ""]
    hdr = "| E | " + " | ".join(OSSE_BAND_LABELS) + " |"
    sep = "|---|" + "|".join(["------"] * len(OSSE_BAND_LABELS)) + "|"
    lines += [hdr, sep]
    for exp in E_TABLE:
        if exp not in results["experiments"]:
            continue
        rm = results["experiments"][exp]["rmse"]
        cells = [f"{rm.get(lab, float('nan')):.4f}" for lab in OSSE_BAND_LABELS]
        lines.append(f"| {exp} | " + " | ".join(cells) + " |")
    lines.append("")
    path.write_text("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--mode", default="cast_column", choices=["cast_column"])
    ap.add_argument("--experiments", default="E0,E1,E2,E3,E4,E5")
    ap.add_argument(
        "--config",
        type=Path,
        default=_ROOT / "saved/runs/phase5_matrix/configs/p5_A_CRPS_v2_s42.json",
    )
    ap.add_argument(
        "--winner-ckpt",
        type=Path,
        default=None,
        help="default: stage2 from twostage manifest s42",
    )
    ap.add_argument("--cast-year", type=int, default=2021)
    ap.add_argument("--n-levels", type=int, default=60)
    ap.add_argument("--max-casts", type=int, default=None)
    ap.add_argument(
        "--rcal",
        choices=["full", "diag"],
        default="full",
        help="E4/E5 R_cal: 'full' = localized Σ_T (planned); 'diag' = v1 fallback",
    )
    ap.add_argument(
        "--rcal-loc-m",
        type=float,
        default=L_V_M,
        help="Gaussian vertical localization length for full R_cal (m; reuses locked L_v)",
    )
    default_out = _ROOT / "saved/runs/phase6_osse/cast_column_s42.json"
    ap.add_argument("--out", type=Path, default=default_out)
    args = ap.parse_args()
    if args.selfcheck:
        selfcheck()
        return 0

    ckpt = args.winner_ckpt
    if ckpt is None:
        man = json.loads(
            (_ROOT / "saved/runs/phase5_matrix/twostage/p5_A_CRPS_v2_s42/manifest.json").read_text()
        )
        ckpt = Path(man["stage2_ckpt"])

    exps = [e.strip() for e in args.experiments.split(",") if e.strip()]
    bad = [e for e in exps if e not in E_TABLE]
    if bad:
        raise SystemExit(f"unknown experiments {bad}")

    print(f"=== OSSE {args.mode} ckpt={ckpt} exps={exps} ===", flush=True)
    results = run_cast_column(
        cfg_path=args.config,
        ckpt=ckpt,
        experiments=exps,
        cast_year=args.cast_year,
        n_levels=args.n_levels,
        max_casts=args.max_casts,
        out_json=args.out,
        rcal_form=args.rcal,
        rcal_loc_m=args.rcal_loc_m,
    )
    # Only the canonical default run writes the committed report; a custom
    # --out (smoke / ablation) writes its md alongside its json so it can't
    # silently clobber reports/osse_results.md.
    if args.out == default_out:
        md = _ROOT.parent / "reports" / "osse_results.md"
    else:
        md = args.out.with_suffix(".md")
    write_results_md(results, md)
    print("wrote", args.out)
    print("wrote", md)
    print("claims", results.get("claims"))
    for exp, r in results["experiments"].items():
        print(f"  {exp}: RMSE={r['rmse']['overall']:.4f} {r['R']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
