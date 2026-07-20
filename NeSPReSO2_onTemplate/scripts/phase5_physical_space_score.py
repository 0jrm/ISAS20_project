#!/usr/bin/env python3
"""Phase 5 physical-space rescoring (prereg amendment 2026-07-17).

Draw M latent samples ~ N(μ, diag((α σ)²)), decode each through the cell path to
native T/S, score ensemble CRPS / PIT / ENCE in physical space, plus σ₀/N² rates.
Optional T1-D isotonic projection cost for A/B (mean path).

``--iso`` = mean-path ΔT / σ₀ columns (default for matrix close).
``--iso-ensemble`` = project every member then re-score CRPS (slow; **deferred** to
ranking finalists only — do not use for the full 9×3 table).

Eval-only. Increments the test-eval counter by one consultation per cell.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

RUN = _ROOT / "saved" / "runs" / "phase5_matrix"
EVAL = RUN / "eval"
PHYS = EVAL / "physical"
SKILL_FLOOR = 0.5903
ENCE_MAX = 0.20

# (rep, head) → latent ENCE survivors under v2 (admission filter)
SURVIVOR_PROB = ("A_CRPS", "A_NLL", "B_CRPS", "B_NLL", "C_NLL")
SURVIVOR_DET = ("A_det", "B_det")
ALL_CELLS = (
    "A_CRPS",
    "A_NLL",
    "A_det",
    "B_CRPS",
    "B_NLL",
    "B_det",
    "C_CRPS",
    "C_NLL",
    "C_det",
)


def _cell_paths(cell: str, seed: int) -> dict:
    tag = f"p5_{cell}_v2_s{seed}"
    cfg = RUN / "configs" / f"{tag}.json"
    if cell.endswith("_det"):
        ckpt = (
            _ROOT
            / "saved"
            / "phase5_matrix"
            / f"{cell}_v2"
            / "models"
            / f"NeSPReSO2_ARGO_GoM_p5_{cell}_v2"
            / tag
            / "model_best.pth"
        )
        return {"cfg": cfg, "ckpt": ckpt, "ence": None, "alpha": None, "det": True}
    man = json.loads((RUN / "twostage" / tag / "manifest.json").read_text())
    ckpt = Path(man["stage2_ckpt"])
    ence = EVAL / f"{tag}_ence.json"
    alpha = ckpt.parent / "sigma_recalib_per_dim.json"
    return {"cfg": cfg if cfg.is_file() else Path(man["stage2_config"]), "ckpt": ckpt, "ence": ence, "alpha": alpha, "det": False}


def _load_alphas(paths: dict, d: int) -> tuple[np.ndarray, str]:
    if paths["alpha"] is not None and paths["alpha"].is_file():
        a = json.loads(paths["alpha"].read_text())
        alphas = np.asarray(a["alphas"], dtype=np.float64)
        if alphas.size == d:
            return alphas, str(a.get("method", "file"))
    if paths["ence"] is not None and paths["ence"].is_file():
        e = json.loads(paths["ence"].read_text())
        recipe = e["best_recipe"]
        if recipe == "none":
            return np.ones(d, dtype=np.float64), recipe
        if recipe == "global":
            am = float(e["val"]["global"]["alpha_mean"])
            return np.full(d, am, dtype=np.float64), recipe
        if recipe == "per_dim":
            return np.asarray(e["per_dim_alphas"], dtype=np.float64), recipe
    return np.ones(d, dtype=np.float64), "ones_fallback"


def _decode_pcs_to_ts(pcs: np.ndarray, cfg: dict, cache: dict) -> tuple[np.ndarray, np.ndarray]:
    """pcs (N, d) → T,S (N, n_z)."""
    outs = cfg["outputs"]
    pca = cache["pca_models"]
    if "joint" in outs:
        from model.joint_eof import reconstruct_joint_eof

        meta = cache["joint_eof_meta"]
        return reconstruct_joint_eof(pcs, meta, pca["joint"])
    nt = int(outs["temperature"])
    ns = int(outs["salinity"])
    T = pca["temperature"].inverse_transform(pcs[:, :nt])
    S = pca["salinity"].inverse_transform(pcs[:, nt : nt + ns])
    return T.astype(np.float64), S.astype(np.float64)


def _decode_dens_z_to_ts(z: np.ndarray, cache: dict, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Standardized dens/spice vector (N, K+n_spice) → T,S after isotonic+invert."""
    from evalphys.gsw_backend import get_gsw
    from evalphys.inversion import ts_from_sigma0_spice
    from model.density_spice import project_monotone_sigma0_ctrl, upsample_pchip
    from scripts.eval_density_spice import _load_depth

    meta = cache["density_spice_meta"]
    k = int(meta["K"])
    mu_s = np.asarray(meta["sigma0_ctrl_mean"], dtype=np.float64)
    sd_s = np.maximum(np.asarray(meta["sigma0_ctrl_std"], dtype=np.float64), 1e-6)
    sig_z, z_tau = z[:, :k], z[:, k:]
    sig_ctrl = sig_z * sd_s + mu_s
    z_ctrl = np.asarray(meta["z_ctrl"], dtype=np.float64)
    sig_ctrl = project_monotone_sigma0_ctrl(sig_ctrl, z_ctrl)
    depth = _load_depth(cache)
    sig_hat = upsample_pchip(sig_ctrl, z_ctrl, depth)
    pca = cache["pca_models"]["spice"]
    tau_z = pca.inverse_transform(z_tau)
    tm = np.asarray(meta["spice_mean"], dtype=np.float64)
    ts = np.asarray(meta["spice_std"], dtype=np.float64)
    tau_hat = tau_z * ts + tm
    lat = np.asarray(cache["LAT"], dtype=np.float64)[idx]
    lon = np.asarray(cache["LON"], dtype=np.float64)[idx]
    n_prof = sig_hat.shape[0]
    gsw = get_gsw()
    p = gsw.p_from_z(-np.broadcast_to(depth, (n_prof, depth.size)), lat[:, None])
    T, S, _ok = ts_from_sigma0_spice(sig_hat, tau_hat, p, lon[:, None], lat[:, None])
    return T, S


def _isotonic_reinvert_ts(T, S, depth, lat, lon, dens_meta: dict):
    from scripts.eval_argo16_isotonic_gate import _isotonic_reinvert

    return _isotonic_reinvert(T, S, depth, lat, lon, dens_meta)


def _ensemble_pit_sup(members: np.ndarray, y: np.ndarray, n_bins: int = 20) -> float:
    """Empirical PIT from ensemble members (M, ...); return sup bin deviation."""
    # PIT ≈ (# members < y + 0.5 * ties) / M
    m = np.asarray(members, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    less = np.sum(m < yy[None, ...], axis=0)
    ties = np.sum(m == yy[None, ...], axis=0)
    pit = (less + 0.5 * ties) / m.shape[0]
    pit = pit[np.isfinite(pit)].ravel()
    if pit.size == 0:
        return float("nan")
    counts, _ = np.histogram(pit, bins=n_bins, range=(0.0, 1.0))
    freq = counts / counts.sum()
    return float(np.max(np.abs(freq - 1.0 / n_bins)))


def _band_crps(crps_tz: np.ndarray, depth: np.ndarray) -> dict[str, float]:
    from scripts.eval_phase4_crps import _band_masks

    out = {}
    for label, mask in _band_masks(depth).items():
        if not np.any(mask):
            continue
        out[label] = float(np.nanmean(crps_tz[:, mask]))
    return out


def score_cell_seed(
    cell: str,
    seed: int,
    *,
    n_members: int = 100,
    do_iso: bool = False,
    iso_ensemble: bool = False,
    profile_chunk: int = 40,
) -> dict:
    from diagnostics.readiness import ensemble_crps
    from evalphys.calibration import ence, spread_skill
    from evalphys.gsw_backend import set_headline_frozen
    from evalphys.metrics import summarize_physical
    from scripts.eval_phase4_crps import predict_mu_sigma

    set_headline_frozen(True)
    paths = _cell_paths(cell, seed)
    cfg = json.loads(paths["cfg"].read_text())
    rng = np.random.default_rng(10_000 + seed)
    prob_block = None
    members_T = members_S = None

    if paths["det"]:
        from preproc.export_v2_cache import build_argo_cache
        import pickle
        import torch
        from model.model import PatchConvMLP
        from scripts.eval_phase4_crps import _split_indices

        cache_path = build_argo_cache(cfg)
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        idx = np.asarray(_split_indices(cfg, cache)["test"], dtype=int)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        arch = dict(cfg["arch"]["args"])
        arch["probabilistic"] = False
        model = PatchConvMLP(**arch).to(device)
        ckpt = torch.load(paths["ckpt"], map_location=device, weights_only=False)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)), strict=False)
        model.eval()
        with torch.no_grad():
            mu = model(torch.tensor(cache["inputs"][idx], dtype=torch.float32, device=device)).cpu().numpy()
        if cfg.get("loss_config", {}).get("mode") == "density_spice":
            from scripts.eval_density_spice import decode_density_spice_to_ts

            T_mu, S_mu, _info = decode_density_spice_to_ts(mu, cache, indices=idx)
        else:
            T_mu, S_mu = _decode_pcs_to_ts(mu, cfg, cache)
        space = "det"
    else:
        pack = predict_mu_sigma(cfg, paths["ckpt"], split="test")
        cache, idx = pack["cache"], pack["idx"]
        mu, sigma = pack["mu"], pack["sigma"]
        alphas, recipe = _load_alphas(paths, mu.shape[1])
        if alphas.size != mu.shape[1]:
            raise ValueError(f"{cell} s{seed}: alpha dim {alphas.size} != {mu.shape[1]}")
        sigma = sigma * alphas[None, :]
        space = pack["space"]
        n = mu.shape[0]
        depth0 = np.asarray(cache["PRES"], dtype=np.float64).reshape(-1)
        T_true0 = np.asarray(cache["profiles"]["temperature"], dtype=np.float64).T[idx]
        S_true0 = np.asarray(cache["profiles"]["salinity"], dtype=np.float64).T[idx]
        members_T = np.empty((n_members, n, depth0.size), dtype=np.float32)
        members_S = np.empty((n_members, n, depth0.size), dtype=np.float32)
        for i0 in range(0, n, profile_chunk):
            i1 = min(n, i0 + profile_chunk)
            sl = slice(i0, i1)
            eps = rng.normal(size=(n_members, i1 - i0, mu.shape[1]))
            z = mu[sl][None, ...] + sigma[sl][None, ...] * eps
            flat = z.reshape(-1, mu.shape[1])
            # reshape order is (m0 all profiles), (m1 all profiles), …
            idx_rep = np.tile(idx[sl], n_members)
            if space == "pcs":
                Tf, Sf = _decode_pcs_to_ts(flat, cfg, cache)
            else:
                Tf, Sf = _decode_dens_z_to_ts(flat, cache, idx_rep)
            members_T[:, sl, :] = Tf.reshape(n_members, i1 - i0, -1)
            members_S[:, sl, :] = Sf.reshape(n_members, i1 - i0, -1)
        T_mu = members_T.mean(axis=0).astype(np.float64)
        S_mu = members_S.mean(axis=0).astype(np.float64)
        T_std = members_T.std(axis=0, ddof=1).astype(np.float64)
        S_std = members_S.std(axis=0, ddof=1).astype(np.float64)

        crps_T = ensemble_crps(members_T, T_true0)
        crps_S = ensemble_crps(members_S, S_true0)
        en_T = ence(T_mu, T_std, T_true0)
        en_S = ence(S_mu, S_std, S_true0)
        ss_T = spread_skill(T_mu, T_std, T_true0)
        prob_block = {
            "crps_mean_TS": float(0.5 * (np.nanmean(crps_T) + np.nanmean(crps_S))),
            "crps_T": float(np.nanmean(crps_T)),
            "crps_S": float(np.nanmean(crps_S)),
            "ence_T": en_T.get("ence"),
            "ence_S": en_S.get("ence"),
            "ence_mean_TS": float(
                np.nanmean([en_T.get("ence") or np.nan, en_S.get("ence") or np.nan])
            ),
            "spearman_T": ss_T.get("spearman_sigma_abs_error"),
            "pit_sup_T": _ensemble_pit_sup(members_T, T_true0),
            "crps_by_band": {"T": _band_crps(crps_T, depth0), "S": _band_crps(crps_S, depth0)},
            "alpha_recipe": recipe,
            "n_members": n_members,
        }

    depth = np.asarray(cache["PRES"], dtype=np.float64).reshape(-1)
    T_true = np.asarray(cache["profiles"]["temperature"], dtype=np.float64).T[idx]
    S_true = np.asarray(cache["profiles"]["salinity"], dtype=np.float64).T[idx]
    lat = np.asarray(cache["LAT"], dtype=np.float64)[idx]
    lon = np.asarray(cache["LON"], dtype=np.float64)[idx]
    t_rmse = float(np.sqrt(np.nanmean((T_mu - T_true) ** 2)))
    s_rmse = float(np.sqrt(np.nanmean((S_mu - S_true) ** 2)))
    phys = summarize_physical(T_mu, S_mu, T_true, S_true, depth, lat, lon)
    s0 = float(phys["sigma0_monotonicity_pred"]["violation_rate_profile"])
    n2 = float(phys["static_stability_pred"]["1e-08"]["violation_rate_profile"])

    out = {
        "cell": cell,
        "seed": seed,
        "checkpoint": str(paths["ckpt"]),
        "space": space,
        "test_consultation": "physical_space_v1",
        "overall_T_rmse": t_rmse,
        "overall_S_rmse": s_rmse,
        "sigma0_profile_rate": s0,
        "n2_profile_rate_1e-8": n2,
        "clears_skill_floor": t_rmse <= SKILL_FLOOR,
        "prob": prob_block,
    }
    if prob_block is not None:
        out["ence_pass_physical"] = (prob_block["ence_mean_TS"] is not None) and (
            prob_block["ence_mean_TS"] < ENCE_MAX
        )

    if do_iso and cell[0] in ("A", "B"):
        from preproc.export_v2_cache import build_argo_cache
        import pickle

        c_cfg = json.loads((RUN / "configs" / f"p5_C_NLL_v2_s{seed}.json").read_text())
        dens_cache = pickle.load(open(build_argo_cache(c_cfg), "rb"))
        dens_meta = dens_cache["density_spice_meta"]
        T2, S2, info = _isotonic_reinvert_ts(T_mu, S_mu, depth, lat, lon, dens_meta)
        t_rmse_iso = float(np.sqrt(np.nanmean((T2 - T_true) ** 2)))
        phys_iso = summarize_physical(T2, S2, T_true, S_true, depth, lat, lon)
        out["isotonic"] = {
            **info,
            "overall_T_rmse": t_rmse_iso,
            "delta_T_rmse": t_rmse_iso - t_rmse,
            "sigma0_profile_rate": float(
                phys_iso["sigma0_monotonicity_pred"]["violation_rate_profile"]
            ),
            "n2_profile_rate_1e-8": float(
                phys_iso["static_stability_pred"]["1e-08"]["violation_rate_profile"]
            ),
        }
        if members_T is not None and iso_ensemble:
            members_Ti = np.empty_like(members_T)
            members_Si = np.empty_like(members_S)
            for m in range(n_members):
                Ti, Si, _ = _isotonic_reinvert_ts(
                    members_T[m].astype(np.float64),
                    members_S[m].astype(np.float64),
                    depth,
                    lat,
                    lon,
                    dens_meta,
                )
                members_Ti[m] = Ti
                members_Si[m] = Si
            crps_Ti = ensemble_crps(members_Ti, T_true)
            crps_Si = ensemble_crps(members_Si, S_true)
            Tmi, Smi = members_Ti.mean(0), members_Si.mean(0)
            Tsi = members_Ti.std(0, ddof=1)
            Ssi = members_Si.std(0, ddof=1)
            out["isotonic"]["prob"] = {
                "crps_mean_TS": float(0.5 * (np.nanmean(crps_Ti) + np.nanmean(crps_Si))),
                "ence_mean_TS": float(
                    np.nanmean(
                        [
                            ence(Tmi, Tsi, T_true).get("ence") or np.nan,
                            ence(Smi, Ssi, S_true).get("ence") or np.nan,
                        ]
                    )
                ),
                "spearman_T": spread_skill(Tmi, Tsi, T_true).get("spearman_sigma_abs_error"),
            }

    return out


def _aggregate(rows: list[dict]) -> dict:
    import statistics as st

    def ms(xs):
        xs = [float(x) for x in xs if x is not None and np.isfinite(x)]
        if not xs:
            return None, None
        if len(xs) == 1:
            return xs[0], 0.0
        return float(st.mean(xs)), float(st.stdev(xs))

    cell = rows[0]["cell"]
    t_m, t_s = ms([r["overall_T_rmse"] for r in rows])
    s0_m, s0_s = ms([r["sigma0_profile_rate"] for r in rows])
    n2_m, n2_s = ms([r["n2_profile_rate_1e-8"] for r in rows])
    out = {
        "cell": cell,
        "n_seeds": len(rows),
        "overall_T_rmse": {"mean": t_m, "std": t_s},
        "sigma0_profile_rate": {"mean": s0_m, "std": s0_s},
        "n2_profile_rate_1e-8": {"mean": n2_m, "std": n2_s},
        "clears_skill_floor_on_mean": t_m is not None and t_m <= SKILL_FLOOR,
        "per_seed": rows,
    }
    if rows[0].get("prob"):
        c_m, c_s = ms([r["prob"]["crps_mean_TS"] for r in rows])
        e_m, e_s = ms([r["prob"]["ence_mean_TS"] for r in rows])
        sp_m, sp_s = ms([r["prob"]["spearman_T"] for r in rows])
        out["physical"] = {
            "crps_mean_TS": {"mean": c_m, "std": c_s},
            "ence_mean_TS": {"mean": e_m, "std": e_s},
            "spearman_T": {"mean": sp_m, "std": sp_s},
            "ence_pass_on_mean": e_m is not None and e_m < ENCE_MAX,
        }
    iso_rows = [r for r in rows if r.get("isotonic")]
    if iso_rows:
        it_m, it_s = ms([r["isotonic"]["overall_T_rmse"] for r in iso_rows])
        id_m, id_s = ms([r["isotonic"]["delta_T_rmse"] for r in iso_rows])
        is0_m, is0_s = ms([r["isotonic"]["sigma0_profile_rate"] for r in iso_rows])
        out["isotonic"] = {
            "overall_T_rmse": {"mean": it_m, "std": it_s},
            "delta_T_rmse": {"mean": id_m, "std": id_s},
            "sigma0_profile_rate": {"mean": is0_m, "std": is0_s},
            "n_seeds_mean_path": len(iso_rows),
        }
        # --iso-ensemble writes isotonic.prob; mean-path --iso does not.
        # Only aggregate when every seed has it (mixed resume → omit, don't KeyError).
        iso_prob_rows = [r for r in iso_rows if r["isotonic"].get("prob")]
        if iso_prob_rows and len(iso_prob_rows) == len(iso_rows):
            ic_m, ic_s = ms([r["isotonic"]["prob"]["crps_mean_TS"] for r in iso_prob_rows])
            ie_m, ie_s = ms([r["isotonic"]["prob"]["ence_mean_TS"] for r in iso_prob_rows])
            out["isotonic"]["physical"] = {
                "crps_mean_TS": {"mean": ic_m, "std": ic_s},
                "ence_mean_TS": {"mean": ie_m, "std": ie_s},
                "ence_pass_on_mean": ie_m is not None and ie_m < ENCE_MAX,
            }
        elif iso_prob_rows:
            out["isotonic"]["physical_note"] = (
                f"skipped iso-ensemble aggregate: {len(iso_prob_rows)}/{len(iso_rows)} "
                "seeds have isotonic.prob (mean-path / --iso-ensemble mix)"
            )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells", default="survivors", help="survivors|all|comma list")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--n-members", type=int, default=100)
    ap.add_argument("--iso", action="store_true", help="A/B mean-path isotonic projection cost")
    ap.add_argument(
        "--iso-ensemble",
        action="store_true",
        help="Also project every ensemble member (slow; physical CRPS after iso)",
    )
    ap.add_argument("--profile-chunk", type=int, default=32)
    ap.add_argument("--resume", action="store_true", help="Skip seeds with existing JSON")
    ap.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Rebuild *_summary.json from existing seed JSONs (no scoring)",
    )
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    if args.selfcheck:
        from diagnostics.readiness import ensemble_crps

        rng = np.random.default_rng(0)
        m = rng.normal(size=(20, 5, 7))
        y = rng.normal(size=(5, 7))
        c = ensemble_crps(m, y)
        assert c.shape == y.shape
        # mean-path-only isotonic must not KeyError when mixed with iso-ensemble
        rows_mix = [
            {"cell": "B_NLL", "overall_T_rmse": 1.0, "sigma0_profile_rate": 0.1,
             "n2_profile_rate_1e-8": 0.1, "isotonic": {"overall_T_rmse": 1.0, "delta_T_rmse": 0.0,
             "sigma0_profile_rate": 0.0, "prob": {"crps_mean_TS": 0.5, "ence_mean_TS": 0.1}}},
            {"cell": "B_NLL", "overall_T_rmse": 1.1, "sigma0_profile_rate": 0.1,
             "n2_profile_rate_1e-8": 0.1, "isotonic": {"overall_T_rmse": 1.1, "delta_T_rmse": 0.0,
             "sigma0_profile_rate": 0.0}},
        ]
        agg_mix = _aggregate(rows_mix)
        assert "physical" not in agg_mix.get("isotonic", {})
        assert "physical_note" in agg_mix["isotonic"]
        print("phase5_physical_space_score selfcheck OK")
        return 0

    if args.cells == "survivors":
        cells = list(SURVIVOR_PROB) + list(SURVIVOR_DET)
    elif args.cells == "all":
        cells = list(ALL_CELLS)
    else:
        cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    seeds = [int(s) for s in args.seeds.split(",")]

    PHYS.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for cell in cells:
        rows = []
        do_iso = bool((args.iso or args.iso_ensemble) and cell[0] in ("A", "B"))
        for seed in seeds:
            out_p = PHYS / f"{cell}_s{seed}.json"
            if args.aggregate_only or (args.resume and out_p.is_file()):
                if not out_p.is_file():
                    raise FileNotFoundError(f"--aggregate-only/--resume missing {out_p}")
                print(f"=== {cell} s{seed} RESUME skip ===", flush=True)
                rows.append(json.loads(out_p.read_text()))
                continue
            print(f"=== {cell} s{seed} iso={do_iso} iso_ens={args.iso_ensemble} ===", flush=True)
            row = score_cell_seed(
                cell,
                seed,
                n_members=args.n_members,
                do_iso=do_iso,
                iso_ensemble=bool(args.iso_ensemble),
                profile_chunk=args.profile_chunk,
            )
            (PHYS / f"{cell}_s{seed}.json").write_text(json.dumps(row, indent=2) + "\n")
            rows.append(row)
            crps = None if not row.get("prob") else row["prob"].get("crps_mean_TS")
            crps_s = "NA" if crps is None else f"{crps:.4f}"
            print(
                f"  T={row['overall_T_rmse']:.4f} s0={row['sigma0_profile_rate']:.4f} crps={crps_s}",
                flush=True,
            )
        agg = _aggregate(rows)
        summaries[cell] = agg
        (PHYS / f"{cell}_summary.json").write_text(json.dumps(agg, indent=2) + "\n")

    # Partial --cells must not wipe prior cells in the matrix rollup.
    summary_path = PHYS / "matrix_physical_summary.json"
    if summary_path.is_file():
        prev = json.loads(summary_path.read_text())
        if isinstance(prev, dict):
            prev.update(summaries)
            summaries = prev
    summary_path.write_text(json.dumps(summaries, indent=2) + "\n")
    print("wrote", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
