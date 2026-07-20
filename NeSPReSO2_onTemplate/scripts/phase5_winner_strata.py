#!/usr/bin/env python3
"""Phase 5 winner — physical-space CRPS/ENCE × depth band × season (prereg §3).

Default cell = A_CRPS (mechanical §3 pick). Reuses phase5 ensemble decode path.
CPU-friendly; writes JSON + markdown under eval/physical/ and reports/.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.eval_phase4_crps import _band_masks  # noqa: E402
from scripts.phase5_physical_space_score import (  # noqa: E402
    _cell_paths,
    _decode_pcs_to_ts,
    _load_alphas,
)


def _ms(xs: list[float]) -> dict:
    xs = [float(x) for x in xs if x is not None and np.isfinite(x)]
    if not xs:
        return {"mean": None, "std": None, "n": 0}
    if len(xs) == 1:
        return {"mean": xs[0], "std": 0.0, "n": 1}
    return {"mean": float(st.mean(xs)), "std": float(st.stdev(xs)), "n": len(xs)}


def strata_one_seed(cell: str, seed: int, *, n_members: int = 100, profile_chunk: int = 32) -> dict:
    from diagnostics.readiness import ensemble_crps
    from evalphys.calibration import ence, season_from_juld, spread_skill
    from scripts.eval_phase4_crps import predict_mu_sigma

    paths = _cell_paths(cell, seed)
    if paths["det"]:
        raise ValueError("strata script expects a probabilistic cell")
    cfg = json.loads(paths["cfg"].read_text())
    pack = predict_mu_sigma(cfg, paths["ckpt"], split="test")
    cache, idx = pack["cache"], pack["idx"]
    mu, sigma = pack["mu"], pack["sigma"]
    alphas, recipe = _load_alphas(paths, mu.shape[1])
    sigma = sigma * alphas[None, :]
    depth = np.asarray(cache["PRES"], dtype=np.float64).reshape(-1)
    T_true = np.asarray(cache["profiles"]["temperature"], dtype=np.float64).T[idx]
    S_true = np.asarray(cache["profiles"]["salinity"], dtype=np.float64).T[idx]
    seasons = season_from_juld(
        np.asarray(cache["JULD"], dtype=np.float64)[idx],
        dataset_tag=cache.get("dataset_tag", "argo_v2"),
    )
    rng = np.random.default_rng(20_000 + seed)
    n = mu.shape[0]
    members_T = np.empty((n_members, n, depth.size), dtype=np.float32)
    members_S = np.empty((n_members, n, depth.size), dtype=np.float32)
    for i0 in range(0, n, profile_chunk):
        i1 = min(n, i0 + profile_chunk)
        sl = slice(i0, i1)
        eps = rng.normal(size=(n_members, i1 - i0, mu.shape[1]))
        z = mu[sl][None, ...] + sigma[sl][None, ...] * eps
        flat = z.reshape(-1, mu.shape[1])
        Tf, Sf = _decode_pcs_to_ts(flat, cfg, cache)
        members_T[:, sl, :] = Tf.reshape(n_members, i1 - i0, -1)
        members_S[:, sl, :] = Sf.reshape(n_members, i1 - i0, -1)

    T_mu = members_T.mean(0).astype(np.float64)
    S_mu = members_S.mean(0).astype(np.float64)
    T_std = members_T.std(0, ddof=1).astype(np.float64)
    S_std = members_S.std(0, ddof=1).astype(np.float64)
    crps_T = ensemble_crps(members_T, T_true)
    crps_S = ensemble_crps(members_S, S_true)
    bands = _band_masks(depth)

    def pack_slice(prof_mask: np.ndarray, lev_mask: np.ndarray | None = None) -> dict:
        if not np.any(prof_mask):
            return {"n": 0}
        m = prof_mask
        if lev_mask is None:
            cT = float(np.nanmean(crps_T[m]))
            cS = float(np.nanmean(crps_S[m]))
            en = ence(T_mu[m], T_std[m], T_true[m])
            ss = spread_skill(T_mu[m], T_std[m], T_true[m])
        else:
            cT = float(np.nanmean(crps_T[m][:, lev_mask]))
            cS = float(np.nanmean(crps_S[m][:, lev_mask]))
            en = ence(T_mu[m][:, lev_mask], T_std[m][:, lev_mask], T_true[m][:, lev_mask])
            ss = spread_skill(T_mu[m][:, lev_mask], T_std[m][:, lev_mask], T_true[m][:, lev_mask])
        return {
            "n": int(prof_mask.sum()),
            "crps_mean_TS": float(0.5 * (cT + cS)),
            "crps_T": cT,
            "crps_S": cS,
            "ence_T": en.get("ence"),
            "spearman_T": ss.get("spearman_sigma_abs_error"),
            "T_rmse": float(np.sqrt(np.nanmean((T_mu[m] - T_true[m]) ** 2))),
        }

    overall = pack_slice(np.ones(n, dtype=bool))
    by_season = {s: pack_slice(seasons == s) for s in ("DJF", "MAM", "JJA", "SON")}
    by_band = {lab: pack_slice(np.ones(n, dtype=bool), mask) for lab, mask in bands.items()}
    by_band_season = {}
    for lab, mask in bands.items():
        by_band_season[lab] = {
            s: pack_slice(seasons == s, mask) for s in ("DJF", "MAM", "JJA", "SON")
        }
    return {
        "cell": cell,
        "seed": seed,
        "alpha_recipe": recipe,
        "n_members": n_members,
        "n_test": n,
        "season_counts": {s: int((seasons == s).sum()) for s in ("DJF", "MAM", "JJA", "SON")},
        "overall": overall,
        "by_season": by_season,
        "by_depth_band": by_band,
        "by_depth_band_x_season": by_band_season,
    }


def _agg_metric(rows: list[dict], path: tuple[str, ...]) -> dict:
    xs = []
    for r in rows:
        cur = r
        ok = True
        for k in path:
            if not isinstance(cur, dict) or k not in cur:
                ok = False
                break
            cur = cur[k]
        if ok and isinstance(cur, (int, float)) and np.isfinite(cur):
            xs.append(float(cur))
    return _ms(xs)


def aggregate(rows: list[dict]) -> dict:
    out = {
        "cell": rows[0]["cell"],
        "n_seeds": len(rows),
        "overall": {
            "crps_mean_TS": _agg_metric(rows, ("overall", "crps_mean_TS")),
            "ence_T": _agg_metric(rows, ("overall", "ence_T")),
            "spearman_T": _agg_metric(rows, ("overall", "spearman_T")),
            "T_rmse": _agg_metric(rows, ("overall", "T_rmse")),
        },
        "by_season": {},
        "by_depth_band": {},
        "by_depth_band_x_season": {},
        "per_seed": rows,
    }
    for s in ("DJF", "MAM", "JJA", "SON"):
        out["by_season"][s] = {
            "crps_mean_TS": _agg_metric(rows, ("by_season", s, "crps_mean_TS")),
            "ence_T": _agg_metric(rows, ("by_season", s, "ence_T")),
            "n_mean": _agg_metric(rows, ("by_season", s, "n")),
        }
    # bands from first seed
    for lab in rows[0]["by_depth_band"]:
        out["by_depth_band"][lab] = {
            "crps_mean_TS": _agg_metric(rows, ("by_depth_band", lab, "crps_mean_TS")),
            "ence_T": _agg_metric(rows, ("by_depth_band", lab, "ence_T")),
        }
        out["by_depth_band_x_season"][lab] = {}
        for s in ("DJF", "MAM", "JJA", "SON"):
            out["by_depth_band_x_season"][lab][s] = {
                "crps_mean_TS": _agg_metric(
                    rows, ("by_depth_band_x_season", lab, s, "crps_mean_TS")
                ),
                "ence_T": _agg_metric(rows, ("by_depth_band_x_season", lab, s, "ence_T")),
            }
    return out


def _fmt(d: dict) -> str:
    m, s = d.get("mean"), d.get("std")
    if m is None:
        return "—"
    return f"{m:.4f}±{(0.0 if s is None else s):.4f}"


def write_md(agg: dict, path: Path) -> None:
    lines = [
        f"# Phase 5 winner strata — {agg['cell']} (physical space)",
        "",
        f"**Seeds:** {agg['n_seeds']} · prereg §3 depth band × season commitment",
        "",
        "## Overall",
        "",
        f"| CRPS (T+S) | ENCE(T) | Spearman(T) | T RMSE |",
        f"|------------|---------|-------------|--------|",
        f"| {_fmt(agg['overall']['crps_mean_TS'])} | {_fmt(agg['overall']['ence_T'])} | "
        f"{_fmt(agg['overall']['spearman_T'])} | {_fmt(agg['overall']['T_rmse'])} |",
        "",
        "## By season",
        "",
        "| season | CRPS | ENCE(T) | n (mean) |",
        "|--------|------|---------|----------|",
    ]
    for s in ("DJF", "MAM", "JJA", "SON"):
        b = agg["by_season"][s]
        lines.append(
            f"| {s} | {_fmt(b['crps_mean_TS'])} | {_fmt(b['ence_T'])} | {_fmt(b['n_mean'])} |"
        )
    lines += [
        "",
        "## By depth band",
        "",
        "| band | CRPS | ENCE(T) |",
        "|------|------|---------|",
    ]
    for lab, b in agg["by_depth_band"].items():
        lines.append(f"| {lab} | {_fmt(b['crps_mean_TS'])} | {_fmt(b['ence_T'])} |")
    lines += [
        "",
        "## Depth band × season (CRPS)",
        "",
        "| band \\ season | DJF | MAM | JJA | SON |",
        "|---------------|-----|-----|-----|-----|",
    ]
    for lab, seasons in agg["by_depth_band_x_season"].items():
        cells = [_fmt(seasons[s]["crps_mean_TS"]) for s in ("DJF", "MAM", "JJA", "SON")]
        lines.append(f"| {lab} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Depth band × season (ENCE T)",
        "",
        "| band \\ season | DJF | MAM | JJA | SON |",
        "|---------------|-----|-----|-----|-----|",
    ]
    for lab, seasons in agg["by_depth_band_x_season"].items():
        cells = [_fmt(seasons[s]["ence_T"]) for s in ("DJF", "MAM", "JJA", "SON")]
        lines.append(f"| {lab} | " + " | ".join(cells) + " |")
    lines.append("")
    path.write_text("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cell", default="A_CRPS")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--n-members", type=int, default=100)
    ap.add_argument("--profile-chunk", type=int, default=32)
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        assert _ms([1.0, 2.0, 3.0])["mean"] == 2.0
        print("phase5_winner_strata selfcheck OK")
        return 0

    seeds = [int(s) for s in args.seeds.split(",")]
    phys = _ROOT / "saved" / "runs" / "phase5_matrix" / "eval" / "physical"
    phys.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in seeds:
        print(f"=== {args.cell} s{seed} strata ===", flush=True)
        row = strata_one_seed(
            args.cell, seed, n_members=args.n_members, profile_chunk=args.profile_chunk
        )
        (phys / f"{args.cell}_s{seed}_strata.json").write_text(json.dumps(row, indent=2) + "\n")
        rows.append(row)
        print(
            f"  CRPS={row['overall']['crps_mean_TS']:.4f} ENCE={row['overall']['ence_T']}",
            flush=True,
        )
    agg = aggregate(rows)
    (phys / f"{args.cell}_strata_summary.json").write_text(json.dumps(agg, indent=2) + "\n")
    md = _ROOT.parent / "reports" / f"phase5_{args.cell}_physical_strata.md"
    write_md(agg, md)
    print("wrote", phys / f"{args.cell}_strata_summary.json")
    print("wrote", md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
