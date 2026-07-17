#!/usr/bin/env python3
"""Val-only σ recalibration recipes for Phase 4 ENCE recovery.

Fits on val only (global / per-dim / depth-band). Scores test once iff best
val ENCE < 0.20. Does not burn test while iterating recipes.
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

from scripts.eval_phase4_crps import (  # noqa: E402
    ENCE_MAX,
    SPEARMAN_BASELINE,
    _band_masks,
    _cal_bundle,
    predict_mu_sigma,
)

_ALPHA_LO, _ALPHA_HI = 0.05, 20.0


def _rmse_over_rmv(err: np.ndarray, sigma: np.ndarray) -> float:
    """Scalar α = RMSE / RMV on finite entries."""
    e = np.asarray(err, dtype=np.float64).ravel()
    s = np.asarray(sigma, dtype=np.float64).ravel()
    m = np.isfinite(e) & np.isfinite(s) & (s > 0)
    if not np.any(m):
        return 1.0
    rmse = float(np.sqrt(np.mean(e[m] ** 2)))
    rmv = float(np.sqrt(np.mean(s[m] ** 2)))
    if rmv <= 0:
        return 1.0
    return float(np.clip(rmse / rmv, _ALPHA_LO, _ALPHA_HI))


def fit_scales(
    mu: np.ndarray,
    sigma: np.ndarray,
    y: np.ndarray,
    *,
    mode: str,
    k: int,
    z_ctrl: np.ndarray | None,
) -> np.ndarray:
    """Return per-dim scale vector (broadcastable to sigma)."""
    err = mu - y
    d = sigma.shape[1]
    if mode == "none":
        return np.ones(d, dtype=np.float64)
    if mode == "global":
        a = _rmse_over_rmv(err, sigma)
        return np.full(d, a, dtype=np.float64)
    if mode == "per_dim":
        out = np.ones(d, dtype=np.float64)
        for j in range(d):
            out[j] = _rmse_over_rmv(err[:, j], sigma[:, j])
        return out
    if mode == "depth_band":
        # Density ctrl: one α per depth band; spice PCs: per-dim.
        if z_ctrl is None:
            raise ValueError("depth_band mode needs z_ctrl")
        out = np.ones(d, dtype=np.float64)
        bands = _band_masks(z_ctrl)
        for _label, bmask in bands.items():
            cols = np.where(bmask)[0]
            if cols.size == 0:
                continue
            a = _rmse_over_rmv(err[:, cols], sigma[:, cols])
            out[cols] = a
        for j in range(k, d):
            out[j] = _rmse_over_rmv(err[:, j], sigma[:, j])
        return out
    raise ValueError(f"unknown mode {mode!r}")


def _selfcheck() -> None:
    """ponytail: if σ≡|err|, per-dim α≈1 and ENCE stays low after scale."""
    rng = np.random.default_rng(0)
    n, d = 200, 8
    mu = rng.normal(size=(n, d))
    y = mu + rng.normal(scale=0.5, size=(n, d))
    sigma = np.abs(mu - y) + 1e-3
    a = fit_scales(mu, sigma, y, mode="per_dim", k=d, z_ctrl=None)
    assert np.all((a > 0.5) & (a < 2.0)), a
    cal = _cal_bundle(mu, sigma * a, y)
    assert cal["ence"] is not None and cal["ence"] < 0.15, cal
    print("ence_recalib_val selfcheck OK")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-c", "--config")
    ap.add_argument("-r", "--checkpoint")
    ap.add_argument("--out-json", default="../reports/phase4_ence_recalib.json")
    ap.add_argument("--out-md", default="../reports/phase4_ence_recalib.md")
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument(
        "--score-test",
        action="store_true",
        help="Score test with best val recipe even if ENCE still misses (default: only if PASS)",
    )
    args = ap.parse_args()
    if args.selfcheck:
        _selfcheck()
        return 0
    if not args.config or not args.checkpoint:
        ap.error("-c/--config and -r/--checkpoint required unless --selfcheck")

    cfg = json.loads(Path(args.config).read_text())
    ckpt = Path(args.checkpoint)
    val = predict_mu_sigma(cfg, ckpt, split="val")
    mu_v, sg_v, y_v = val["mu"], val["sigma"], val["y"]
    k = int(val["k"])
    z_ctrl = np.asarray(val["meta"]["z_ctrl"], dtype=np.float64)

    recipes = ("none", "global", "per_dim", "depth_band")
    val_rows = {}
    scales = {}
    for mode in recipes:
        a = fit_scales(mu_v, sg_v, y_v, mode=mode, k=k, z_ctrl=z_ctrl)
        scales[mode] = a
        cal = _cal_bundle(mu_v, sg_v * a, y_v)
        val_rows[mode] = {
            **cal,
            "alpha_mean": float(np.mean(a)),
            "alpha_min": float(np.min(a)),
            "alpha_max": float(np.max(a)),
            "ence_pass": cal["ence"] is not None and cal["ence"] < ENCE_MAX,
            "spearman_pass": (
                cal["spearman_sigma_abs_err"] is not None
                and cal["spearman_sigma_abs_err"] > SPEARMAN_BASELINE
            ),
        }

    # Best by val ENCE (finite only); prefer pass then lowest ENCE
    ranked = sorted(
        recipes,
        key=lambda m: (
            not val_rows[m]["ence_pass"],
            val_rows[m]["ence"] if val_rows[m]["ence"] is not None else 9e9,
        ),
    )
    best = ranked[0]
    best_pass = bool(val_rows[best]["ence_pass"] and val_rows[best]["spearman_pass"])

    test_rows = None
    if best_pass or args.score_test:
        test = predict_mu_sigma(cfg, ckpt, split="test")
        a = scales[best]
        cal = _cal_bundle(test["mu"], test["sigma"] * a, test["y"])
        test_rows = {
            "recipe": best,
            **cal,
            "ence_pass": cal["ence"] is not None and cal["ence"] < ENCE_MAX,
            "spearman_pass": (
                cal["spearman_sigma_abs_err"] is not None
                and cal["spearman_sigma_abs_err"] > SPEARMAN_BASELINE
            ),
        }
        test_rows["pass"] = bool(test_rows["ence_pass"] and test_rows["spearman_pass"])

    # Depth-band α summary (for diagnosis)
    band_alphas = {}
    a_db = scales["depth_band"]
    for label, bmask in _band_masks(z_ctrl).items():
        cols = np.where(bmask)[0]
        if cols.size:
            band_alphas[label] = float(a_db[cols[0]])

    out = {
        "checkpoint": str(ckpt),
        "cache": val["cache_path"],
        "fit_split": "val",
        "n_val": int(mu_v.shape[0]),
        "ence_max": ENCE_MAX,
        "val": val_rows,
        "best_recipe": best,
        "best_val_pass": best_pass,
        "depth_band_alphas": band_alphas,
        "per_dim_alphas": scales["per_dim"].tolist(),
        "test": test_rows,
        "note": (
            "Scales fitted on val only. Test scored only if best val ENCE passes "
            "(or --score-test). Spearman invariant to positive σ scale."
        ),
    }

    out_j, out_m = Path(args.out_json), Path(args.out_md)
    out_j.parent.mkdir(parents=True, exist_ok=True)
    out_j.write_text(json.dumps(out, indent=2, default=str) + "\n")

    lines = [
        "# Phase 4 — ENCE recovery via val-only σ recalibration",
        "",
        f"**Checkpoint:** `{ckpt}`  ",
        f"**Cache:** `{val['cache_path']}`  ",
        f"**Fit split:** val n={mu_v.shape[0]}",
        "",
        "## Val recipes (fit + score on val)",
        "",
        "| recipe | CRPS | ENCE | slope | Spearman | α mean [min,max] | ENCE pass |",
        "|--------|------|------|-------|----------|------------------|-----------|",
    ]
    for mode in recipes:
        r = val_rows[mode]
        mark = "**" if mode == best else ""
        lines.append(
            f"| {mark}{mode}{mark} | {r['crps_mean']:.4f} | {r['ence']:.4f} | "
            f"{r['spread_skill_slope']:.3f} | {r['spearman_sigma_abs_err']:.4f} | "
            f"{r['alpha_mean']:.3f} [{r['alpha_min']:.2f},{r['alpha_max']:.2f}] | "
            f"{'yes' if r['ence_pass'] else 'NO'} |"
        )
    lines += [
        "",
        f"**Best (val ENCE):** `{best}` — "
        f"{'PASS' if best_pass else 'MISS'} (ENCE < {ENCE_MAX})",
        "",
        "### Depth-band α (density ctrl)",
        "",
        "| band | α |",
        "|------|---|",
    ]
    for label, a in band_alphas.items():
        lines.append(f"| {label} | {a:.3f} |")
    lines.append("")
    if test_rows is None:
        lines += [
            "## Test",
            "",
            "Not scored — best val recipe still misses ENCE. "
            "Do not burn test; next lever = longer stage-2 / train-time σ.",
            "",
        ]
    else:
        t = test_rows
        lines += [
            "## Test (one score, best val recipe)",
            "",
            f"**Recipe:** `{t['recipe']}`  ",
            f"**Anchors:** {'PASS' if t['pass'] else 'MISS'} "
            f"(ENCE={t['ence']:.4f}; Spearman={t['spearman_sigma_abs_err']:.4f})",
            "",
            f"| CRPS | ENCE | slope | Spearman |",
            f"|------|------|-------|----------|",
            f"| {t['crps_mean']:.4f} | {t['ence']:.4f} | "
            f"{t['spread_skill_slope']:.3f} | {t['spearman_sigma_abs_err']:.4f} |",
            "",
        ]
    lines.append(f"**Note:** {out['note']}")
    out_m.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_j} and {out_m}")
    print(f"BEST={best} VAL_PASS={best_pass} ENCE={val_rows[best]['ence']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
