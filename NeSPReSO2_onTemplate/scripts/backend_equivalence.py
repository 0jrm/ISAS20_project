#!/usr/bin/env python3
"""gsw vs gsw_torch equivalence on test-split profiles (PLAN-v2-recovery F.3).

Writes ``reports/backend_equivalence.{json,md}`` with max-abs / RMS per quantity —
upstream-issue evidence for GSW-Torch (JOSS), even when headline stays on reference gsw.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _as_numpy(x):
    if hasattr(x, "detach"):
        return np.asarray(x.detach().cpu().numpy(), dtype=np.float64)
    return np.asarray(x, dtype=np.float64)


def _max_rms(a: np.ndarray, b: np.ndarray) -> dict:
    d = np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))
    finite = np.isfinite(d)
    if not finite.any():
        return {"max_abs": None, "rms": None, "n": 0}
    dd = d[finite]
    return {"max_abs": float(dd.max()), "rms": float(np.sqrt(np.mean(dd**2))), "n": int(finite.sum())}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "-c",
        "--cache",
        default="/unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project/data/cache/train_ready_4411c65ee518.pkl",
    )
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out", type=Path, default=_ROOT.parent / "reports" / "backend_equivalence.json")
    args = ap.parse_args()

    if importlib.util.find_spec("gsw_torch") is None:
        report = {"status": "skipped", "reason": "gsw_torch not importable"}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print("SKIP: gsw_torch not importable")
        return 0

    import torch

    from base.split_utils import build_split_indices
    from evalphys.gsw_backend import get_gsw, package_versions, set_headline_frozen
    from evalphys.metrics import mixed_layer_depth, static_stability_violations, to_teos10

    with open(args.cache, "rb") as f:
        cache = pickle.load(f)
    T_all = np.asarray(cache["profiles"]["temperature"], dtype=np.float64).T
    S_all = np.asarray(cache["profiles"]["salinity"], dtype=np.float64).T
    depth = np.asarray(cache["PRES"], dtype=np.float64)
    lat = np.asarray(cache["LAT"], dtype=np.float64)
    lon = np.asarray(cache["LON"], dtype=np.float64)
    n = T_all.shape[0]
    dl = dict(
        split_mode="chronological",
        split_config=None,
        train_frac=0.7,
        val_frac=0.15,
        test_frac=0.15,
        split_seed=42,
        unassigned="exclude",
    )
    te = np.asarray(
        build_split_indices(n, cache["JULD"], dl, dataset_tag=cache.get("dataset_tag", "argo_v2"))["test"],
        dtype=int,
    )
    rng = np.random.default_rng(42)
    idx = rng.choice(te, size=min(args.n, te.size), replace=False)
    T, S = T_all[idx], S_all[idx]
    lat_i, lon_i = lat[idx], lon[idx]

    gsw_ref = get_gsw("gsw")
    set_headline_frozen(False)
    gsw_t = get_gsw("gsw_torch", allow_torch_for_training=True)

    sa_r, ct_r, p_r = to_teos10(T, S, depth, lat_i, lon_i)
    sig_r = np.asarray(gsw_ref.sigma0(sa_r, ct_r), dtype=np.float64)
    spice_r = np.asarray(gsw_ref.spiciness0(sa_r, ct_r), dtype=np.float64)

    lat_t = torch.as_tensor(lat_i, dtype=torch.float64)[:, None]
    lon_t = torch.as_tensor(lon_i, dtype=torch.float64)[:, None]
    S_t = torch.as_tensor(S, dtype=torch.float64)
    T_t = torch.as_tensor(T, dtype=torch.float64)
    p_t = torch.as_tensor(p_r, dtype=torch.float64)
    sa_t = gsw_t.SA_from_SP(S_t, p_t, lon_t, lat_t)
    ct_t = gsw_t.CT_from_t(sa_t, T_t, p_t)
    sig_t = _as_numpy(gsw_t.sigma0(sa_t, ct_t))
    spice_t = _as_numpy(gsw_t.spiciness0(sa_t, ct_t))
    sa_tn, ct_tn = _as_numpy(sa_t), _as_numpy(ct_t)

    diffs = {
        "sigma0": {**_max_rms(sig_r, sig_t), "atol": 1e-6},
        "spiciness0": {**_max_rms(spice_r, spice_t), "atol": 1e-6},
        "SA": {**_max_rms(sa_r, sa_tn), "atol": 1e-6},
        "CT": {**_max_rms(ct_r, ct_tn), "atol": 1e-6},
    }
    for k, v in diffs.items():
        v["pass"] = v["max_abs"] is not None and v["max_abs"] < v["atol"]

    # N²: reference gsw.Nsquared vs gsw_torch if available; else skip with reason
    n2_diffs = []
    n2_max_all = []
    flip_rows = []
    for i in range(T.shape[0]):
        n2_r, _ = gsw_ref.Nsquared(sa_r[i], ct_r[i], p_r[i], lat_i[i])
        n2_r = np.asarray(n2_r, dtype=np.float64)
        try:
            n2_tt, _ = gsw_t.Nsquared(
                sa_t[i],
                ct_t[i],
                p_t[i],
                torch.as_tensor(lat_i[i], dtype=torch.float64),
            )
            n2_tt = _as_numpy(n2_tt)
        except Exception as exc:  # noqa: BLE001 — optional API
            diffs["N2"] = {"status": "unavailable", "reason": str(exc)}
            break
        d = _max_rms(n2_r, n2_tt)
        n2_diffs.append(d)
        if d["max_abs"] is not None:
            n2_max_all.append(d["max_abs"])
        # Near-threshold flips at N2_TOL=1e-8
        viol_r = n2_r < -1e-8
        viol_t = n2_tt < -1e-8
        flip = viol_r != viol_t
        if np.any(flip):
            for k in np.where(flip)[0]:
                flip_rows.append(
                    {
                        "profile_i": int(i),
                        "level_k": int(k),
                        "N2_gsw": float(n2_r[k]),
                        "N2_gsw_torch": float(n2_tt[k]),
                    }
                )
    else:
        if n2_diffs:
            diffs["N2"] = {
                "max_abs": float(max(n2_max_all)) if n2_max_all else None,
                "rms": float(np.mean([d["rms"] for d in n2_diffs if d["rms"] is not None])),
                "atol": 1e-10,
                "pass": (max(n2_max_all) < 1e-10) if n2_max_all else False,
                "n_profiles": len(n2_diffs),
            }

    # Violation counts under headline gsw (reference path only — identical by construction)
    set_headline_frozen(True)
    viol_counts = {}
    for tol in (0.0, 1e-9, 1e-8, 1e-7):
        o = static_stability_violations(T, S, depth, lat_i, lon_i, n2_tol=tol)
        viol_counts[f"{tol:.0e}" if tol else "0"] = {
            "n_violations": o["n_violations"],
            "violation_rate_level": o["violation_rate_level"],
        }

    mld = mixed_layer_depth(T, S, depth, lat_i, lon_i)
    fail = [k for k, v in diffs.items() if isinstance(v, dict) and v.get("pass") is False]
    report = {
        "status": "ok",
        "n_profiles": int(idx.size),
        "gsw_versions": package_versions(),
        "gsw_backend_headline": "gsw",
        "diffs": diffs,
        "violation_counts_headline_gsw": viol_counts,
        "n2_tol_1e-8_flip_count": len(flip_rows),
        "n2_tol_1e-8_flips_sample": flip_rows[:50],
        "mld_finite_frac": float(np.isfinite(mld).mean()),
        "thresholds": {"sigma0_spice_SA_CT_atol": 1e-6, "N2_atol": 1e-10},
        "pass": len(fail) == 0 and len(flip_rows) == 0,
        "failed_quantities": fail,
        "note": (
            "Headline metrics remain on reference gsw. Failures here are upstream "
            "gsw_torch discrepancy evidence (JOSS), not a STOP for evalphys."
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    md = args.out.with_suffix(".md")
    lines = [
        "# gsw vs gsw_torch backend equivalence (F.3)",
        "",
        f"n_profiles={report['n_profiles']}  |  versions={report['gsw_versions']}",
        f"**pass={report['pass']}**  failed={fail}  N² flips @1e-8={len(flip_rows)}",
        "",
        "| quantity | max_abs | RMS | atol | pass |",
        "|----------|---------|-----|------|------|",
    ]
    for name, v in diffs.items():
        if not isinstance(v, dict) or "max_abs" not in v:
            lines.append(f"| {name} | — | — | — | {v} |")
            continue
        lines.append(
            f"| {name} | {v.get('max_abs')} | {v.get('rms')} | {v.get('atol')} | {v.get('pass')} |"
        )
    lines += ["", report["note"], ""]
    md.write_text("\n".join(lines))
    print(json.dumps({"pass": report["pass"], "failed": fail, "out": str(args.out), "md": str(md)}, indent=2))
    set_headline_frozen(True)
    # Always 0: discrepancy is JOSS evidence, not a broken script
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
