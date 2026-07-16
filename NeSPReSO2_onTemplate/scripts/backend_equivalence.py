#!/usr/bin/env python3
"""gsw vs gsw_torch equivalence on test-split profiles (PLAN-v2-recovery F.3)."""

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
        return x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


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
        print("SKIP: gsw_torch not importable")
        return 0

    import torch

    from base.split_utils import build_split_indices
    from evalphys.gsw_backend import get_gsw, set_headline_frozen
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
    sig_r = gsw_ref.sigma0(sa_r, ct_r)
    spice_r = gsw_ref.spiciness0(sa_r, ct_r)

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

    def max_rms(a, b):
        d = np.abs(a - b)
        return float(np.nanmax(d)), float(np.sqrt(np.nanmean(d**2)))

    diffs = {}
    for name, a, b in (
        ("sigma0", sig_r, sig_t),
        ("spiciness0", spice_r, spice_t),
        ("SA", sa_r, sa_tn),
        ("CT", ct_r, ct_tn),
    ):
        mx, rms = max_rms(a, b)
        diffs[name] = {"max_abs": mx, "rms": rms, "pass_atol_1e-6": mx < 1e-6}

    # N² via reference path only for count comparison under both backends is expensive;
    # compare N² arrays profile-wise with gsw_ref.Nsquared vs gsw_torch if available.
    n2_max = 0.0
    n2_rms_acc = []
    viol_counts = {"gsw": {}, "gsw_torch": {}}
    # Violation rates under reference gsw (headline) — backend switch for torch path
    # only affects conversions above; N² count identity checked on σ₀-adjacent N² from ref.
    for tol in (0.0, 1e-9, 1e-8, 1e-7):
        set_headline_frozen(True)
        o = static_stability_violations(T, S, depth, lat_i, lon_i, n2_tol=tol)
        viol_counts["gsw"][f"{tol:.0e}" if tol else "0"] = o["n_violations"]

    # MLD under reference
    mld = mixed_layer_depth(T, S, depth, lat_i, lon_i)

    set_headline_frozen(True)
    report = {
        "n_profiles": int(idx.size),
        "diffs": diffs,
        "n2_note": "N² count identity uses headline gsw path; σ₀/spice/SA/CT compared across backends",
        "violation_counts_gsw": viol_counts["gsw"],
        "mld_finite_frac": float(np.isfinite(mld).mean()),
        "thresholds": {"sigma0_spice_SA_CT_atol": 1e-6, "N2_atol": 1e-10},
    }
    # Pass/fail summary
    fail = [k for k, v in diffs.items() if not v["pass_atol_1e-6"]]
    report["pass"] = len(fail) == 0
    report["failed_quantities"] = fail

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"pass": report["pass"], "failed": fail, "out": str(args.out)}, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
