#!/usr/bin/env python3
"""Phase 3.4 acceptance: (T,S)→(σ₀,τ)→(T,S) round-trip on test profiles."""

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

from base.split_utils import build_split_indices
from evalphys.gsw_backend import get_gsw
from evalphys.inversion import sigma0_spice_from_ts, ts_from_sigma0_spice


def run_roundtrip(cache_path: Path, *, n_profiles: int = 500, seed: int = 0) -> dict:
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    T = np.asarray(cache["true_profiles"]["temperature"], dtype=np.float64)
    S = np.asarray(cache["true_profiles"]["salinity"], dtype=np.float64)
    lat = np.asarray(cache["LAT"], dtype=np.float64)
    lon = np.asarray(cache["LON"], dtype=np.float64)
    depth = np.asarray(cache["PRES"], dtype=np.float64)
    if T.shape[0] != lat.shape[0]:
        T, S = T.T, S.T
    n = lat.shape[0]
    dl_cfg = {
        "split_mode": "chronological",
        "train_frac": 0.7,
        "val_frac": 0.15,
        "test_frac": 0.15,
        "split_seed": 42,
        "unassigned": "exclude",
    }
    splits = build_split_indices(
        n, cache["JULD"], dl_cfg, dataset_tag=cache.get("dataset_tag", "argo_v2")
    )
    te = np.asarray(splits["test"], dtype=int)
    rng = np.random.default_rng(seed)
    take = min(n_profiles, te.size)
    idx = rng.choice(te, size=take, replace=False)

    T_i, S_i = T[idx], S[idx]
    lat_i, lon_i = lat[idx], lon[idx]
    gsw = get_gsw()
    p = gsw.p_from_z(-np.broadcast_to(depth, (take, depth.size)), lat_i[:, None])
    # warm-start Newton from truth TEOS (round-trip fidelity, not cold-start)
    sa0 = gsw.SA_from_SP(S_i, p, lon_i[:, None], lat_i[:, None])
    ct0 = gsw.CT_from_t(sa0, T_i, p)
    sig, tau = sigma0_spice_from_ts(T_i, S_i, p, lon_i[:, None], lat_i[:, None])
    T_hat, S_hat, ok = ts_from_sigma0_spice(
        sig, tau, p, lon_i[:, None], lat_i[:, None], sa0=sa0, ct0=ct0
    )
    dT = np.abs(T_hat - T_i)
    dS = np.abs(S_hat - S_i)
    fail = float(1.0 - ok.mean())
    out = {
        "n_profiles": int(take),
        "gsw_backend": "gsw",
        "max_abs_dT": float(np.nanmax(dT)),
        "max_abs_dS": float(np.nanmax(dS)),
        "newton_fail_rate": fail,
        "pass": bool(
            np.nanmax(dT) < 0.01 and np.nanmax(dS) < 0.01 and fail < 0.001
        ),
        "thresholds": {"max_abs_dT": 0.01, "max_abs_dS": 0.01, "newton_fail_rate": 0.001},
    }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--n-profiles", type=int, default=500)
    ap.add_argument("--out", default="../reports/phase3_roundtrip.json")
    args = ap.parse_args(argv)
    out = run_roundtrip(Path(args.cache), n_profiles=args.n_profiles)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"wrote {path}")
    return 0 if out["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
