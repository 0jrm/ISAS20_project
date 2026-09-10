#!/usr/bin/env python3
"""Measure gated-neighbor (km, dt, dlat, dlon) under nearest-k vs the full candidate pool.

Writes reports/pair_mix_census.json. Train-only stats. No model.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _moments(x):
    x = np.asarray(x, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0}
    q = np.quantile(x, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "skew": float(((x - x.mean()) ** 3).mean() / (x.std() ** 3 + 1e-12)),
        "q05": float(q[0]),
        "q25": float(q[1]),
        "q50": float(q[2]),
        "q75": float(q[3]),
        "q95": float(q[4]),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def main() -> int:
    import pickle

    from base.split_utils import build_split_indices
    from base.util import read_json
    from preproc.pair_table import (
        _neighbors_for_target,
        build_pair_table,
        haversine_km,
    )

    cfg = read_json(str(_ROOT / "config/argo/config_argo_stoch_eof_pair.json"))
    path = Path(cfg["data_loader"]["args"]["cache_path"])
    if not path.is_absolute():
        path = _ROOT / path
    with open(path, "rb") as f:
        cache = pickle.load(f)
    n = int(cache["inputs"].shape[0])
    split = build_split_indices(
        n,
        cache["JULD"],
        cfg["data_loader"]["args"],
        dataset_tag=cache.get("dataset_tag", "argo_v2"),
        v2_src=cfg["io"].get("v2_src"),
    )
    lat = np.asarray(cache["LAT"], dtype=np.float64).ravel()
    lon = np.asarray(cache["LON"], dtype=np.float64).ravel()
    juld = np.asarray(cache["JULD"], dtype=np.float64).ravel()
    train = [int(i) for i in split["train"]]

    n_cand = []
    pool_km, pool_dt, pool_dlat, pool_dlon = [], [], [], []
    for i in train:
        dt = juld[i] - juld
        km = haversine_km(lat[i], lon[i], lat, lon)
        js = np.arange(n, dtype=np.int32)
        ok = (js != i) & (dt >= 3) & (dt <= 45) & (km <= 250)
        cand = js[ok]
        n_cand.append(int(cand.size))
        if cand.size == 0:
            continue
        pool_km.append(km[ok])
        pool_dt.append(dt[ok])
        pool_dlat.append(lat[i] - lat[cand])
        pool_dlon.append(lon[i] - lon[cand])

    n_cand = np.asarray(n_cand, dtype=np.int32)
    pool_km = np.concatenate(pool_km) if pool_km else np.zeros(0)
    pool_dt = np.concatenate(pool_dt) if pool_dt else np.zeros(0)
    pool_dlat = np.concatenate(pool_dlat) if pool_dlat else np.zeros(0)
    pool_dlon = np.concatenate(pool_dlon) if pool_dlon else np.zeros(0)

    pt = build_pair_table(cache, split, k_train=4, k_eval=1)
    tr = pt["train"]
    nearest_km = haversine_km(lat[tr["target_i"]], lon[tr["target_i"]], lat[tr["source_j"]], lon[tr["source_j"]])

    km_edges = np.array([0.0, 50.0, 100.0, 175.0, 250.0])
    dt_edges = np.array([3.0, 10.0, 20.0, 45.0])

    def occupancy(km, dt):
        h, _, _ = np.histogram2d(km, dt, bins=[km_edges, dt_edges])
        return h.astype(int).tolist()

    synth_path = _ROOT / "../data/cache/synth_pcs_stoch_eof_s42.npy"
    pcs = {
        "argo_l2_mean": float(np.linalg.norm(np.asarray(cache["targets"], dtype=np.float64), axis=1).mean()),
    }
    if synth_path.is_file():
        syn = np.load(synth_path)
        pcs["synth_shape"] = list(syn.shape)
        pcs["synth_l2_mean"] = float(np.linalg.norm(syn.astype(np.float64), axis=1).mean())
        pcs["l2_ratio_synth_over_argo"] = pcs["synth_l2_mean"] / max(pcs["argo_l2_mean"], 1e-12)

    empty = int((n_cand == 0).sum())
    rec_k = int(np.clip(np.median(n_cand[n_cand > 0]), 4, 12)) if (n_cand > 0).any() else 4
    payload = {
        "n_train_targets": int(len(train)),
        "n_train_targets_with_neighbor": int((n_cand > 0).sum()),
        "n_train_targets_empty": empty,
        "n_cand": _moments(n_cand),
        "frac_targets_ge_12_cand": float((n_cand >= 12).mean()),
        "frac_targets_ge_8_cand": float((n_cand >= 8).mean()),
        "pool": {
            "km": _moments(pool_km),
            "dt": _moments(pool_dt),
            "dlat": _moments(pool_dlat),
            "dlon": _moments(pool_dlon),
            "km_dt_bins": occupancy(pool_km, pool_dt),
            "km_edges": km_edges.tolist(),
            "dt_edges": dt_edges.tolist(),
        },
        "nearest_k4": {
            "n_pairs": int(tr.shape[0]),
            "km": _moments(nearest_km),
            "dt": _moments(tr["dt_days"]),
            "dlat": _moments(tr["dlat"]),
            "dlon": _moments(tr["dlon"]),
            "km_dt_bins": occupancy(nearest_km, np.asarray(tr["dt_days"])),
        },
        "pcs": pcs,
        "recommend": {
            "k_train_stratified": rec_k,
            "note": (
                "nearest-k piles mass in the close-km short-dt bin. "
                "stratified round-robin over the km×dt bins, then 2× mix (argo, synth) "
                "on those (i,j). val/test keep nearest-1 geography; mix only the source PCs."
            ),
            "train_rows_if_k_and_mix2": int((n_cand > 0).sum() * rec_k * 2),
        },
    }
    out = _ROOT.parent / "reports" / "pair_mix_census.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
