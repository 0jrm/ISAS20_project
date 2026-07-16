#!/usr/bin/env python3
"""T2 — stale satellite fraction per chronological split (SST, SSH/ADT, SSS)."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import date
from pathlib import Path

import h5py
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from base.split_utils import build_split_indices

DEFAULT_H5 = (
    "/unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project/"
    "data/NeSPReSO_v2_ARGO_GoM_sat/satellite_NeSPReSO_v2_ARGO_GoM.h5"
)
DEFAULT_CACHE = (
    "/unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project/"
    "data/cache/train_ready_4411c65ee518.pkl"
)
STALE_GATE_FRAC = 0.05


def _stale_mask(patch: np.ndarray) -> np.ndarray:
    """True when all temporal slices are identical (time-constant patch)."""
    a = np.nan_to_num(patch, nan=-999.0)
    ref = a[:, :1, ...]
    return np.all(np.abs(a - ref) < 1e-6, axis=(1, 2, 3))


def audit_splits(h5_path: Path, cache_path: Path) -> dict:
    with h5py.File(h5_path, "r") as f:
        jd = f["stations/julian_date"][:]
        variables = {
            "SST": f["ostia/analysed_sst"][:],
            "SSH_adt": f["ssh/adt"][:],
            "SSS": f["sss/sos"][:],
        }
    stale = {name: _stale_mask(arr) for name, arr in variables.items()}

    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    n = cache["inputs"].shape[0]
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

    days70 = jd - 2440587.5
    ords = (days70 + date(1970, 1, 1).toordinal()).astype(int)

    rows = []
    gate_embargo = False
    for sp in ("train", "val", "test"):
        idx = np.asarray(splits[sp], dtype=int)
        o = ords[idx]
        row = {
            "split": sp,
            "n": int(idx.size),
            "date_min": str(date.fromordinal(int(o.min()))),
            "date_max": str(date.fromordinal(int(o.max()))),
        }
        for var, mask in stale.items():
            frac = float(mask[idx].mean()) if idx.size else 0.0
            row[f"stale_frac_{var}"] = frac
            if sp in ("val", "test") and frac > STALE_GATE_FRAC:
                gate_embargo = True
        rows.append(row)

    return {
        "h5_path": str(h5_path),
        "cache_path": str(cache_path),
        "stale_gate_threshold": STALE_GATE_FRAC,
        "headline_metrics_embargoed": gate_embargo,
        "splits": rows,
    }


def to_md(data: dict) -> str:
    lines = [
        "# Stale satellite audit by split (T2)",
        "",
        f"HDF5: `{data['h5_path']}`",
        f"Cache: `{data['cache_path']}`",
        f"Gate threshold: stale fraction > {data['stale_gate_threshold']:.0%} in val or test ⇒ headline metrics embargoed.",
        "",
        f"**Gate status:** {'EMBARGOED' if data['headline_metrics_embargoed'] else 'OPEN'}",
        "",
        "| split | n | dates | stale SST | stale SSH/ADT | stale SSS |",
        "|-------|---|-------|-----------|---------------|-----------|",
    ]
    for r in data["splits"]:
        lines.append(
            f"| {r['split']} | {r['n']} | {r['date_min']} .. {r['date_max']} | "
            f"{r['stale_frac_SST']:.3f} | {r['stale_frac_SSH_adt']:.3f} | {r['stale_frac_SSS']:.3f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--h5", type=Path, default=DEFAULT_H5)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--out-json", type=Path, default=_ROOT.parent / "reports" / "stale_by_split.json")
    ap.add_argument("--out-md", type=Path, default=_ROOT.parent / "reports" / "stale_by_split.md")
    args = ap.parse_args()
    data = audit_splits(args.h5, args.cache)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(data, indent=2) + "\n")
    args.out_md.write_text(to_md(data))
    print(f"gate={'EMBARGOED' if data['headline_metrics_embargoed'] else 'OPEN'}")
    print(f"wrote {args.out_json} and {args.out_md}")


if __name__ == "__main__":
    main()
