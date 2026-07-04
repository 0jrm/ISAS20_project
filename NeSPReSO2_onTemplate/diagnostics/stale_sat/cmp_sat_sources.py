#!/usr/bin/env python3
"""Compare L4 patch-cache center-pixel sat values vs point-cache sat values per split."""
import pickle
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

ROOT = Path("/unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project/NeSPReSO2_onTemplate")
sys.path.insert(0, str(ROOT))
from base.split_utils import build_split_indices

CACHE_DIR = Path("/unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project/data/cache")
with open(CACHE_DIR / "train_ready_4411c65ee518.pkl", "rb") as f:
    patch = pickle.load(f)
with open(CACHE_DIR / "train_ready_ff2393a1ea21.pkl", "rb") as f:
    point = pickle.load(f)

n = patch["inputs"].shape[0]
dl_cfg = {"split_mode": "chronological", "split_config": None, "train_frac": 0.7,
          "val_frac": 0.15, "test_frac": 0.15, "split_seed": 42, "unassigned": "exclude"}
splits = build_split_indices(n, patch["JULD"], dl_cfg, dataset_tag="argo_l4", v2_src=None)

assert np.allclose(patch["JULD"], point["JULD"]), "JULD mismatch between caches"
center = 6 * 25 + 12
Xp = patch["inputs"]
Xq = point["inputs"]
names = ("sss", "sst", "ssh")

def d2date(j):
    return date.fromordinal(int(j))

print("split date ranges:")
for sp in ("train", "val", "test"):
    idx = np.asarray(splits[sp], dtype=int)
    j = patch["JULD"][idx]
    print(f"  {sp:5s} {d2date(j.min())} .. {d2date(j.max())}  n={len(idx)}")

print("\nper-split center-pixel (L4 cache) vs point-cache sat value:")
for k, nm in enumerate(names):
    pc = Xp[:, 10 + k * 175 + center]
    qc = Xq[:, 6 + k]
    print(f"\n  {nm}:")
    for sp in ("train", "val", "test"):
        idx = np.asarray(splits[sp], dtype=int)
        a, b = pc[idx], qc[idx]
        r = np.corrcoef(a, b)[0, 1]
        print(f"    {sp:5s} corr={r:+.3f}  L4 mean={a.mean():8.3f} std={a.std():6.3f} | point mean={b.mean():8.3f} std={b.std():6.3f} | mean|diff|={np.abs(a-b).mean():.3f}")

# correlation vs time in 60-day bins over whole record
print("\nSST corr(L4 center, point) per 120-day bin:")
pc = Xp[:, 10 + 1 * 175 + center]
qc = Xq[:, 7]
j = patch["JULD"]
bins = ((j - j.min()) // 120).astype(int)
for b in np.unique(bins):
    m = bins == b
    if m.sum() < 20:
        continue
    r = np.corrcoef(pc[m], qc[m])[0, 1]
    print(f"  bin {b:02d} {d2date(j[m].min())} n={m.sum():4d} corr={r:+.3f} L4mean={pc[m].mean():6.2f} ptmean={qc[m].mean():6.2f}")

print("\nSSH corr per 120-day bin:")
pc = Xp[:, 10 + 2 * 175 + center]
qc = Xq[:, 8]
for b in np.unique(bins):
    m = bins == b
    if m.sum() < 20:
        continue
    r = np.corrcoef(pc[m], qc[m])[0, 1]
    print(f"  bin {b:02d} {d2date(j[m].min())} n={m.sum():4d} corr={r:+.3f} L4mean={pc[m].mean():6.3f} ptmean={qc[m].mean():6.3f}")
