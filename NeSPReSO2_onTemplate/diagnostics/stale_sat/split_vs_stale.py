#!/usr/bin/env python3
"""How much of each chronological split has stale (time-constant) satellite patches?"""
import pickle
import sys
from datetime import date
from pathlib import Path

import h5py
import numpy as np

ROOT = Path("/unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project/NeSPReSO2_onTemplate")
sys.path.insert(0, str(ROOT))
from base.split_utils import build_split_indices

H5 = "/unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project/data/NeSPReSO_v2_ARGO_GoM_sat/satellite_NeSPReSO_v2_ARGO_GoM.h5"
CACHE = "/unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project/data/cache/train_ready_4411c65ee518.pkl"

with h5py.File(H5, "r") as f:
    jd = f["stations"]["julian_date"][:]
    sst = f["ostia"]["analysed_sst"][:]
a = np.nan_to_num(sst, nan=-999.0)
stale = np.all(np.abs(a - a[:, :1]) < 1e-6, axis=(1, 2, 3))

with open(CACHE, "rb") as f:
    cache = pickle.load(f)
n = cache["inputs"].shape[0]
dl_cfg = {"split_mode": "chronological", "split_config": None, "train_frac": 0.7,
          "val_frac": 0.15, "test_frac": 0.15, "split_seed": 42, "unassigned": "exclude"}
splits = build_split_indices(n, cache["JULD"], dl_cfg, dataset_tag="argo_l4", v2_src=None)

days70 = jd - 2440587.5
ords = (days70 + date(1970, 1, 1).toordinal()).astype(int)
for sp in ("train", "val", "test"):
    idx = np.asarray(splits[sp], dtype=int)
    o = ords[idx]
    print(f"{sp:5s} n={len(idx):4d}  {date.fromordinal(int(o.min()))} .. {date.fromordinal(int(o.max()))}  stale_SST_frac={stale[idx].mean():.3f}")
