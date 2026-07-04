#!/usr/bin/env python3
"""Derive a z-scored-input copy of the L4 patch cache (train-split stats)."""
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project/NeSPReSO2_onTemplate")
sys.path.insert(0, str(ROOT))
from base.split_utils import build_split_indices

SRC = "/unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project/data/cache/train_ready_4411c65ee518.pkl"
DST = Path(__file__).parent / "train_ready_e2_std.pkl"

with open(SRC, "rb") as f:
    cache = pickle.load(f)

n = cache["inputs"].shape[0]
dl_cfg = {"split_mode": "chronological", "split_config": None, "train_frac": 0.7,
          "val_frac": 0.15, "test_frac": 0.15, "split_seed": 42, "unassigned": "exclude"}
splits = build_split_indices(n, cache["JULD"], dl_cfg, dataset_tag="argo_l4", v2_src=None)
tr = np.asarray(splits["train"], dtype=int)

X = np.asarray(cache["inputs"], dtype=np.float32)
mu = X[tr].mean(axis=0)
sd = X[tr].std(axis=0)
sd = np.maximum(sd, 1e-6)
cache["inputs"] = ((X - mu) / sd).astype(np.float32)
cache["input_standardization"] = {"mean": mu, "std": sd, "split": "train_chronological_0.7"}

with open(DST, "wb") as f:
    pickle.dump(cache, f)
print(DST, cache["inputs"].shape, "col-std range:", cache["inputs"][tr].std(0).min(), cache["inputs"][tr].std(0).max())
