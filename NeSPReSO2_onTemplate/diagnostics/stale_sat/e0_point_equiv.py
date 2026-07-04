#!/usr/bin/env python3
"""E0: can a plain MLP on point-equivalent features from the L4 patch cache
match the point model? Feature sets:
  A) 6 harmonics + 3 patch-center sat values (= point-model equivalent)
  B) 10 scalars only (harmonics + basin + bathy; no local sat)
  C) 13 = 10 scalars + 3 centers
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path("/unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project/NeSPReSO2_onTemplate")
sys.path.insert(0, str(ROOT))
from model.metric import per_variable_rmse
from base.split_utils import build_split_indices

CACHE = "/unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project/data/cache/train_ready_4411c65ee518.pkl"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

with open(CACHE, "rb") as f:
    cache = pickle.load(f)

n = cache["inputs"].shape[0]
dl_cfg = {"split_mode": "chronological", "split_config": None, "train_frac": 0.7,
          "val_frac": 0.15, "test_frac": 0.15, "split_seed": 42, "unassigned": "exclude"}
splits = build_split_indices(n, cache["JULD"], dl_cfg, dataset_tag="argo_l4", v2_src=None)
tr, va, te = (np.asarray(splits[s], dtype=int) for s in ("train", "val", "test"))

X = cache["inputs"]
center = 6 * 25 + 12
centers = np.stack([X[:, 10 + j * 175 + center] for j in range(3)], axis=1)  # sss, sst, ssh
feats = {
    "A: 6 harmonics + 3 centers (point-equiv)": np.hstack([X[:, :6], centers]),
    "B: 10 scalars (no local sat)": X[:, :10],
    "C: 10 scalars + 3 centers": np.hstack([X[:, :10], centers]),
}
Y = cache["targets"].astype(np.float32)
W = torch.tensor(np.asarray(cache["weights"], dtype=np.float32), device=DEV)


def run(name, F):
    mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-6
    F = (F - mu) / sd
    Xtr = torch.tensor(F[tr], dtype=torch.float32, device=DEV)
    Ytr = torch.tensor(Y[tr], device=DEV)
    Xva = torch.tensor(F[va], dtype=torch.float32, device=DEV)
    Yva = torch.tensor(Y[va], device=DEV)

    model = nn.Sequential(
        nn.Linear(F.shape[1], 1024), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(1024, 32),
    ).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    def wmse(p, t):
        return (W * (p - t) ** 2).mean()

    best_val, best_state, bad = np.inf, None, 0
    nb = int(np.ceil(len(tr) / 512))
    for ep in range(4000):
        model.train()
        perm = torch.randperm(len(tr), device=DEV)
        for b in range(nb):
            idx = perm[b * 512 : (b + 1) * 512]
            opt.zero_grad()
            loss = wmse(model(Xtr[idx]), Ytr[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = wmse(model(Xva), Yva).item()
        if vl < best_val - 1e-5:
            best_val, bad = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 300:
                break
    model.load_state_dict(best_state)
    model.eval()
    print(f"\n{name}  (stopped ep={ep}, best val wmse={best_val:.4f})")
    with torch.no_grad():
        for sp, idx in (("train", tr), ("val", va), ("test", te)):
            pred = model(torch.tensor(F[idx], dtype=torch.float32, device=DEV)).cpu().numpy()
            r = per_variable_rmse(pred, Y[idx], cache["pca_models"], cache["outputs"])
            print(f"  {sp:5s} T={r['temperature']:.3f}  S={r['salinity']:.4f}")


for name, F in feats.items():
    run(name, F)
