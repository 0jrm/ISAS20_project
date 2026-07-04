#!/usr/bin/env python3
"""Diagnose L4-patch model collapse on chronological test split."""
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project/NeSPReSO2_onTemplate")
sys.path.insert(0, str(ROOT))

import model.model as module_arch
from model.metric import per_variable_rmse
from base.split_utils import build_split_indices

CACHE_DIR = Path("/unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project/data/cache")
PATCH_CACHE = CACHE_DIR / "train_ready_4411c65ee518.pkl"
POINT_CACHE = CACHE_DIR / "train_ready_ff2393a1ea21.pkl"
PATCH_CKPT = ROOT / "saved/models/NeSPReSO2_ARGO_GoM_patch_l4/0701_102436/model_best.pth"
POINT_CKPT = ROOT / "saved/models/NeSPReSO2_ARGO_GoM/argo16_scales/model_best.pth"

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_cache(p):
    with open(p, "rb") as f:
        return pickle.load(f)


def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    arch = cfg["arch"]
    model = getattr(module_arch, arch["type"])(**arch["args"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(DEV).eval()
    return model, cfg, ckpt.get("epoch")


def splits_for(cache, cfg):
    dl_args = cfg["data_loader"]["args"]
    dl_cfg = {
        "split_mode": dl_args.get("split_mode", "chronological"),
        "split_config": dl_args.get("split_config"),
        "train_frac": dl_args.get("train_frac", 0.7),
        "val_frac": dl_args.get("val_frac", 0.15),
        "test_frac": dl_args.get("test_frac", 0.15),
        "split_seed": dl_args.get("split_seed", 42),
        "unassigned": dl_args.get("unassigned", "exclude"),
    }
    n = cache["inputs"].shape[0]
    try:
        v2_src = cfg["io"].get("v2_src")
    except Exception:
        v2_src = None
    return build_split_indices(n, cache.get("JULD"), dl_cfg,
                               dataset_tag=cache.get("dataset_tag", "unknown"),
                               v2_src=v2_src)


@torch.no_grad()
def predict(model, X):
    outs = []
    for i in range(0, X.shape[0], 2048):
        xb = torch.tensor(X[i : i + 2048], dtype=torch.float32, device=DEV)
        outs.append(model(xb).cpu().numpy())
    return np.concatenate(outs)


def rmse_by_split(name, model, cache, splits):
    print(f"\n=== {name}: per-split profile RMSE ===")
    res = {}
    for sp in ("train", "val", "test"):
        idx = np.asarray(splits[sp], dtype=int)
        pred = predict(model, cache["inputs"][idx])
        tgt = cache["targets"][idx]
        r = per_variable_rmse(pred, tgt, cache["pca_models"], cache["outputs"])
        res[sp] = (pred, r)
        print(f"  {sp:5s} n={len(idx):5d}  T={r['temperature']:.3f}  S={r['salinity']:.4f}")
    return res


def monthly_rmse(cache, splits, pred_by_split, label):
    print(f"\n=== {label}: T RMSE by ~30-day bin (val+test) ===")
    from model.loss import reconstruct_physical_profiles
    for sp in ("val", "test"):
        idx = np.asarray(splits[sp], dtype=int)
        pred, _ = pred_by_split[sp]
        juld = cache["JULD"][idx]
        p = reconstruct_physical_profiles(pred, cache["pca_models"], cache["outputs"])
        t = reconstruct_physical_profiles(cache["targets"][idx], cache["pca_models"], cache["outputs"])
        diffT = p["temperature"] - t["temperature"]  # (depth, n)
        bins = ((juld - juld.min()) // 30).astype(int)
        for b in np.unique(bins):
            m = bins == b
            r = float(np.sqrt(np.mean(diffT[:, m] ** 2)))
            print(f"  {sp} bin{b:02d} juld[{juld[m].min():.0f},{juld[m].max():.0f}] n={m.sum():4d} T_RMSE={r:.3f}")


def ablations(model, cache, splits):
    X = cache["inputs"]
    n_enc = 10
    per_var = 175  # 7*5*5
    center = 6 * 25 + 12  # last time step, center pixel
    var_slices = {v: slice(n_enc + i * per_var, n_enc + (i + 1) * per_var) for i, v in enumerate(("sss", "sst", "ssh"))}
    tr = np.asarray(splits["train"], dtype=int)
    train_mean_patch = X[tr, n_enc:].mean(axis=0)
    train_mean_scal = X[tr, :n_enc].mean(axis=0)

    def eval_X(Xm, idx):
        pred = predict(model, Xm[idx])
        return per_variable_rmse(pred, cache["targets"][idx], cache["pca_models"], cache["outputs"])

    variants = {}
    variants["baseline"] = X

    Xc = X.copy()
    for v, sl in var_slices.items():
        c = X[:, sl.start + center]
        Xc[:, sl] = c[:, None]
    variants["patch->center broadcast"] = Xc

    Xm = X.copy()
    Xm[:, n_enc:] = train_mean_patch[None, :]
    variants["patch->train-mean (no local sat)"] = Xm

    Xz = X.copy()
    Xz[:, n_enc:] = 0.0
    variants["patch->zeros"] = Xz

    Xb = X.copy()
    Xb[:, 6:10] = train_mean_scal[6:10][None, :]
    variants["basin+bathy scalars->train-mean"] = Xb

    Xd = X.copy()  # demean ssh patch per sample (remove level, keep texture)
    sl = var_slices["ssh"]
    Xd[:, sl] = X[:, sl] - X[:, sl].mean(axis=1, keepdims=True)
    variants["ssh patch demeaned per-sample"] = Xd

    print("\n=== Patch-model inference ablations (T RMSE / S RMSE) ===")
    hdr = f"  {'variant':38s}" + "".join(f"{sp:>16s}" for sp in ("train", "val", "test"))
    print(hdr)
    for name, Xv in variants.items():
        row = f"  {name:38s}"
        for sp in ("train", "val", "test"):
            idx = np.asarray(splits[sp], dtype=int)
            r = eval_X(Xv, idx)
            row += f"  {r['temperature']:.3f}/{r['salinity']:.4f}"
        print(row)


def drift_stats(cache, splits):
    X = cache["inputs"]
    n_enc = 10
    names = ["timecos", "timesin", "latcos", "latsin", "loncos", "lonsin",
             "basin_sss", "basin_sst", "basin_ssh", "bathy_depth"]
    print("\n=== Scalar input mean (std) per split ===")
    for i, nm in enumerate(names):
        row = f"  {nm:10s}"
        for sp in ("train", "val", "test"):
            idx = np.asarray(splits[sp], dtype=int)
            row += f"  {X[idx, i].mean():9.4f}({X[idx, i].std():.3f})"
        print(row)
    per_var = 175
    print("\n=== Patch-block mean (std of per-sample means) per split ===")
    for j, v in enumerate(("sss", "sst", "ssh")):
        sl = slice(n_enc + j * per_var, n_enc + (j + 1) * per_var)
        row = f"  {v:10s}"
        for sp in ("train", "val", "test"):
            idx = np.asarray(splits[sp], dtype=int)
            pm = X[idx, sl].mean(axis=1)
            row += f"  {pm.mean():9.4f}({pm.std():.3f})"
        print(row)
    print("\n=== Target PC variance (first 4 T PCs) per split ===")
    T = cache["targets"]
    for sp in ("train", "val", "test"):
        idx = np.asarray(splits[sp], dtype=int)
        print(f"  {sp:5s} var={np.var(T[idx, :4], axis=0).round(2)} mean={T[idx, :4].mean(axis=0).round(2)}")


def main():
    patch_cache = load_cache(PATCH_CACHE)
    point_cache = load_cache(POINT_CACHE)
    patch_model, patch_cfg, patch_ep = load_model(PATCH_CKPT)
    point_model, point_cfg, point_ep = load_model(POINT_CKPT)
    print(f"patch ckpt epoch={patch_ep}  point ckpt epoch={point_ep}")
    print(f"patch N={patch_cache['inputs'].shape}  point N={point_cache['inputs'].shape}")

    sp_patch = splits_for(patch_cache, patch_cfg)
    sp_point = splits_for(point_cache, point_cfg)
    for sp in ("train", "val", "test"):
        a, b = np.asarray(sp_patch[sp]), np.asarray(sp_point[sp])
        same = len(a) == len(b) and np.array_equal(a, b)
        print(f"split {sp}: patch n={len(a)} point n={len(b)} identical={same}")

    res_point = rmse_by_split("POINT (raw)", point_model, point_cache, sp_point)
    res_patch = rmse_by_split("PATCH L4 (raw)", patch_model, patch_cache, sp_patch)

    monthly_rmse(point_cache, sp_point, res_point, "POINT")
    monthly_rmse(patch_cache, sp_patch, res_patch, "PATCH")

    drift_stats(patch_cache, sp_patch)
    ablations(patch_model, patch_cache, sp_patch)


if __name__ == "__main__":
    main()
