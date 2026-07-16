#!/usr/bin/env python3
"""Blame-split swap test: (true σ₀, pred τ) vs (pred σ₀, true τ) → T RMSE.

Decomposes density_spice overall-T error into density-error vs spice-error
contributions in one eval pass (no retraining).
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

ARGO16_T_RMSE_OVERALL = 0.4158


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main() -> int:
    from base.split_utils import build_split_indices
    from evalphys.gsw_backend import get_gsw, set_headline_frozen
    from evalphys.inversion import sigma0_spice_from_ts, ts_from_sigma0_spice
    from model.density_spice import decode_sigma0_ctrl, encode_a_from_sigma0_ctrl, upsample_pchip
    from model.model import PatchConvMLP
    from preproc.export_v2_cache import build_argo_cache
    from scripts.eval_density_spice import decode_density_spice_to_ts

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-c", "--config", required=True)
    p.add_argument("-r", "--resume", required=True, help="density_spice checkpoint")
    p.add_argument("--out-md", default="../reports/phase3_blame_swap.md")
    p.add_argument("--out-json", default="../reports/phase3_blame_swap.json")
    args = p.parse_args()

    set_headline_frozen(True)
    cfg = json.loads(Path(args.config).read_text())
    cache_path = build_argo_cache(cfg)
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)

    n = int(cache["inputs"].shape[0])
    dl = cfg["data_loader"]["args"]
    idx = build_split_indices(
        n,
        cache.get("JULD"),
        {
            "split_mode": dl.get("split_mode", "chronological"),
            "train_frac": float(dl.get("train_frac", 0.7)),
            "val_frac": float(dl.get("val_frac", 0.15)),
            "test_frac": float(dl.get("test_frac", 0.15)),
            "split_seed": int(dl.get("split_seed", 42)),
            "unassigned": dl.get("unassigned", "exclude"),
        },
        dataset_tag=cfg["io"].get("dataset_tag", "argo_v2"),
        v2_src=cfg["io"].get("v2_src"),
    )["test"]

    ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
    arch = dict(cfg["arch"]["args"])
    arch.setdefault("probabilistic", False)
    model = PatchConvMLP(**arch)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    X = torch.tensor(cache["inputs"][idx], dtype=torch.float32)
    with torch.no_grad():
        mu_raw = model(X).numpy()

    meta = cache["density_spice_meta"]
    pca = cache["pca_models"]["spice"]
    z_ctrl = np.asarray(meta["z_ctrl"], dtype=np.float64)
    dz = np.asarray(meta["dz_tilde"], dtype=np.float64)
    k = int(meta["K"])
    n_spice = int(meta["n_spice"])
    a_clim = meta.get("a_clim")
    if a_clim is None:
        a_clim = encode_a_from_sigma0_ctrl(
            np.asarray(meta["sigma0_ctrl_mean"], dtype=np.float64), dz, z_ctrl
        )
    a = mu_raw[:, :k] + np.asarray(a_clim, dtype=np.float64)
    with torch.no_grad():
        sig_ctrl_pred = decode_sigma0_ctrl(
            torch.from_numpy(a.astype(np.float32)),
            torch.from_numpy(dz.astype(np.float32)),
        ).numpy()
    depth = np.asarray(cache["PRES"], dtype=np.float64).reshape(-1)
    sig_pred = upsample_pchip(sig_ctrl_pred, z_ctrl, depth)
    tau_pred = pca.inverse_transform(mu_raw[:, k : k + n_spice]) * np.asarray(
        meta["spice_std"], dtype=np.float64
    ) + np.asarray(meta["spice_mean"], dtype=np.float64)

    T = np.asarray(cache["profiles"]["temperature"], dtype=np.float64).T[idx]
    S = np.asarray(cache["profiles"]["salinity"], dtype=np.float64).T[idx]
    lat = np.asarray(cache["LAT"], dtype=np.float64)[idx]
    lon = np.asarray(cache["LON"], dtype=np.float64)[idx]
    gsw = get_gsw()
    n_prof, n_lev = T.shape
    p_grid = gsw.p_from_z(-np.broadcast_to(depth, (n_prof, n_lev)), lat[:, None])
    sig_true, tau_true = sigma0_spice_from_ts(T, S, p_grid, lon[:, None], lat[:, None])

    T_both, _, _ = decode_density_spice_to_ts(mu_raw, cache, indices=idx)
    T_true_sig_pred_tau, _, ok1 = ts_from_sigma0_spice(
        sig_true, tau_pred, p_grid, lon[:, None], lat[:, None]
    )
    T_pred_sig_true_tau, _, ok2 = ts_from_sigma0_spice(
        sig_pred, tau_true, p_grid, lon[:, None], lat[:, None]
    )

    rows = {
        "pred_both": _rmse(T_both, T),
        "true_sigma0_pred_tau": _rmse(T_true_sig_pred_tau, T),  # spice error only
        "pred_sigma0_true_tau": _rmse(T_pred_sig_true_tau, T),  # density error only
        "argo16_overall": ARGO16_T_RMSE_OVERALL,
        "gap_to_argo16x1_10": ARGO16_T_RMSE_OVERALL * 1.10,
        "n_test": int(len(idx)),
        "inv_fail_frac_spice_only": float(1.0 - np.mean(ok1)),
        "inv_fail_frac_density_only": float(1.0 - np.mean(ok2)),
        "checkpoint": str(args.resume),
    }
    # Which branch owns more of the T error?
    rows["dominant_branch"] = (
        "density"
        if rows["pred_sigma0_true_tau"] > rows["true_sigma0_pred_tau"]
        else "spice"
    )

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rows, indent=2) + "\n")
    md = [
        "# Phase 3 — blame-split swap test",
        "",
        f"**Checkpoint:** `{args.resume}`",
        f"**Test n:** {rows['n_test']}",
        "",
        "| Reconstruction | overall T RMSE | vs argo16×1.10 |",
        "|----------------|----------------|----------------|",
        f"| pred σ₀ + pred τ (full) | {rows['pred_both']:.4f} | {rows['pred_both']/rows['gap_to_argo16x1_10']:.2f}× |",
        f"| **true σ₀ + pred τ** (spice error) | {rows['true_sigma0_pred_tau']:.4f} | {rows['true_sigma0_pred_tau']/rows['gap_to_argo16x1_10']:.2f}× |",
        f"| **pred σ₀ + true τ** (density error) | {rows['pred_sigma0_true_tau']:.4f} | {rows['pred_sigma0_true_tau']/rows['gap_to_argo16x1_10']:.2f}× |",
        f"| gate floor (argo16×1.10) | {rows['gap_to_argo16x1_10']:.4f} | 1.00× |",
        "",
        f"**Dominant branch:** `{rows['dominant_branch']}` "
        f"(higher T RMSE when that branch is predicted and the other is truth).",
        "",
        "Read: if density-error row ≫ spice-error row, the 0.72→0.46 gap lives in σ₀; "
        "decouple / density-only next. If the reverse, spice still owns skill.",
        "",
    ]
    out_md.write_text("\n".join(md) + "\n")
    print(json.dumps(rows, indent=2))
    print(f"wrote {out_md} and {out_json}")
    # ponytail: ceiling = overall T RMSE only; upgrade = depth-band table if needed
    assert rows["pred_both"] > 0.0
    assert rows["true_sigma0_pred_tau"] > 0.0 and rows["pred_sigma0_true_tau"] > 0.0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
