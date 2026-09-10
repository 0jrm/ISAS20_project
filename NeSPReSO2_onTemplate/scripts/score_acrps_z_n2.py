#!/usr/bin/env python3
"""N² smoothness of mean test profiles: A_CRPS s42 / z32 vs native-z CRPS cells."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_REPO = _ROOT.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evalphys.metrics import static_stability_violations
from scripts.thermocline_scorecard import _load_ckpt_pred, _try_real_bundle

CELLS = (
    (
        "A_CRPS_s42",
        "config/argo/config_argo_profile_direct.json",
        "saved/phase5_matrix/A_CRPS_v2/models/NeSPReSO2_ARGO_GoM_p5_A_CRPS_v2_p5_A_CRPS_v2_s42_s2/p5_A_CRPS_v2_s42_s2/model_best.pth",
    ),
    (
        "A_CRPS_z32",
        "config/argo/config_argo_A_CRPS_z32.json",
        "saved/acrps_phys_pca32_b/models/NeSPReSO2_ARGO_GoM_acrps_phys_pca32_b_acrps_phys_pca32_b_s42_s2/acrps_phys_pca32_b_s42_s2/model_best.pth",
    ),
    (
        "A_CRPS_z",
        "config/argo/config_argo_A_CRPS_z.json",
        "saved/acrps_z/models/NeSPReSO2_ARGO_GoM_A_CRPS_z_acrps_z_s42_s2/acrps_z_s42_s2/model_best.pth",
    ),
    (
        "A_CRPS_z_roni_ops",
        "config/argo/config_argo_A_CRPS_z_roni_ops.json",
        "saved/acrps_z_roni_ops/models/NeSPReSO2_ARGO_GoM_A_CRPS_z_roni_ops_acrps_z_roni_ops_s42_s2/acrps_z_roni_ops_s42_s2/model_best.pth",
    ),
)


def _mean_profile_n2(T, S, z, lat, lon):
    t = np.nanmean(T, axis=0, keepdims=True)
    s = np.nanmean(S, axis=0, keepdims=True)
    lat_m = np.array([float(np.nanmean(lat))])
    lon_m = np.array([float(np.nanmean(lon))])
    stab = static_stability_violations(t, s, z, lat_m, lon_m)
    d2t = t[0, 2:] - 2.0 * t[0, 1:-1] + t[0, :-2]
    d2s = s[0, 2:] - 2.0 * s[0, 1:-1] + s[0, :-2]
    return {
        "n2_profile": stab.get("violation_rate_profile"),
        "n2_level": stab.get("violation_rate_level"),
        "mean_T_d2_rms": float(np.sqrt(np.nanmean(d2t ** 2))),
        "mean_S_d2_rms": float(np.sqrt(np.nanmean(d2s ** 2))),
    }


def score_cell(label, cfg_rel, ckpt_rel):
    cfg_path = _ROOT / cfg_rel
    ckpt = _ROOT / ckpt_rel
    row = {"label": label, "checkpoint": str(ckpt), "exists": ckpt.is_file()}
    if not ckpt.is_file():
        row["error"] = "missing checkpoint"
        return row
    bundle, err = _try_real_bundle(cfg_path)
    if bundle is None:
        row["error"] = err
        return row
    pred = _load_ckpt_pred(ckpt, bundle)
    if pred is None:
        row["error"] = "decode failed"
        return row
    T_hat, S_hat = pred
    T_true, S_true = bundle["T_true"], bundle["S_true"]
    z, lat, lon = bundle["z"], bundle["lat"], bundle["lon"]
    stab = static_stability_violations(T_hat, S_hat, z, lat, lon)
    row["n"] = int(T_hat.shape[0])
    row["T_rmse"] = float(np.sqrt(np.nanmean((T_hat - T_true) ** 2)))
    row["S_rmse"] = float(np.sqrt(np.nanmean((S_hat - S_true) ** 2)))
    row["n2_profile"] = stab.get("violation_rate_profile")
    row["n2_level"] = stab.get("violation_rate_level")
    row["mean_profile"] = _mean_profile_n2(T_hat, S_hat, z, lat, lon)
    row["mean_profile_true"] = _mean_profile_n2(T_true, S_true, z, lat, lon)
    return row


def main() -> int:
    rows = [score_cell(*c) for c in CELLS]
    out = _REPO / "reports" / "eval_A_CRPS_z_n2.json"
    out.write_text(json.dumps({"cells": rows}, indent=2) + "\n")
    print(json.dumps({"cells": rows}, indent=2))
    md = ["# Native-z CRPS vs A_CRPS s42 / z32 (N²)", "", "| cell | T RMSE | S RMSE | N² profile | N² level | mean-profile N² level | mean T d² RMS |", "|------|-------:|-------:|-----------:|---------:|----------------------:|--------------:|"]
    for r in rows:
        if r.get("error"):
            md.append(f"| {r['label']} | — | — | {r['error']} | | | |")
            continue
        mp = r["mean_profile"]
        md.append(
            f"| {r['label']} | {r['T_rmse']:.3f} | {r['S_rmse']:.3f} | "
            f"{r['n2_profile']:.3f} | {r['n2_level']:.4f} | "
            f"{mp['n2_level']:.4f} | {mp['mean_T_d2_rms']:.4g} |"
        )
    (_REPO / "reports" / "eval_A_CRPS_z_n2.md").write_text("\n".join(md) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
