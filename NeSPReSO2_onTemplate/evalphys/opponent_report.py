"""20 °C RMSE and per-level anomaly r for A_CRPS, noprofile xb, persistence, and WOA."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from evalphys.permutation_null import BAND_HI, BAND_LO, per_level_anomaly_r


def z20_rmse(pred: pd.Series, truth: pd.Series) -> float:
    ok = pred.notna() & truth.notna()
    if int(ok.sum()) == 0:
        return float("nan")
    err = pred.loc[ok].to_numpy(dtype=np.float64) - truth.loc[ok].to_numpy(dtype=np.float64)
    return float(np.sqrt(np.mean(err * err)))


def mean_per_level_r(t_pred: np.ndarray, t_true: np.ndarray, valid: np.ndarray) -> float:
    return float(np.nanmean(per_level_anomaly_r(t_pred, t_true, valid)))


def stack_level_field(levels: pd.DataFrame, cast_ids: np.ndarray, col: str) -> tuple[np.ndarray, np.ndarray]:
    z = np.sort(levels["z"].drop_duplicates().to_numpy(dtype=np.float64))
    z_index = {float(v): j for j, v in enumerate(z)}
    order = {int(c): i for i, c in enumerate(cast_ids)}
    n = len(cast_ids)
    out = np.full((n, z.size), np.nan)
    tok = np.zeros((n, z.size), dtype=bool)
    ids = levels["cast_id"].to_numpy()
    vals = levels[col].to_numpy(dtype=np.float64)
    zs = levels["z"].to_numpy(dtype=np.float64)
    valid = levels["valid_t"].to_numpy() if "valid_t" in levels.columns else np.ones(len(levels), dtype=bool)
    for cid, val, zi, ok in zip(ids, vals, zs, valid):
        i = order.get(int(cid))
        if i is None:
            continue
        j = z_index[float(zi)]
        out[i, j] = val
        tok[i, j] = bool(ok) and np.isfinite(val)
    band = (z >= BAND_LO) & (z < BAND_HI)
    return out[:, band], tok[:, band]


def gate_casts(casts: pd.DataFrame) -> pd.DataFrame:
    return casts.loc[(casts["era"] == "2025") & casts["in_bbox"]].copy()


def format_a2_report(
    casts: pd.DataFrame,
    levels: pd.DataFrame,
    perm_z: pd.DataFrame,
    woa_long: pd.DataFrame,
    woa_feat: pd.DataFrame,
    parquet_stem: str,
) -> str:
    gate = gate_casts(casts)
    ids = gate.loc[gate["model"] == "A_CRPS", "cast_id"].to_numpy()
    lines = [
        f"parquet {parquet_stem}",
        "slice 2025 in_bbox. Pattern r is the mean of per-level anomaly r on 50-200 m.",
        "This file is a report. It is not a go-rule.",
        "",
        f"{'member':16s}  {'z20_rmse_m':>11s}  {'per_level_r':>11s}",
    ]
    rows = []
    xb = gate.loc[gate["model"] == "A_CRPS"]
    rows.append(("ops_xb", z20_rmse(xb["xb_z20_m"], xb["argo_z20_m"]), _r_from_perm(perm_z, "A_CRPS", "r_xb")))
    for model, label in (("A_CRPS", "A_CRPS"), ("xb_noprofile", "xb_noprofile"), ("persistence", "persistence")):
        sub = gate.loc[gate["model"] == model]
        rmse = z20_rmse(sub["nes_z20_m"], sub["argo_z20_m"])
        if model == "persistence":
            r = _r_from_levels(levels, sub["cast_id"].to_numpy(), "t_nes", "persistence")
        else:
            r = _r_from_perm(perm_z, model, "r_nes")
        rows.append((label, rmse, r))
    joined = xb.merge(woa_feat, on="cast_id", how="left")
    woa_rmse = z20_rmse(joined["woa_z20_m"], joined["argo_z20_m"]) if "woa_z20_m" in joined.columns else float("nan")
    woa_r = _r_woa(levels, woa_long, ids)
    rows.append(("WOA23", woa_rmse, woa_r))
    for label, rmse, r in rows:
        lines.append(f"{label:16s}  {rmse:11.2f}  {r:11.4f}")
    n_np = int(gate.loc[gate["model"] == "xb_noprofile", "nes_z20_m"].notna().sum())
    lines.append("")
    lines.append(f"xb_noprofile 20 C hits in this slice: {n_np}")
    return "\n".join(lines) + "\n"


def _r_from_perm(perm_z: pd.DataFrame, model: str, col: str) -> float:
    sub = perm_z.loc[
        (perm_z["model"] == model)
        & (perm_z["era"] == "2025")
        & (perm_z["slice"] == "in_bbox")
        & (perm_z["z"] >= BAND_LO)
        & (perm_z["z"] < BAND_HI)
    ]
    if sub.empty or col not in sub.columns:
        return float("nan")
    return float(sub[col].mean())


def _r_from_levels(levels: pd.DataFrame, cast_ids: np.ndarray, col: str, model: str) -> float:
    sub = levels.loc[levels["cast_id"].isin(set(cast_ids.tolist()))]
    if "model" in sub.columns:
        sub = sub.loc[sub["model"] == model]
    pred, ok = stack_level_field(sub, cast_ids, col)
    truth, ok_t = stack_level_field(sub, cast_ids, "t_argo")
    return mean_per_level_r(pred, truth, ok & ok_t)


def _r_woa(levels: pd.DataFrame, woa_long: pd.DataFrame, cast_ids: np.ndarray) -> float:
    argo = levels.loc[(levels["model"] == "A_CRPS") & levels["cast_id"].isin(set(cast_ids.tolist()))]
    if argo.empty or woa_long.empty:
        return float("nan")
    merged = argo.merge(woa_long, on=["cast_id", "z"], how="inner")
    merged["valid_t"] = merged["valid_t"] & np.isfinite(merged["t_woa"])
    pred_tbl = merged.rename(columns={"t_woa": "t_pred"})
    pred, ok = stack_level_field(pred_tbl, cast_ids, "t_pred")
    truth, ok_t = stack_level_field(merged, cast_ids, "t_argo")
    return mean_per_level_r(pred, truth, ok & ok_t)


def write_a2_report(text: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path
