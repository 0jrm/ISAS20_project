"""Within-cell permutation nulls for pattern correlation, 20 °C RMSE, and innovation."""

from __future__ import annotations

import numpy as np
import pandas as pd

PERM_N_DEFAULT = 200
BAND_LO = 50.0
BAND_HI = 200.0


def _cell_mask(casts: pd.DataFrame, slice_name: str, era_name: str) -> np.ndarray:
    era_ok = np.ones(len(casts), dtype=bool) if era_name == "all" else casts["era"].to_numpy() == era_name
    if slice_name == "in_bbox":
        return era_ok & casts["in_bbox"].to_numpy()
    if slice_name == "ood":
        return era_ok & casts["ood"].to_numpy()
    if slice_name == "lc":
        return era_ok & casts["lc"].to_numpy()
    raise ValueError(slice_name)


def murphy_terms(t_pred: np.ndarray, t_true: np.ndarray, valid: np.ndarray) -> tuple[float, float, float]:
    """bias² and amplitude on pooled valid levels. Pattern r is the mean of per-level anomaly r."""
    m = valid & np.isfinite(t_pred) & np.isfinite(t_true)
    if int(m.sum()) < 4:
        return float("nan"), float("nan"), float("nan")
    err = t_pred[m] - t_true[m]
    bias2 = float(np.mean(err) ** 2)
    ap = t_pred[m] - np.mean(t_pred[m])
    at = t_true[m] - np.mean(t_true[m])
    sp = float(np.std(ap, ddof=0))
    st = float(np.std(at, ddof=0))
    amp = sp / st if st > 0 else float("nan")
    if t_pred.ndim == 2 and t_true.ndim == 2 and valid.ndim == 2:
        patt = float(np.nanmean(per_level_anomaly_r(t_pred, t_true, valid)))
    else:
        if sp == 0 or st == 0:
            patt = float("nan")
        else:
            patt = float(np.corrcoef(ap, at)[0, 1])
    return bias2, amp, patt


def per_level_anomaly_r(t_pred: np.ndarray, t_true: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Correlation across casts at each depth after removing that depth's mean profile."""
    pred = np.asarray(t_pred, dtype=np.float64)
    truth = np.asarray(t_true, dtype=np.float64)
    ok = np.asarray(valid, dtype=bool) & np.isfinite(pred) & np.isfinite(truth)
    n_z = pred.shape[1]
    r = np.full(n_z, np.nan, dtype=np.float64)
    for j in range(n_z):
        m = ok[:, j]
        if int(m.sum()) < 4:
            continue
        p = pred[m, j] - np.mean(pred[m, j])
        t = truth[m, j] - np.mean(truth[m, j])
        sp = float(np.std(p, ddof=0))
        st = float(np.std(t, ddof=0))
        if sp == 0.0 or st == 0.0:
            continue
        r[j] = float(np.corrcoef(p, t)[0, 1])
    return r


def _cast_band_stack(levels: pd.DataFrame, cast_ids: np.ndarray, col: str) -> tuple[np.ndarray, np.ndarray]:
    z = levels["z"].to_numpy()
    in_band = (z >= BAND_LO) & (z < BAND_HI)
    sub = levels.loc[in_band]
    ids = sub["cast_id"].to_numpy()
    vals = sub[col].to_numpy()
    valid = sub["valid_t"].to_numpy()
    order = {int(c): i for i, c in enumerate(cast_ids)}
    n = len(cast_ids)
    z_band = np.unique(sub["z"].to_numpy())
    nz = z_band.size
    z_index = {float(v): j for j, v in enumerate(z_band)}
    out = np.full((n, nz), np.nan)
    tok = np.zeros((n, nz), dtype=bool)
    for cid, val, ok, zi in zip(ids, vals, valid, sub["z"].to_numpy()):
        i = order.get(int(cid))
        if i is None:
            continue
        j = z_index[float(zi)]
        out[i, j] = val
        tok[i, j] = bool(ok)
    return out, tok


def cell_observables(casts: pd.DataFrame, levels: pd.DataFrame) -> dict[str, float]:
    t_nes, ok = _cast_band_stack(levels, casts["cast_id"].to_numpy(), "t_nes")
    t_xb, ok_xb = _cast_band_stack(levels, casts["cast_id"].to_numpy(), "t_xb")
    t_argo, ok_a = _cast_band_stack(levels, casts["cast_id"].to_numpy(), "t_argo")
    valid = ok & ok_xb & ok_a
    bias2, amp, patt = murphy_terms(t_nes, t_argo, valid)
    bias2_xb, amp_xb, patt_xb = murphy_terms(t_xb, t_argo, valid)

    def _mean_valid(arr, m):
        out = np.full(arr.shape[0], np.nan)
        for i in range(arr.shape[0]):
            sel = m[i]
            if sel.any():
                out[i] = float(np.mean(arr[i, sel]))
        return out

    y = _mean_valid(t_argo - t_xb, valid)
    x = _mean_valid(t_nes - t_xb, valid)
    both = np.isfinite(x) & np.isfinite(y)
    if int(both.sum()) < 4:
        slope = float("nan")
        sign_ag = float("nan")
    else:
        slope = float(np.polyfit(x[both], y[both], 1)[0])
        sign_ag = float(np.mean(np.sign(x[both]) == np.sign(y[both])))
    z20_rmse = float(
        np.sqrt(np.nanmean((casts["nes_z20_m"].to_numpy() - casts["argo_z20_m"].to_numpy()) ** 2))
    )
    z20_rmse_xb = float(
        np.sqrt(np.nanmean((casts["xb_z20_m"].to_numpy() - casts["argo_z20_m"].to_numpy()) ** 2))
    )
    return {
        "n_cast": float(len(casts)),
        "murphy_bias2": bias2,
        "murphy_amp": amp,
        "murphy_pattern": patt,
        "murphy_xb_bias2": bias2_xb,
        "murphy_xb_amp": amp_xb,
        "murphy_xb_pattern": patt_xb,
        "z20_rmse_nes": z20_rmse,
        "z20_rmse_xb": z20_rmse_xb,
        "innov_slope": slope,
        "innov_sign": sign_ag,
    }


def permutation_cache(
    casts: pd.DataFrame,
    levels: pd.DataFrame,
    *,
    n_perm: int = PERM_N_DEFAULT,
    seed: int = 0,
) -> pd.DataFrame:
    slices = ("in_bbox", "ood", "lc")
    eras = ("all", "2024", "2025")
    rows = []
    depth_rows = []
    rng = np.random.default_rng(seed)
    for model, gmodel in casts.groupby("model", sort=False):
        lev_m = levels.loc[levels["model"] == model]
        for slice_name in slices:
            for era_name in eras:
                mask = _cell_mask(gmodel, slice_name, era_name)
                cell = gmodel.loc[mask]
                if len(cell) < 8:
                    continue
                cell_ids = set(cell["cast_id"].to_numpy())
                lev_c = lev_m.loc[lev_m["cast_id"].isin(cell_ids)]
                obs = cell_observables(cell, lev_c)
                t_nes, ok = _cast_band_stack(lev_c, cell["cast_id"].to_numpy(), "t_nes")
                t_xb, ok_xb = _cast_band_stack(lev_c, cell["cast_id"].to_numpy(), "t_xb")
                t_argo, ok_a = _cast_band_stack(lev_c, cell["cast_id"].to_numpy(), "t_argo")
                valid = ok & ok_xb & ok_a
                z20_nes = cell["nes_z20_m"].to_numpy()
                z20_argo = cell["argo_z20_m"].to_numpy()
                n = len(cell)

                def _mean_valid(arr, m):
                    out = np.full(arr.shape[0], np.nan)
                    for i in range(arr.shape[0]):
                        sel = m[i]
                        if sel.any():
                            out[i] = float(np.mean(arr[i, sel]))
                    return out

                y = _mean_valid(t_argo - t_xb, valid)
                r_obs = per_level_anomaly_r(t_nes, t_argo, valid)
                r_xb = per_level_anomaly_r(t_xb, t_argo, valid)
                null_patt = []
                null_z20 = []
                null_slope = []
                null_sign = []
                null_r = []
                for _ in range(n_perm):
                    perm = rng.permutation(n)
                    t_sh = t_nes[perm]
                    z20_sh = z20_nes[perm]
                    v_sh = valid[perm]
                    v = v_sh & valid
                    _, _, patt = murphy_terms(t_sh, t_argo, v)
                    null_patt.append(patt)
                    null_r.append(per_level_anomaly_r(t_sh, t_argo, v))
                    null_z20.append(float(np.sqrt(np.nanmean((z20_sh - z20_argo) ** 2))))
                    x = _mean_valid(t_sh - t_xb, v)
                    both = np.isfinite(x) & np.isfinite(y)
                    if int(both.sum()) < 4:
                        null_slope.append(np.nan)
                        null_sign.append(np.nan)
                    else:
                        null_slope.append(float(np.polyfit(x[both], y[both], 1)[0]))
                        null_sign.append(float(np.mean(np.sign(x[both]) == np.sign(y[both]))))
                def zscore(obs_v, null):
                    arr = np.asarray(null, dtype=np.float64)
                    mu = float(np.nanmean(arr))
                    sd = float(np.nanstd(arr))
                    if not np.isfinite(sd) or sd == 0:
                        return float("nan")
                    return float((obs_v - mu) / sd)

                rows.append(
                    {
                        "model": model,
                        "slice": slice_name,
                        "era": era_name,
                        **obs,
                        "n_perm": n_perm,
                        "null_pattern_mean": float(np.nanmean(null_patt)),
                        "null_pattern_std": float(np.nanstd(null_patt)),
                        "z_pattern": zscore(obs["murphy_pattern"], null_patt),
                        "null_z20_rmse_mean": float(np.nanmean(null_z20)),
                        "z_z20_rmse": zscore(obs["z20_rmse_nes"], null_z20),
                        "null_innov_slope_mean": float(np.nanmean(null_slope)),
                        "z_innov_slope": zscore(obs["innov_slope"], null_slope),
                        "null_innov_sign_mean": float(np.nanmean(null_sign)),
                        "z_innov_sign": zscore(obs["innov_sign"], null_sign),
                    }
                )
                z_band = np.unique(
                    lev_c.loc[(lev_c["z"] >= BAND_LO) & (lev_c["z"] < BAND_HI), "z"].to_numpy(
                        dtype=np.float64
                    )
                )
                null_r_arr = np.vstack(null_r) if null_r else np.empty((0, z_band.size))
                for j, zj in enumerate(z_band):
                    depth_rows.append(
                        {
                            "model": model,
                            "slice": slice_name,
                            "era": era_name,
                            "z": float(zj),
                            "r_nes": float(r_obs[j]) if j < r_obs.size else float("nan"),
                            "r_xb": float(r_xb[j]) if j < r_xb.size else float("nan"),
                            "r_null_mean": float(np.nanmean(null_r_arr[:, j])) if null_r_arr.size else float("nan"),
                            "r_null_std": float(np.nanstd(null_r_arr[:, j])) if null_r_arr.size else float("nan"),
                            "n_perm": n_perm,
                        }
                    )
    return pd.DataFrame(rows), pd.DataFrame(depth_rows)
