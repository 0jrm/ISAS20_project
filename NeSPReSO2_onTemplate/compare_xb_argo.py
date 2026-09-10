#!/usr/bin/env python3
"""Score shortlisted NeSPReSO cells vs TSIS Argo / xb / xa on inov z-levels.

Interpolate native 1 m (0–1800) onto profiles_xb.nc z. Score only valid==1
and z<=1800. Satellite path is nespreso_api GOFFISH (SMAP/MUR/AVISO), not the
2015–2022 cube. Do not mix checkpoint PCA with another cache.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_REPO = _ROOT.parent
_API = Path("/unity/g2/jmiranda/nespreso_api")
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

Z_NES_MAX = 1800.0
Z_NEAR_M = 10.0
TRAIN_LAT = (19.23, 28.70)
TRAIN_LON = (-96.80, -82.67)
NC_DEFAULT = _REPO / "profiles_xb.nc"
ZIP_DEFAULT = _REPO / "abozec_argo_xb_profiles.zip"
OUT_DEFAULT = _REPO / "reports" / "xb_argo_compare"
API_DEFAULT = _API
CACHE_PCA32 = _REPO / "data/cache/train_ready_4ee013852d33.pkl"
CACHE_OPS = _REPO / "data/cache/train_ready_heave_ops_pca32.pkl"
CACHE_P16 = _REPO / "data/cache/train_ready_3adcff404b0b.pkl"

# Census bbox. ~11% of this file sits east of -82.67.
MODELS = (
    {
        "name": "v1",
        "kind": "ozavala",
        "track": "map",
        "v2_name": None,
        "features": "A_CRPS",
        "ckpt": _API / "models" / "ocean_tensorscript.pt",
        "cache": _API / "models" / "pca_stats.pkl",
    },
    {
        "name": "A_CRPS",
        "kind": "local_pca",
        "track": "map",
        "v2_name": "A_CRPS",
        "features": "A_CRPS",
        "ckpt": _ROOT
        / "saved/phase5_matrix/A_CRPS_v2/models/NeSPReSO2_ARGO_GoM_p5_A_CRPS_v2_p5_A_CRPS_v2_s42_s2/p5_A_CRPS_v2_s42_s2/model_best.pth",
        "cache": CACHE_P16,
    },
    {
        "name": "HeaveFast",
        "kind": "v2_cell",
        "track": "map",
        "v2_name": "HeaveFast",
        "features": "HeaveFast",
        "ckpt": _ROOT
        / "saved/models/NeSPReSO2_ARGO_GoM_heave_residual_fast/heave_fast_s42/model_best.pth",
        "cache": CACHE_P16,
    },
    {
        "name": "A_CRPS_z32",
        "kind": "local_pca",
        "track": "map",
        "v2_name": None,
        "features": "A_CRPS",
        "ckpt": _ROOT
        / "saved/acrps_phys_pca32_b/models/NeSPReSO2_ARGO_GoM_acrps_phys_pca32_b_acrps_phys_pca32_b_s42_s2/acrps_phys_pca32_b_s42_s2/model_best.pth",
        "cache": CACHE_PCA32,
    },
    {
        "name": "stoch_eof",
        "kind": "local_pca",
        "track": "map",
        "v2_name": None,
        "features": "A_CRPS",
        "ckpt": _ROOT
        / "saved/stoch_eof/models/NeSPReSO2_ARGO_GoM_stoch_eof_stoch_eof_s42_s2/stoch_eof_s42_s2/model_best.pth",
        "cache": CACHE_PCA32,
    },
    {
        "name": "A_CRPS_z32_ops",
        "kind": "local_pca",
        "track": "map",
        "v2_name": None,
        "features": "ops",
        "ckpt": _ROOT
        / "saved/acrps_z32_ops/models/NeSPReSO2_ARGO_GoM_A_CRPS_z32_ops_acrps_z32_ops_s42_s2/acrps_z32_ops_s42_s2/model_best.pth",
        "cache": CACHE_OPS,
    },
    {
        "name": "pc_aux",
        "kind": "local_pca",
        "track": "map",
        "v2_name": None,
        "features": "ops",
        "ckpt": _ROOT / "saved/pc_aux/models/NeSPReSO2_ARGO_GoM_pc_aux_pc_aux_s2/pc_aux_s2/model_best.pth",
        "cache": CACHE_OPS,
    },
    {
        "name": "stoch_eof_pair",
        "kind": "pair_pca",
        "track": "pair",
        "mix_flag": None,
        "v2_name": None,
        "features": "A_CRPS",
        "ckpt": _ROOT
        / "saved/stoch_eof_pair/models/NeSPReSO2_ARGO_GoM_stoch_eof_pair_stoch_eof_pair_s42_s2/stoch_eof_pair_s42_s2/model_best.pth",
        "cache": CACHE_PCA32,
    },
    {
        "name": "stoch_eof_pair_mix",
        "kind": "pair_pca",
        "track": "pair",
        "mix_flag": 1.0,
        "v2_name": None,
        "features": "A_CRPS",
        "ckpt": _ROOT
        / "saved/stoch_eof_pair_mix/models/NeSPReSO2_ARGO_GoM_stoch_eof_pair_mix_stoch_eof_pair_mix_s42_s2/stoch_eof_pair_mix_s42_s2/model_best.pth",
        "cache": CACHE_PCA32,
    },
    {
        "name": "persistence",
        "kind": "persistence",
        "track": "pair",
        "v2_name": None,
        "features": "A_CRPS",
        "ckpt": None,
        "cache": None,
    },
)

COLORS = {
    "xb": "#7A7A7A",
    "xa": "#111111",
    "v1": "#1D3557",
    "A_CRPS": "#C81D25",
    "HeaveFast": "#3D348B",
    "A_CRPS_z32": "#0B6E4F",
    "stoch_eof": "#2A9D8F",
    "A_CRPS_z32_ops": "#E09F3E",
    "pc_aux": "#9B5DE5",
    "stoch_eof_pair": "#00BBF9",
    "stoch_eof_pair_mix": "#F15BB5",
    "persistence": "#6C757D",
}


def interp_native_to_obs(
    src_z: np.ndarray, T: np.ndarray, S: np.ndarray, dst_z: np.ndarray, zmax: float = Z_NES_MAX
) -> tuple[np.ndarray, np.ndarray]:
    """Linear interp onto inov z. Levels below zmax stay NaN."""
    src_z = np.asarray(src_z, dtype=np.float64).reshape(-1)
    dst_z = np.asarray(dst_z, dtype=np.float64).reshape(-1)
    T = np.asarray(T, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    n = T.shape[0]
    To = np.full((n, dst_z.size), np.nan)
    So = np.full((n, dst_z.size), np.nan)
    ok_z = (dst_z >= src_z[0]) & (dst_z <= min(float(src_z[-1]), zmax))
    if not ok_z.any():
        return To, So
    zq = dst_z[ok_z]
    for i in range(n):
        ti, si = T[i], S[i]
        m = np.isfinite(ti) & np.isfinite(si) & np.isfinite(src_z)
        if m.sum() < 2:
            continue
        To[i, ok_z] = np.interp(zq, src_z[m], ti[m])
        So[i, ok_z] = np.interp(zq, src_z[m], si[m])
    return To, So


def _rmse_bias(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> dict:
    m = np.asarray(mask, dtype=bool) & np.isfinite(a) & np.isfinite(b)
    if not m.any():
        return {"rmse": None, "bias": None, "n": 0}
    d = a[m] - b[m]
    return {"rmse": float(np.sqrt(np.mean(d * d))), "bias": float(np.mean(d)), "n": int(m.sum())}


def _band_mask(z: np.ndarray, lo: float, hi: float) -> np.ndarray:
    if np.isfinite(hi):
        return (z >= lo) & (z < hi)
    return z >= lo


def interp_obs_to_native(
    src_z: np.ndarray, T: np.ndarray, S: np.ndarray, dst_z: np.ndarray, valid: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    src_z = np.asarray(src_z, dtype=np.float64).reshape(-1)
    dst_z = np.asarray(dst_z, dtype=np.float64).reshape(-1)
    T = np.asarray(T, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    n = T.shape[0]
    To = np.full((n, dst_z.size), np.nan)
    So = np.full((n, dst_z.size), np.nan)
    for i in range(n):
        m = np.isfinite(T[i]) & np.isfinite(S[i]) & np.isfinite(src_z)
        if valid is not None:
            m &= np.asarray(valid[i], dtype=bool)
        if m.sum() < 2:
            continue
        To[i] = np.interp(dst_z, src_z[m], T[i, m], left=np.nan, right=np.nan)
        So[i] = np.interp(dst_z, src_z[m], S[i, m], left=np.nan, right=np.nan)
    return To, So


def _decode_pc_sigma(pca, sigma_pc: np.ndarray) -> np.ndarray:
    """Physical σ from PC σ: sqrt(σ_z² @ V²), matching PCAHeteroPhysLoss.decode_sigma."""
    V2 = np.asarray(pca.components_, dtype=np.float64) ** 2
    ev = np.asarray(pca.explained_variance_, dtype=np.float64)
    if bool(getattr(pca, "whiten", False)):
        scale = np.sqrt(np.maximum(ev, 1e-12))
    else:
        scale = np.ones(V2.shape[0], dtype=np.float64)
    native = np.asarray(sigma_pc, dtype=np.float64) * scale[None, :]
    var = (native ** 2) @ V2
    return np.sqrt(np.maximum(var, 1e-12))


def score_vs_argo(
    pred_T: np.ndarray,
    pred_S: np.ndarray,
    argo_T: np.ndarray,
    argo_S: np.ndarray,
    z: np.ndarray,
    valid_T: np.ndarray,
    valid_S: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    *,
    n2: bool = True,
    sig_T: np.ndarray | None = None,
) -> dict:
    from evalphys.constants import DEPTH_BAND_LABELS, DEPTH_BANDS
    from evalphys.metrics import isotherm_depth, static_stability_violations

    z = np.asarray(z, dtype=np.float64).reshape(-1)
    z_ok = z <= Z_NES_MAX
    mT = valid_T & z_ok[None, :]
    mS = valid_S & z_ok[None, :]
    out: dict = {
        "T": _rmse_bias(pred_T, argo_T, mT),
        "S": _rmse_bias(pred_S, argo_S, mS),
        "T_band": {},
        "S_band": {},
    }
    for label, (lo, hi) in zip(DEPTH_BAND_LABELS, DEPTH_BANDS):
        b = _band_mask(z, lo, hi)
        out["T_band"][label] = _rmse_bias(pred_T, argo_T, mT & b[None, :])
        out["S_band"][label] = _rmse_bias(pred_S, argo_S, mS & b[None, :])
    T_a = np.where(mT, argo_T, np.nan)
    T_p = np.where(mT, pred_T, np.nan)
    S_a = np.where(mS, argo_S, np.nan)
    d26_p, cov_p = isotherm_depth(T_p, z, 26.0)
    d26_a, cov_a = isotherm_depth(T_a, z, 26.0)
    out["D26"] = {**_rmse_bias(d26_p, d26_a, np.isfinite(d26_p) & np.isfinite(d26_a)), "coverage_pred": cov_p, "coverage_argo": cov_a}
    both = mT & mS
    Tp = np.where(both, pred_T, np.nan)
    Sp = np.where(both, pred_S, np.nan)
    Ta = np.where(both, argo_T, np.nan)
    Sa = np.where(both, argo_S, np.nan)
    keep = np.isfinite(Tp).sum(axis=1) >= 8
    if n2:
        if keep.any():
            stab = static_stability_violations(Tp[keep], Sp[keep], z, lat[keep], lon[keep])
            out["n2_level"] = stab.get("violation_rate_level")
            out["n2_profile"] = stab.get("violation_rate_profile")
            out["n2_n"] = int(keep.sum())
        else:
            out["n2_level"] = None
            out["n2_profile"] = None
            out["n2_n"] = 0
    from evalphys.metrics import (
        heave_vs_shape_split,
        max_n2_depth,
        mixed_layer_depth,
        ohc_rmse_by_layer,
        steric_height_cm,
        water_mass_rmse,
    )

    if keep.any():
        lat_k, lon_k = lat[keep], lon[keep]
        mld_p = mixed_layer_depth(Tp[keep], Sp[keep], z, lat_k, lon_k)
        mld_a = mixed_layer_depth(Ta[keep], Sa[keep], z, lat_k, lon_k)
        out["MLD"] = _rmse_bias(mld_p, mld_a, np.isfinite(mld_p) & np.isfinite(mld_a))
        d20_p, cov20_p = isotherm_depth(T_p, z, 20.0)
        d20_a, cov20_a = isotherm_depth(T_a, z, 20.0)
        out["D20"] = {
            **_rmse_bias(d20_p, d20_a, np.isfinite(d20_p) & np.isfinite(d20_a)),
            "coverage_pred": cov20_p,
            "coverage_argo": cov20_a,
        }
        zn2_p = max_n2_depth(Tp[keep], Sp[keep], z, lat_k, lon_k)
        zn2_a = max_n2_depth(Ta[keep], Sa[keep], z, lat_k, lon_k)
        out["max_n2_depth"] = _rmse_bias(zn2_p, zn2_a, np.isfinite(zn2_p) & np.isfinite(zn2_a))
        out["heave_vs_shape"] = heave_vs_shape_split(T_p, T_a, z, d26_p, d26_a)
        out["ohc"] = ohc_rmse_by_layer(T_p, T_a, z, valid=mT)
        out["water_mass"] = water_mass_rmse(Tp[keep], Sp[keep], Ta[keep], Sa[keep], z, lat_k, lon_k)
        eta_p = steric_height_cm(Tp[keep], Sp[keep], z, lat_k, lon_k)
        eta_a = steric_height_cm(Ta[keep], Sa[keep], z, lat_k, lon_k)
        out["steric_cm"] = _rmse_bias(eta_p, eta_a, np.isfinite(eta_p) & np.isfinite(eta_a))
    else:
        out["MLD"] = {"rmse": None, "bias": None, "n": 0}
        out["D20"] = {"rmse": None, "bias": None, "n": 0, "coverage_pred": 0.0, "coverage_argo": 0.0}
        out["max_n2_depth"] = {"rmse": None, "bias": None, "n": 0}
        out["heave_vs_shape"] = {}
        out["ohc"] = {}
        out["water_mass"] = {}
        out["steric_cm"] = {"rmse": None, "bias": None, "n": 0}
    if sig_T is not None:
        from evalphys.metrics import spatial_sigma_consistency

        sT = np.asarray(sig_T, dtype=np.float64)
        out["sigma_T"] = spatial_sigma_consistency(
            T_p, np.where(mT, sT, np.nan), T_a, lon, lat, z=z, valid=mT
        )
    return out


def bin1deg(lon: np.ndarray, lat: np.ndarray, val: np.ndarray) -> dict:
    """Mean of val in 1° cells. Returns lon0, lat0, mean, count grids."""
    lon = np.asarray(lon, dtype=np.float64).reshape(-1)
    lat = np.asarray(lat, dtype=np.float64).reshape(-1)
    val = np.asarray(val, dtype=np.float64).reshape(-1)
    m = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(val)
    lon, lat, val = lon[m], lat[m], val[m]
    if lon.size == 0:
        return {"lon0": np.array([]), "lat0": np.array([]), "mean": np.zeros((0, 0)), "n": np.zeros((0, 0), dtype=int)}
    i0, i1 = int(np.floor(lon.min())), int(np.floor(lon.max()))
    j0, j1 = int(np.floor(lat.min())), int(np.floor(lat.max()))
    lon0 = np.arange(i0, i1 + 1, dtype=np.float64)
    lat0 = np.arange(j0, j1 + 1, dtype=np.float64)
    ni, nj = lon0.size, lat0.size
    acc = np.zeros((nj, ni), dtype=np.float64)
    cnt = np.zeros((nj, ni), dtype=np.int32)
    ii = np.floor(lon).astype(int) - i0
    jj = np.floor(lat).astype(int) - j0
    for k in range(lon.size):
        acc[jj[k], ii[k]] += val[k]
        cnt[jj[k], ii[k]] += 1
    mean = np.full_like(acc, np.nan)
    ok = cnt > 0
    mean[ok] = acc[ok] / cnt[ok]
    return {"lon0": lon0, "lat0": lat0, "mean": mean, "n": cnt}


def _as_str(arr) -> np.ndarray:
    return np.array([str(x) for x in np.asarray(arr)])


def load_xb(path: Path) -> dict:
    from netCDF4 import Dataset

    ds = Dataset(str(path))
    z = np.array(ds["z"][:], dtype=np.float64)
    out = {
        "z": z,
        "lon": np.array(ds["lon"][:], dtype=np.float64),
        "lat": np.array(ds["lat"][:], dtype=np.float64),
        "T_argo": np.array(ds["T_argo"][:], dtype=np.float64),
        "S_argo": np.array(ds["S_argo"][:], dtype=np.float64),
        "T_xb": np.array(ds["T_xb"][:], dtype=np.float64),
        "S_xb": np.array(ds["S_xb"][:], dtype=np.float64),
        "T_xa": np.array(ds["T_xa"][:], dtype=np.float64),
        "S_xa": np.array(ds["S_xa"][:], dtype=np.float64),
        "valid_T": np.array(ds["valid_T"][:]).astype(bool),
        "valid_S": np.array(ds["valid_S"][:]).astype(bool),
        "time_iso": _as_str(ds["time_iso"][:]),
        "analysis_date": _as_str(ds["analysis_date"][:]),
        "dataset": _as_str(ds["dataset"][:]),
    }
    ds.close()
    out["times"] = [datetime.strptime(s[:10], "%Y-%m-%d") for s in out["time_iso"]]
    out["ood"] = (
        (out["lat"] < TRAIN_LAT[0])
        | (out["lat"] > TRAIN_LAT[1])
        | (out["lon"] < TRAIN_LON[0])
        | (out["lon"] > TRAIN_LON[1])
    )
    return out


def slice_xb(xb: dict, nkeep: int) -> dict:
    out = dict(xb)
    n = min(nkeep, xb["lon"].size)
    for k, v in xb.items():
        if k == "z":
            continue
        if isinstance(v, np.ndarray) and v.shape[0] == xb["lon"].size:
            out[k] = v[:n]
        elif isinstance(v, list) and len(v) == xb["lon"].size:
            out[k] = v[:n]
    return out


def load_preds_nc(path: Path) -> dict:
    from netCDF4 import Dataset

    skip = {"T_argo", "S_argo", "T_xb", "S_xb", "T_xa", "S_xa"}
    ds = Dataset(str(path))
    found = [v[2:] for v in ds.variables if v.startswith("T_") and v not in skip]
    order = [m["name"] for m in MODELS if m["name"] in found]
    order += [n for n in found if n not in set(order)]
    preds = {}
    for name in order:
        pack = {
            "T": np.array(ds[f"T_{name}"][:], dtype=np.float64),
            "S": np.array(ds[f"S_{name}"][:], dtype=np.float64),
        }
        if f"sigT_{name}" in ds.variables:
            pack["sigT"] = np.array(ds[f"sigT_{name}"][:], dtype=np.float64)
            pack["sigS"] = np.array(ds[f"sigS_{name}"][:], dtype=np.float64)
        preds[name] = pack
    ds.close()
    return preds


def _api_on_path(api: Path) -> None:
    s = str(api)
    if s not in sys.path:
        sys.path.insert(0, s)
    try:
        from services.common.v2_spec import ensure_geof_on_path

        ensure_geof_on_path()
    except Exception:
        pass


def load_sat(times, lat, lon, cache_path: Path | None):
    if cache_path is not None and cache_path.is_file():
        z = np.load(cache_path)
        print(f"sat cache {cache_path}", flush=True)
        return z["sss"], z["sst"], z["ssh"]
    from services.accessor.sat import load_satellite_data

    sss, sst, ssh = load_satellite_data(list(times), lat, lon)
    sss = np.asarray(sss, dtype=np.float64).reshape(-1)
    sst = np.asarray(sst, dtype=np.float64).reshape(-1)
    ssh = np.asarray(ssh, dtype=np.float64).reshape(-1)
    if cache_path is not None:
        np.savez(cache_path, sss=sss, sst=sst, ssh=ssh)
    return sss, sst, ssh


def load_ops(times, lat, lon, cache_path: Path | None):
    if cache_path is not None and cache_path.is_file():
        z = np.load(cache_path)
        print(f"ops cache {cache_path}", flush=True)
        return z["ops"]
    from services.accessor.v2_inputs import sample_ops_or_503

    ops = sample_ops_or_503(list(times), lat, lon, strict=False)
    if cache_path is not None:
        np.savez(cache_path, ops=ops)
    return ops


def _resolve_routing_spec(path) -> Path | None:
    if path is None:
        return None
    p = Path(str(path))
    candidates = [p, _ROOT / p, _REPO / "reports" / p.name, _ROOT.parent / "reports" / p.name]
    for c in candidates:
        if c.is_file():
            return c.resolve()
    return p


class LocalPCA:
    """PCA-inverse cell not in the DA serve registry (A_CRPS_z32 family)."""

    def __init__(self, ckpt: Path, cache_path: Path):
        import torch
        import model.model as module_arch
        from sklearn.base import InconsistentVersionWarning

        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        ckcfg = state.get("config")
        if not isinstance(ckcfg, dict):
            ckcfg = getattr(ckcfg, "config", None) or getattr(ckcfg, "_config", None) or {}
        arch = dict(ckcfg["arch"])
        args = dict(arch["args"])
        if args.get("routing_spec"):
            args["routing_spec"] = str(_resolve_routing_spec(args["routing_spec"]))
        self.model = getattr(module_arch, arch["type"])(**args)
        self.model.load_state_dict(state["state_dict"])
        self.model.eval()
        self.output_dim = int(sum(ckcfg["outputs"].values()))
        self.n_t = int(ckcfg["outputs"]["temperature"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InconsistentVersionWarning)
            with cache_path.open("rb") as f:
                cache = pickle.load(f)
        self.pca = cache["pca_models"]
        self.depth = np.asarray(cache["PRES"], dtype=np.float64).reshape(-1)
        self.input_dim = int(args["input_dim"])
        self.ckpt = ckpt
        self.cache_path = cache_path
        self.probabilistic = bool(args.get("probabilistic"))

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
        import torch

        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(f"expected (N, {self.input_dim}), got {x.shape}")
        with torch.no_grad():
            out = self.model(torch.as_tensor(x)).cpu().numpy()
        mu = out[:, : self.output_dim] if out.shape[-1] == 2 * self.output_dim else out
        T = self.pca["temperature"].inverse_transform(mu[:, : self.n_t])
        S = self.pca["salinity"].inverse_transform(mu[:, self.n_t : self.n_t + self.n_t])
        sigT = sigS = None
        if out.shape[-1] == 2 * self.output_dim:
            from model.prob_head import softplus_sigma

            raw = torch.as_tensor(out[:, self.output_dim :])
            sz = softplus_sigma(raw, sigma_min=1e-3).cpu().numpy()
            sigT = _decode_pc_sigma(self.pca["temperature"], sz[:, : self.n_t])
            sigS = _decode_pc_sigma(self.pca["salinity"], sz[:, self.n_t : self.n_t + self.n_t])
        return (
            np.asarray(T, dtype=np.float64),
            np.asarray(S, dtype=np.float64),
            None if sigT is None else np.asarray(sigT, dtype=np.float64),
            None if sigS is None else np.asarray(sigS, dtype=np.float64),
        )


class Ozavala:
    """Legacy SAT cell: TorchScript + 15-PC sklearn in nespreso_api/models."""

    def __init__(self, ckpt: Path, pca_path: Path):
        import torch
        from sklearn.base import InconsistentVersionWarning

        self.model = torch.jit.load(str(ckpt), map_location="cpu")
        self.model.eval()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InconsistentVersionWarning)
            with pca_path.open("rb") as f:
                stats = pickle.load(f)
        self.pca_temp = stats["pca_temp"]
        self.pca_sal = stats["pca_sal"]
        self.depth = np.arange(0, 1801, dtype=np.float64)
        self.input_dim = 9

    def predict(self, x: np.ndarray):
        import torch

        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(f"Ozavala expected (N, {self.input_dim}), got {x.shape}")
        with torch.no_grad():
            pcs = self.model(torch.as_tensor(x)).cpu().numpy()
        T = self.pca_temp.inverse_transform(pcs[:, :15])
        S = self.pca_sal.inverse_transform(pcs[:, 15:30])
        return np.asarray(T, dtype=np.float64), np.asarray(S, dtype=np.float64), None, None


def predict_model(spec: dict, x: np.ndarray, idx: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Return native-z T,S,(sigT,sigS) of shape (n_all, n_z) with NaNs off idx."""
    empty_sig = None
    if spec["kind"] == "v2_cell":
        from services.kernel.v2_cells import predict_profiles

        dec = predict_profiles(spec["v2_name"], x, seed=42)
        src_z = np.asarray(dec.depth, dtype=np.float64).reshape(-1)
        T = np.full((n, src_z.size), np.nan)
        S = np.full((n, src_z.size), np.nan)
        T[idx] = np.asarray(dec.temperature, dtype=np.float64).T
        S[idx] = np.asarray(dec.salinity, dtype=np.float64).T
        return T, S, src_z, empty_sig, empty_sig
    if spec["kind"] == "ozavala":
        cell = Ozavala(spec["ckpt"], spec["cache"])
        Tp, Sp, sigT, sigS = cell.predict(x)
    else:
        cell = LocalPCA(spec["ckpt"], spec["cache"])
        Tp, Sp, sigT, sigS = cell.predict(x)
    src_z = cell.depth
    T = np.full((n, src_z.size), np.nan)
    S = np.full((n, src_z.size), np.nan)
    T[idx] = Tp
    S[idx] = Sp
    sT = sS = None
    if sigT is not None:
        sT = np.full((n, src_z.size), np.nan)
        sS = np.full((n, src_z.size), np.nan)
        sT[idx] = sigT
        sS[idx] = sigS
    return T, S, src_z, sT, sS


def _era_masks(analysis_date: np.ndarray) -> dict[str, np.ndarray]:
    y = np.array([s[:4] for s in analysis_date])
    return {
        "all": np.ones(y.size, dtype=bool),
        "2024": y == "2024",
        "2025": y == "2025",
    }


def _slice_masks(xb: dict) -> dict[str, np.ndarray]:
    from evalphys.constants import LC_LAT_RANGE, LC_LON_RANGE

    out = _era_masks(xb["analysis_date"])
    out["in_bbox"] = ~xb["ood"]
    out["ood"] = np.asarray(xb["ood"], dtype=bool)
    out["lc"] = (
        (xb["lat"] >= LC_LAT_RANGE[0])
        & (xb["lat"] <= LC_LAT_RANGE[1])
        & (xb["lon"] >= LC_LON_RANGE[0])
        & (xb["lon"] <= LC_LON_RANGE[1])
    )
    return out


def _profiles_to_pcs(T: np.ndarray, S: np.ndarray, pca) -> np.ndarray:
    nt = int(pca["temperature"].n_components_)
    n = T.shape[0]
    out = np.full((n, 2 * nt), np.nan, dtype=np.float32)
    mt, ms = pca["temperature"].mean_, pca["salinity"].mean_
    for i in range(n):
        ti, si = T[i], S[i]
        if np.isfinite(ti).sum() < 8 or np.isfinite(si).sum() < 8:
            continue
        tt = np.where(np.isfinite(ti), ti, mt)
        ss = np.where(np.isfinite(si), si, ms)
        out[i, :nt] = pca["temperature"].transform(tt.reshape(1, -1))[0]
        out[i, nt:] = pca["salinity"].transform(ss.reshape(1, -1))[0]
    return out


def build_xb_pairs(xb: dict):
    from base.split_utils import dates_to_juld
    from preproc.pair_table import nearest_eval_pairs

    juld = dates_to_juld([t.strftime("%Y-%m-%d") for t in xb["times"]], dataset_tag="argo_v2")
    pairs = nearest_eval_pairs(xb["lat"], xb["lon"], juld)
    for row in pairs:
        if float(row["dt_days"]) <= 0:
            raise RuntimeError("pair source is not strictly earlier")
    return pairs, juld


def residual_stats(pred, argo, valid, z) -> dict:
    err = np.where(valid & (z[None, :] <= Z_NES_MAX), pred - argo, np.nan)
    return {
        "mean": np.nanmean(err, axis=0).tolist(),
        "q25": np.nanpercentile(err, 25, axis=0).tolist(),
        "q75": np.nanpercentile(err, 75, axis=0).tolist(),
    }


def plot_residuals(xb, preds: dict, out_dir: Path, var: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    z = xb["z"]
    argo = xb[f"{var}_argo"]
    valid = xb[f"valid_{var}"]
    eras = _era_masks(xb["analysis_date"])
    long_name = "Temperature" if var == "T" else "Salinity"
    unit = "°C" if var == "T" else ""
    era_title = {"all": "All casts", "2024": "2024", "2025": "2025"}

    fig = plt.figure(figsize=(12.8, 6.6))
    gs = GridSpec(
        2,
        3,
        height_ratios=[1.0, 0.20],
        hspace=0.12,
        wspace=0.20,
        top=0.88,
        bottom=0.04,
        left=0.07,
        right=0.99,
        figure=fig,
    )
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    with np.errstate(all="ignore"):
        for ax, (era, mask) in zip(axes, eras.items()):
            series = [("xb", xb[f"{var}_{'xb'}"]), ("xa", xb[f"{var}_{'xa'}"])]
            series += [(name, preds[name][var]) for name in preds]
            for name, arr in series:
                err = np.where(valid[mask] & (z[None, :] <= Z_NES_MAX), arr[mask] - argo[mask], np.nan)
                mean = np.nanmean(err, axis=0)
                lo = np.nanpercentile(err, 25, axis=0)
                hi = np.nanpercentile(err, 75, axis=0)
                ok = np.isfinite(mean)
                c = COLORS.get(name, "#333333")
                ax.plot(mean[ok], z[ok], color=c, lw=1.6, label=name, zorder=4)
                ax.fill_betweenx(z[ok], lo[ok], hi[ok], color=c, alpha=0.12, linewidth=0, zorder=1)
            ax.axvline(0.0, color="#666666", lw=0.9, zorder=3)
            ax.set_title(f"{era_title[era]}   n = {int(mask.sum())}", fontsize=11)
            ax.set_xlabel(f"{var} residual (pred - Argo) {unit}".rstrip())
            ax.set_ylim(Z_NES_MAX, 0)
            ax.grid(True, which="major", linestyle="--", color="0.65", linewidth=0.7, zorder=2)
            ax.tick_params(labelsize=8)
    axes[0].set_ylabel("depth z (m)")
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)
    handles, labels = axes[0].get_legend_handles_labels()
    leg_ax = fig.add_subplot(gs[1, :])
    leg_ax.axis("off")
    leg_ax.legend(
        handles,
        labels,
        loc="center",
        ncol=min(len(labels), 7),
        frameon=False,
        fontsize=9,
        handlelength=2.2,
        columnspacing=1.4,
    )
    fig.suptitle(f"{long_name} residual vs Argo", fontsize=13, y=0.98)
    fig.savefig(out_dir / f"residual_{var}.png", dpi=140)
    plt.close(fig)


def plot_maps(xb, preds: dict, sss, sst, ssh, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = xb["z"]
    iz = int(np.where(np.isclose(z, Z_NEAR_M))[0][0])
    vT = xb["valid_T"][:, iz]
    vS = xb["valid_S"][:, iz]
    lon, lat = xb["lon"], xb["lat"]

    def _grid(val, mask):
        return bin1deg(lon[mask], lat[mask], val[mask])

    def _draw(ax, packet, cmap, vmin, vmax, title, colorbar=False):
        if packet["lon0"].size == 0:
            ax.set_title(title + " (empty)")
            return None
        lon_e = np.append(packet["lon0"], packet["lon0"][-1] + 1)
        lat_e = np.append(packet["lat0"], packet["lat0"][-1] + 1)
        m = np.ma.masked_invalid(packet["mean"])
        pcm = ax.pcolormesh(lon_e, lat_e, m, cmap=cmap, vmin=vmin, vmax=vmax, shading="flat")
        ax.set_title(title, fontsize=9)
        ax.set_aspect("equal", adjustable="box")
        if colorbar:
            ax.figure.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
        return pcm

    def _residual_maps(fields, truth, valid, vmin, vmax, unit, outfile, suptitle):
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        n_col = 4
        n_row = int(np.ceil((len(names) + 1) / n_col))
        fig, axes = plt.subplots(n_row, n_col, figsize=(14.2, 3.3 * n_row), squeeze=False)
        last = None
        for i, name in enumerate(names):
            ax = axes[i // n_col][i % n_col]
            d = fields[name][:, iz] - truth[:, iz]
            last = _draw(ax, _grid(d, valid), "RdBu_r", vmin, vmax, name, colorbar=False)
        for j in range(len(names), n_row * n_col - 1):
            axes[j // n_col][j % n_col].axis("off")
        holder = axes[-1][-1]
        holder.set_xticks([])
        holder.set_yticks([])
        for spine in holder.spines.values():
            spine.set_visible(False)
        sm = last if last is not None else ScalarMappable(
            norm=Normalize(vmin=vmin, vmax=vmax), cmap="RdBu_r"
        )
        if last is None:
            sm.set_array([])
        cb_ax = inset_axes(holder, width="22%", height="78%", loc="center")
        cb = fig.colorbar(sm, cax=cb_ax)
        cb.set_label(unit, fontsize=10)
        fig.suptitle(suptitle)
        fig.subplots_adjust(top=0.90, hspace=0.32, wspace=0.18)
        fig.savefig(out_dir / outfile, dpi=140)
        plt.close(fig)

    names = ["xb", "xa"] + list(preds.keys())
    T_fields = {"xb": xb["T_xb"], "xa": xb["T_xa"], **{k: v["T"] for k, v in preds.items()}}
    S_fields = {"xb": xb["S_xb"], "xa": xb["S_xa"], **{k: v["S"] for k, v in preds.items()}}
    _residual_maps(T_fields, xb["T_argo"], vT, -2.0, 2.0, "T residual (°C)", "map_T10.png", "1° mean temperature residual at 10 m")
    _residual_maps(S_fields, xb["S_argo"], vS, -0.4, 0.4, "S residual", "map_S10.png", "1° mean salinity residual at 10 m")

    fig, axes = plt.subplots(1, 4, figsize=(14.5, 3.4))
    sst_c = np.asarray(sst, dtype=np.float64) - 273.15
    _draw(axes[0], bin1deg(lon, lat, sst_c), "inferno", 20, 32, "sat SST (°C)", colorbar=True)
    _draw(axes[1], bin1deg(lon, lat, sss), "viridis", 32, 38, "sat SSS", colorbar=True)
    _draw(axes[2], bin1deg(lon, lat, ssh), "coolwarm", -0.4, 0.4, "sat SSH (m)", colorbar=True)
    pkt_n = bin1deg(lon, lat, np.where(vT, 1.0, np.nan))
    _draw(axes[3], {"lon0": pkt_n["lon0"], "lat0": pkt_n["lat0"], "mean": pkt_n["n"].astype(float), "n": pkt_n["n"]}, "gray", 0, None, "n casts / cell", colorbar=True)
    fig.suptitle("1° satellite inputs at Argo sites (not NeSPReSO outputs)")
    fig.tight_layout()
    fig.savefig(out_dir / "map_sat.png", dpi=140)
    plt.close(fig)


def write_nc(path: Path, xb, preds: dict, sss, sst, ssh) -> None:
    from netCDF4 import Dataset

    ds = Dataset(str(path), "w")
    ds.createDimension("profile", xb["lon"].size)
    ds.createDimension("z", xb["z"].size)
    ds.createDimension("iso_len", 10)
    for name, arr, dims in (
        ("z", xb["z"], ("z",)),
        ("lon", xb["lon"], ("profile",)),
        ("lat", xb["lat"], ("profile",)),
        ("T_argo", xb["T_argo"], ("profile", "z")),
        ("S_argo", xb["S_argo"], ("profile", "z")),
        ("T_xb", xb["T_xb"], ("profile", "z")),
        ("S_xb", xb["S_xb"], ("profile", "z")),
        ("T_xa", xb["T_xa"], ("profile", "z")),
        ("S_xa", xb["S_xa"], ("profile", "z")),
        ("sss_sat", sss, ("profile",)),
        ("sst_sat_K", sst, ("profile",)),
        ("ssh_sat", ssh, ("profile",)),
    ):
        v = ds.createVariable(name, "f4", dims, fill_value=np.nan)
        v[:] = arr
    vt = ds.createVariable("valid_T", "i1", ("profile", "z"))
    vt[:] = xb["valid_T"].astype(np.int8)
    vs = ds.createVariable("valid_S", "i1", ("profile", "z"))
    vs[:] = xb["valid_S"].astype(np.int8)
    for name, pack in preds.items():
        t = ds.createVariable(f"T_{name}", "f4", ("profile", "z"), fill_value=np.nan)
        t[:] = pack["T"]
        s = ds.createVariable(f"S_{name}", "f4", ("profile", "z"), fill_value=np.nan)
        s[:] = pack["S"]
        if pack.get("sigT") is not None:
            st = ds.createVariable(f"sigT_{name}", "f4", ("profile", "z"), fill_value=np.nan)
            st[:] = pack["sigT"]
            ss = ds.createVariable(f"sigS_{name}", "f4", ("profile", "z"), fill_value=np.nan)
            ss[:] = pack["sigS"]
    if "pair_source_j" in xb:
        pj = ds.createVariable("pair_source_j", "i4", ("profile",), fill_value=-1)
        pj[:] = xb["pair_source_j"]
    ds.setncattr("title", "NeSPReSO vs TSIS Argo/xb/xa on inov z")
    ds.setncattr("comment", "NeSPReSO interpolated onto z; score valid==1 and z<=1800")
    ds.close()


def _fmt(x, nd=3):
    return "—" if x is None else f"{x:.{nd}f}"


def _nested(d, *keys):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _reading(stats: dict) -> str:
    all_ = stats["slices"]["all"]
    y25 = stats["slices"].get("2025", all_)
    map_names = [m for m in stats["models"] if stats["track"].get(m, "map") == "map"]
    t50 = {m: _nested(all_["models"][m], "T_band", "50-200", "rmse") for m in map_names}
    xb50 = _nested(all_["xb"], "T_band", "50-200", "rmse")
    finite = {k: v for k, v in t50.items() if v is not None}
    best = min(finite, key=finite.get) if finite else None
    parts = [
        f"Independent-era file from `{ZIP_DEFAULT.name}` ({stats['n']} casts, training cache ends 2022-02-27). "
        f"Do not put chrono-test RMSE next to these numbers.",
        f"Closest map cell on 50–200 m T is {best} ({finite.get(best):.3f} vs xb {xb50:.3f})."
        if best and xb50 is not None
        else "",
        f"2025 n={y25['n']}. Ingest gate: beat xb on 50–200 m T, D26, and OHC 0–300 without worse N².",
    ]
    return " ".join(p for p in parts if p)


def write_md(path: Path, stats: dict) -> None:
    map_names = [m for m in stats["models"] if stats["track"].get(m, "map") == "map"]
    pair_names = [m for m in stats["models"] if stats["track"].get(m) == "pair"]
    heads = ["xb", "xa"] + map_names
    all_ = stats["slices"]["all"]

    def _row_get(name, *keys):
        block = all_["xb"] if name == "xb" else all_["xa"] if name == "xa" else all_["models"][name]
        return _nested(block, *keys)

    lines = [
        "# NeSPReSO vs TSIS Argo / xb / xa",
        "",
        f"Source `{ZIP_DEFAULT.name}` → `{stats['nc']}`. {stats['n']} casts, {stats['n_cycles']} cycles, "
        f"z ≤ {int(Z_NES_MAX)} m, `valid==1`. Training cache ends 2022-02-27; these dates are 2024–2025.",
        "",
        f"Spatial OOD vs training bbox: {stats['n_ood']} / {stats['n']} "
        f"({100.0 * stats['n_ood'] / stats['n']:.1f}%). Mostly east of 82.67°W.",
        "",
        "Satellite SST/SSS/SSH from GOFFISH (MUR / SMAP / AVISO) at the Argo day, not `analysis_date`.",
        "",
        "## Headline (all casts, map track)",
        "",
        "| model | T 50-200 | D26 m | OHC 0-300 GJ/m² | MLD m | σ₀ | N² level | beats xb |",
        "|-------|---------:|------:|----------------:|------:|----:|---------:|----------|",
    ]
    xb50 = _row_get("xb", "T_band", "50-200", "rmse")
    xb_d26 = _row_get("xb", "D26", "rmse")
    xb_ohc = _row_get("xb", "ohc", "0-300", "rmse")
    for name in heads:
        t50 = _row_get(name, "T_band", "50-200", "rmse")
        d26 = _row_get(name, "D26", "rmse")
        ohc = _row_get(name, "ohc", "0-300", "rmse")
        mld = _row_get(name, "MLD", "rmse")
        s0 = _row_get(name, "water_mass", "sigma0", "rmse")
        n2 = _row_get(name, "n2_level")
        beats = "—"
        if name not in ("xb", "xa") and None not in (t50, d26, ohc, xb50, xb_d26, xb_ohc):
            beats = "yes" if (t50 < xb50 and d26 < xb_d26 and ohc < xb_ohc) else "no"
        lines.append(
            f"| {name} | {_fmt(t50)} | {_fmt(d26)} | {_fmt(ohc)} | {_fmt(mld)} | {_fmt(s0)} | {_fmt(n2, 4)} | {beats} |"
        )
    lines += ["", "## T/S RMSE by era", "", "| era | n | field | " + " | ".join(heads) + " |"]
    lines.append("|-----|--:|-------|" + "|".join(["---:" for _ in heads]) + "|")
    for era, block in stats["slices"].items():
        if era not in ("all", "2024", "2025"):
            continue
        n = block["n"]
        for field in ("T", "S"):
            cells = []
            for name in heads:
                row = block["xb"] if name == "xb" else block["xa"] if name == "xa" else block["models"].get(name)
                cells.append(_fmt(_nested(row or {}, field, "rmse")))
            lines.append(f"| {era} | {n} | {field} RMSE | " + " | ".join(cells) + " |")
        for field in ("T", "S"):
            cells = []
            for name in heads:
                row = block["xb"] if name == "xb" else block["xa"] if name == "xa" else block["models"].get(name)
                cells.append(_fmt(_nested(row or {}, field, "bias")))
            lines.append(f"| {era} | {n} | {field} bias | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Bias vs Argo (pred − Argo, all, map track)",
        "",
        "Negative T is too cold. Negative D26 is too shallow. Negative OHC is too little heat. Negative MLD is too shallow.",
        "",
        "| model | T | T 50-200 | S | D26 m | OHC 0-300 | MLD m | σ₀ | spice |",
        "|-------|--:|---------:|--:|------:|----------:|------:|----:|------:|",
    ]
    for name in heads:
        lines.append(
            f"| {name} | {_fmt(_row_get(name, 'T', 'bias'))} | {_fmt(_row_get(name, 'T_band', '50-200', 'bias'))} | "
            f"{_fmt(_row_get(name, 'S', 'bias'))} | {_fmt(_row_get(name, 'D26', 'bias'))} | "
            f"{_fmt(_row_get(name, 'ohc', '0-300', 'bias'))} | {_fmt(_row_get(name, 'MLD', 'bias'))} | "
            f"{_fmt(_row_get(name, 'water_mass', 'sigma0', 'bias'))} | "
            f"{_fmt(_row_get(name, 'water_mass', 'spice', 'bias'))} |"
        )
    lines += ["", "## Temperature RMSE by band (all)", "", "| band | " + " | ".join(heads) + " |"]
    lines.append("|------|" + "|".join(["---:" for _ in heads]) + "|")
    for band in all_["xb"]["T_band"]:
        cells = [_fmt(_row_get(name, "T_band", band, "rmse")) for name in heads]
        lines.append(f"| {band} | " + " | ".join(cells) + " |")
    lines += ["", "## Temperature bias by band (pred − Argo, all)", "", "| band | " + " | ".join(heads) + " |"]
    lines.append("|------|" + "|".join(["---:" for _ in heads]) + "|")
    for band in all_["xb"]["T_band"]:
        cells = [_fmt(_row_get(name, "T_band", band, "bias")) for name in heads]
        lines.append(f"| {band} | " + " | ".join(cells) + " |")
    lines += ["", "## Calibration (physical T σ, all)", "", "| model | ENCE(T) | CRPS(T) | cov 68% | cell Spearman | 50-200 ENCE |"]
    lines.append("|-------|--------:|--------:|--------:|--------------:|------------:|")
    for name in map_names:
        sig = _row_get(name, "sigma_T") or {}
        pooled = sig.get("pooled") or {}
        b200 = (sig.get("by_depth_band") or {}).get("50-200") or {}
        if pooled.get("n"):
            lines.append(
                f"| {name} | {_fmt(pooled.get('ence'))} | {_fmt(pooled.get('crps_mean'))} | "
                f"{_fmt(pooled.get('coverage_68'))} | {_fmt(sig.get('spearman_cells_rmse_vs_sigma'))} | "
                f"{_fmt(b200.get('ence'))} |"
            )
    if pair_names and "pair" in stats["slices"]:
        pb = stats["slices"]["pair"]
        lines += [
            "",
            f"## Pair track (legal earlier neighbor in this file, n={pb['n']})",
            "",
            "| model | T RMSE | T bias | T 50-200 | D26 | D26 bias | OHC 0-300 | OHC bias | MLD | MLD bias |",
            "|-------|-------:|-------:|---------:|----:|---------:|----------:|---------:|----:|---------:|",
        ]
        for name in ["xb", "xa"] + pair_names:
            row = pb["xb"] if name == "xb" else pb["xa"] if name == "xa" else pb["models"].get(name, {})
            lines.append(
                f"| {name} | {_fmt(_nested(row, 'T', 'rmse'))} | {_fmt(_nested(row, 'T', 'bias'))} | "
                f"{_fmt(_nested(row, 'T_band', '50-200', 'rmse'))} | {_fmt(_nested(row, 'D26', 'rmse'))} | "
                f"{_fmt(_nested(row, 'D26', 'bias'))} | {_fmt(_nested(row, 'ohc', '0-300', 'rmse'))} | "
                f"{_fmt(_nested(row, 'ohc', '0-300', 'bias'))} | {_fmt(_nested(row, 'MLD', 'rmse'))} | "
                f"{_fmt(_nested(row, 'MLD', 'bias'))} |"
            )
    lines += [
        "",
        "## OOD / Loop Current (all, map T 50-200)",
        "",
        "| slice | n | xb | " + " | ".join(map_names) + " |",
        "|-------|--:|---:|" + "|".join(["---:" for _ in map_names]) + "|",
    ]
    for key in ("in_bbox", "ood", "lc"):
        if key not in stats["slices"]:
            continue
        block = stats["slices"][key]
        cells = [_fmt(_nested(block["xb"], "T_band", "50-200", "rmse"))]
        for m in map_names:
            cells.append(_fmt(_nested(block["models"].get(m, {}), "T_band", "50-200", "rmse")))
        lines.append(f"| {key} | {block['n']} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Files",
        "",
        "- `profiles_nes.nc` — T/S (and σ when present) on inov z",
        "- `stats.json`",
        "- `residual_T.png` / `residual_S.png`",
        "- `map_T10.png` / `map_S10.png` / `map_sat.png` / `map_ence.png`",
        "",
        "## Reading",
        "",
        _reading(stats),
        "",
        "CRPS-head σ is not TSIS R. Per-band ENCE still has to clear 0.20 before anyone writes it into `err`.",
        "",
    ]
    path.write_text("\n".join(lines))


A1_BANDS = (("0-50", 0.0, 50.0), ("50-200", 50.0, 200.0), ("200-800", 200.0, 800.0))
A1_SLICE_NAMES = ("in_bbox", "ood", "lc")
A1_ERA_NAMES = ("all", "2024", "2025")
DA_READINESS_SKIP_TRACK = frozenset({"pair"})


def _a1_region_masks(xb: dict) -> dict[str, np.ndarray]:
    from evalphys.constants import LC_LAT_RANGE, LC_LON_RANGE

    return {
        "in_bbox": ~np.asarray(xb["ood"], dtype=bool),
        "ood": np.asarray(xb["ood"], dtype=bool),
        "lc": (
            (xb["lat"] >= LC_LAT_RANGE[0])
            & (xb["lat"] <= LC_LAT_RANGE[1])
            & (xb["lon"] >= LC_LON_RANGE[0])
            & (xb["lon"] <= LC_LON_RANGE[1])
        ),
    }


def run_da_readiness_a1(
    xb: dict,
    preds: dict,
    out_dir: Path,
    *,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    from evalphys.metrics import bootstrap_blend_over_casts

    da_dir = out_dir / "da_readiness"
    da_dir.mkdir(parents=True, exist_ok=True)
    z = xb["z"]
    valid = xb["valid_T"]
    argo = xb["T_argo"]
    e_xb = xb["T_xb"] - argo
    eras = _era_masks(xb["analysis_date"])
    regions = _a1_region_masks(xb)
    map_models = [m["name"] for m in MODELS if m.get("track", "map") not in DA_READINESS_SKIP_TRACK]
    names = [n for n in map_models if n in preds]
    rows = []
    for name in names:
        e_nes = preds[name]["T"] - argo
        for band_name, lo, hi in A1_BANDS:
            for slice_name in A1_SLICE_NAMES:
                for era_name in A1_ERA_NAMES:
                    mask = regions[slice_name] & eras[era_name]
                    if not mask.any():
                        continue
                    cell = bootstrap_blend_over_casts(
                        e_nes[mask],
                        e_xb[mask],
                        valid[mask],
                        z,
                        lo,
                        hi,
                        n_boot=n_boot,
                        seed=seed,
                        pool="levels",
                    )
                    rows.append(
                        {
                            "model": name,
                            "band": band_name,
                            "slice": slice_name,
                            "era": era_name,
                            "n": int(cell["n"]),
                            "rho": cell["rho"],
                            "w_star": cell["w_star"],
                            "rmse_xb": cell["rmse_xb"],
                            "rmse_nes": cell["rmse_nes"],
                            "rmse_blend": cell["rmse_blend"],
                            "ci_lo": cell["ci_lo"],
                            "ci_hi": cell["ci_hi"],
                        }
                    )
    csv_path = da_dir / "blend.csv"
    fields = [
        "model",
        "band",
        "slice",
        "era",
        "n",
        "rho",
        "w_star",
        "rmse_xb",
        "rmse_nes",
        "rmse_blend",
        "ci_lo",
        "ci_hi",
    ]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row[k] for k in fields})
    therm = [
        r
        for r in rows
        if r["band"] == "50-200"
        and np.isfinite(r["w_star"])
        and r["w_star"] > 0.25
        and np.isfinite(r["ci_hi"])
        and r["ci_hi"] < r["rmse_xb"]
    ]
    stats = {
        "n_casts": int(xb["lon"].size),
        "n_models": len(names),
        "n_cells": len(rows),
        "a1_gate_pass": bool(therm),
        "a1_pass_cells": therm,
        "blend_csv": str(csv_path),
    }
    (da_dir / "stats.json").write_text(json.dumps(stats, indent=2, default=str))
    print(
        f"A1 gate {'PASS' if therm else 'FAIL'}  cells={len(rows)}  pass={len(therm)}  {csv_path}",
        flush=True,
    )
    return stats


def selfcheck() -> None:
    src = np.arange(0.0, 11.0, 1.0)
    dst = np.array([0.0, 2.5, 5.0, 12.0, 20.0])
    T = np.vstack([src, 2.0 * src])
    S = np.vstack([10.0 + src, 20.0 + src])
    To, So = interp_native_to_obs(src, T, S, dst, zmax=10.0)
    assert abs(To[0, 1] - 2.5) < 1e-12
    assert abs(To[1, 2] - 10.0) < 1e-12
    assert not np.isfinite(To[0, 3])  # 12 m > zmax
    assert abs(So[0, 2] - 15.0) < 1e-12
    Tn, Sn = interp_obs_to_native(
        np.array([0.0, 5.0, 10.0]),
        np.array([[0.0, 5.0, 10.0]]),
        np.array([[1.0, 1.0, 1.0]]),
        np.array([2.5]),
    )
    assert abs(Tn[0, 0] - 2.5) < 1e-12
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[1.0, 4.0], [3.0, 8.0]])
    m = np.array([[True, True], [True, False]])
    r = _rmse_bias(a, b, m)
    assert r["n"] == 3
    assert abs(r["rmse"] - np.sqrt((0.0 + 4.0 + 0.0) / 3)) < 1e-12
    assert abs(r["bias"] - (-2.0 / 3)) < 1e-12
    pkt = bin1deg(np.array([-90.1, -90.2, -89.1]), np.array([25.1, 25.2, 25.3]), np.array([1.0, 3.0, 5.0]))
    assert pkt["n"][0, 0] == 2
    assert abs(pkt["mean"][0, 0] - 2.0) < 1e-12
    from evalphys.constants import CP_J_KGK, RHO0_KGM3
    from evalphys.metrics import ocean_heat_content
    from preproc.pair_table import nearest_eval_pairs

    q = ocean_heat_content(np.array([[1.0, 1.0]]), np.array([0.0, 10.0]), z_max=10.0)
    assert abs(q[0] - RHO0_KGM3 * CP_J_KGK * 10.0 * 1e-9) < 1e-12
    lat = np.array([25.0, 25.01, 25.02])
    lon = np.array([-90.0, -90.01, -90.02])
    juld = np.array([100.0, 110.0, 120.0])
    pairs = nearest_eval_pairs(lat, lon, juld, max_km=250, min_dt=3, max_dt=45)
    assert pairs.size >= 1
    assert np.all(pairs["dt_days"] > 0)
    print("selfcheck: compare_xb_argo helpers ok")


def _score_preds(xb, preds, do_n2: bool) -> dict:
    slices = _slice_masks(xb)
    pair_j = xb.get("pair_source_j")
    if pair_j is not None:
        slices["pair"] = np.asarray(pair_j, dtype=np.int32) >= 0
    out = {}
    for sname, mask in slices.items():
        if not mask.any():
            continue
        block = {"n": int(mask.sum())}
        for tag, Tp, Sp in (("xb", xb["T_xb"], xb["S_xb"]), ("xa", xb["T_xa"], xb["S_xa"])):
            block[tag] = score_vs_argo(
                Tp[mask], Sp[mask], xb["T_argo"][mask], xb["S_argo"][mask],
                xb["z"], xb["valid_T"][mask], xb["valid_S"][mask],
                xb["lat"][mask], xb["lon"][mask], n2=do_n2 and sname in ("all", "pair"),
            )
        block["models"] = {}
        for name, pack in preds.items():
            block["models"][name] = score_vs_argo(
                pack["T"][mask], pack["S"][mask], xb["T_argo"][mask], xb["S_argo"][mask],
                xb["z"], xb["valid_T"][mask], xb["valid_S"][mask],
                xb["lat"][mask], xb["lon"][mask], n2=do_n2 and sname in ("all", "pair"),
                sig_T=pack.get("sigT")[mask]
                if pack.get("sigT") is not None and np.isfinite(pack["T"][mask]).any()
                else None,
            )
        out[sname] = block
    return out


def plot_ence_map(xb, preds, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    packs = [(n, p) for n, p in preds.items() if p.get("sigT") is not None]
    if not packs:
        return
    z = xb["z"]
    iz = int(np.where(np.isclose(z, Z_NEAR_M))[0][0])
    vT = xb["valid_T"][:, iz]
    n_col = 3
    n_row = int(np.ceil(len(packs) / n_col))
    fig, axes = plt.subplots(n_row, n_col, figsize=(12.0, 3.2 * n_row), squeeze=False)
    for i, (name, pack) in enumerate(packs):
        ax = axes[i // n_col][i % n_col]
        cov = (np.abs(pack["T"][:, iz] - xb["T_argo"][:, iz]) < pack["sigT"][:, iz]).astype(np.float64)
        cov[~vT] = np.nan
        pkt = bin1deg(xb["lon"], xb["lat"], cov)
        if pkt["lon0"].size == 0:
            ax.set_title(name + " (empty)")
            continue
        lon_e = np.append(pkt["lon0"], pkt["lon0"][-1] + 1)
        lat_e = np.append(pkt["lat0"], pkt["lat0"][-1] + 1)
        pcm = ax.pcolormesh(lon_e, lat_e, np.ma.masked_invalid(pkt["mean"]), cmap="viridis", vmin=0, vmax=1, shading="flat")
        ax.set_title(f"{name} 10 m 1σ coverage", fontsize=9)
        ax.set_aspect("equal", adjustable="box")
        fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
    for j in range(len(packs), n_row * n_col):
        axes[j // n_col][j % n_col].axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "map_ence.png", dpi=140)
    plt.close(fig)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nc", default=str(NC_DEFAULT))
    p.add_argument("--out", default=str(OUT_DEFAULT))
    p.add_argument("--api", default=str(API_DEFAULT))
    p.add_argument("--models", nargs="*", default=[m["name"] for m in MODELS])
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--no-n2", action="store_true")
    p.add_argument("--selfcheck", action="store_true")
    p.add_argument("--mode", choices=("all", "predict", "score", "da-readiness"), default="all")
    p.add_argument("--nes-nc", default="", help="profiles_nes.nc for --mode da-readiness or score")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--from-stats", default="", help="rewrite markdown from an existing stats.json")
    p.add_argument("--plots-only", action="store_true", help="redraw figures from out/profiles_nes.nc")
    args = p.parse_args(argv)
    if args.selfcheck:
        selfcheck()
        return 0
    if args.from_stats:
        stats = json.loads(Path(args.from_stats).read_text())
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        write_md(_REPO / "reports" / "eval_xb_argo.md", stats)
        write_md(out_dir / "eval_xb_argo.md", stats)
        print(f"rewrote markdown from {args.from_stats}", flush=True)
        return 0

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = [m for m in MODELS if m["name"] in set(args.models)]
    if not specs:
        raise SystemExit(f"no models matched {args.models}")

    xb = load_xb(Path(args.nc))
    if args.limit:
        xb = slice_xb(xb, int(args.limit))

    n = xb["lon"].size
    print(f"loaded {n} casts from {args.nc} (zip {ZIP_DEFAULT.name})", flush=True)
    pred_nc = Path(args.nes_nc) if args.nes_nc else out_dir / "profiles_nes.nc"
    sat_npz = out_dir / "sat_scalars.npz"

    if args.mode == "da-readiness":
        if not pred_nc.is_file():
            raise SystemExit(f"da-readiness needs {pred_nc}")
        preds = load_preds_nc(pred_nc)
        run_da_readiness_a1(xb, preds, out_dir, n_boot=int(args.n_boot))
        return 0

    if args.plots_only or args.mode == "score":
        if not pred_nc.is_file() or not sat_npz.is_file():
            raise SystemExit(f"score/plots need {pred_nc} and {sat_npz}")
        preds = load_preds_nc(pred_nc)
        zsat = np.load(sat_npz)
        sss, sst, ssh = zsat["sss"], zsat["sst"], zsat["ssh"]
        if "pair_source_j" in __import__("netCDF4").Dataset(str(pred_nc)).variables:
            from netCDF4 import Dataset

            ds = Dataset(str(pred_nc))
            xb["pair_source_j"] = np.array(ds["pair_source_j"][:], dtype=np.int32)
            ds.close()
        if args.plots_only:
            plot_residuals(xb, preds, out_dir, "T")
            plot_residuals(xb, preds, out_dir, "S")
            plot_maps(xb, preds, sss, sst, ssh, out_dir)
            plot_ence_map(xb, preds, out_dir)
            print(f"redrew plots in {out_dir}", flush=True)
            return 0
        do_n2 = not args.no_n2
        stats = {
            "nc": str(Path(args.nc).resolve()),
            "zip": str(ZIP_DEFAULT),
            "n": n,
            "n_cycles": int(len(set(xb["analysis_date"]))),
            "n_ood": int(xb["ood"].sum()),
            "models": list(preds.keys()),
            "track": {m["name"]: m.get("track", "map") for m in MODELS},
            "n_sat": {k: int(np.isfinite(v["T"]).any(axis=1).sum()) for k, v in preds.items()},
            "slices": _score_preds(xb, preds, do_n2),
        }
        (out_dir / "stats.json").write_text(json.dumps(stats, indent=2, default=str))
        plot_residuals(xb, preds, out_dir, "T")
        plot_residuals(xb, preds, out_dir, "S")
        plot_maps(xb, preds, sss, sst, ssh, out_dir)
        plot_ence_map(xb, preds, out_dir)
        write_md(_REPO / "reports" / "eval_xb_argo.md", stats)
        write_md(out_dir / "eval_xb_argo.md", stats)
        print(f"scored {out_dir}", flush=True)
        return 0

    _api_on_path(Path(args.api))
    sss, sst, ssh = load_sat(xb["times"], xb["lat"], xb["lon"], sat_npz)
    sat_ok = np.isfinite(sss) & np.isfinite(sst) & np.isfinite(ssh)
    print(f"sat finite {int(sat_ok.sum())}/{n}", flush=True)

    need_ops = any(s["features"] == "ops" for s in specs)
    ops = None
    if need_ops:
        ops = load_ops(xb["times"], xb["lat"], xb["lon"], out_dir / "ops_19.npz")
        ops_ok = sat_ok & np.isfinite(ops).all(axis=1)
        print(f"ops finite {int(ops_ok.sum())}/{n}", flush=True)
    else:
        ops_ok = sat_ok

    from services.accessor.v2_inputs import build_v2_inputs

    feat_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def features_for(kind: str) -> tuple[np.ndarray, np.ndarray]:
        if kind in feat_cache:
            return feat_cache[kind]
        if kind == "ops":
            idx = np.where(ops_ok)[0]
            x = build_v2_inputs(
                "ops",
                [xb["times"][i] for i in idx],
                xb["lat"][idx],
                xb["lon"][idx],
                sss[idx],
                sst[idx],
                ssh[idx],
                ops=ops[idx],
            )
        elif kind == "HeaveFast":
            idx = np.where(sat_ok)[0]
            x = build_v2_inputs(
                "HeaveFast",
                [xb["times"][i] for i in idx],
                xb["lat"][idx],
                xb["lon"][idx],
                sss[idx],
                sst[idx],
                ssh[idx],
            )
        else:
            idx = np.where(sat_ok)[0]
            x = build_v2_inputs(
                "A_CRPS",
                [xb["times"][i] for i in idx],
                xb["lat"][idx],
                xb["lon"][idx],
                sss[idx],
                sst[idx],
                ssh[idx],
            )
        feat_cache[kind] = (x, idx)
        return x, idx

    pair_specs = [s for s in specs if s["kind"] in ("pair_pca", "persistence")]
    pairs = None
    if pair_specs:
        pairs, _juld = build_xb_pairs(xb)
        src_j = np.full(n, -1, dtype=np.int32)
        src_j[pairs["target_i"]] = pairs["source_j"]
        xb["pair_source_j"] = src_j
        print(f"pair neighbors {int(pairs.size)}/{n}", flush=True)

    preds_native = {}
    for spec in specs:
        print(f"predict {spec['name']}", flush=True)
        if spec["kind"] == "persistence":
            To = np.full((n, xb["z"].size), np.nan)
            So = np.full((n, xb["z"].size), np.nan)
            if pairs is not None and pairs.size:
                ti, sj = pairs["target_i"], pairs["source_j"]
                To[ti] = xb["T_argo"][sj]
                So[ti] = xb["S_argo"][sj]
            preds_native[spec["name"]] = {"T": To, "S": So, "n_sat": int((To[:, 0] == To[:, 0]).sum())}
            continue
        if spec["kind"] == "pair_pca":
            from preproc.pair_table import pack_pair_inputs

            x9, idx9 = features_for("A_CRPS")
            x9_full = np.full((n, x9.shape[1]), np.nan, dtype=np.float32)
            x9_full[idx9] = x9
            cell = LocalPCA(spec["ckpt"], spec["cache"])
            Tn, Sn = interp_obs_to_native(
                xb["z"], xb["T_argo"], xb["S_argo"], cell.depth, xb["valid_T"] & xb["valid_S"]
            )
            pcs = _profiles_to_pcs(Tn, Sn, cell.pca)
            keep = []
            for k, row in enumerate(pairs):
                i, j = int(row["target_i"]), int(row["source_j"])
                if np.isfinite(x9_full[i]).all() and np.isfinite(pcs[j]).all():
                    keep.append(k)
            keep = np.asarray(keep, dtype=np.int32)
            if keep.size == 0:
                preds_native[spec["name"]] = {
                    "T": np.full((n, xb["z"].size), np.nan),
                    "S": np.full((n, xb["z"].size), np.nan),
                    "n_sat": 0,
                }
                continue
            sub = pairs[keep]
            xp = pack_pair_inputs(x9_full, pcs, sub)
            if spec.get("mix_flag") is not None:
                flag = np.full((xp.shape[0], 1), float(spec["mix_flag"]), dtype=np.float32)
                xp = np.concatenate([xp, flag], axis=1)
            Tp, Sp, sigT, sigS = cell.predict(xp)
            T = np.full((n, cell.depth.size), np.nan)
            S = np.full((n, cell.depth.size), np.nan)
            T[sub["target_i"]] = Tp
            S[sub["target_i"]] = Sp
            To, So = interp_native_to_obs(cell.depth, T, S, xb["z"])
            pack = {"T": To, "S": So, "n_sat": int(sub.size)}
            if sigT is not None:
                sT = np.full((n, cell.depth.size), np.nan)
                sS = np.full((n, cell.depth.size), np.nan)
                sT[sub["target_i"]] = sigT
                sS[sub["target_i"]] = sigS
                sTo, _ = interp_native_to_obs(cell.depth, sT, sT, xb["z"])
                sSo, _ = interp_native_to_obs(cell.depth, sS, sS, xb["z"])
                pack["sigT"], pack["sigS"] = sTo, sSo
            preds_native[spec["name"]] = pack
            continue
        x, idx = features_for(spec["features"])
        T, S, src_z, sigT, sigS = predict_model(spec, x, idx, n)
        To, So = interp_native_to_obs(src_z, T, S, xb["z"])
        pack = {"T": To, "S": So, "n_sat": int(idx.size), "idx": idx}
        if sigT is not None:
            sTo, _ = interp_native_to_obs(src_z, sigT, sigT, xb["z"])
            sSo, _ = interp_native_to_obs(src_z, sigS, sigS, xb["z"])
            pack["sigT"], pack["sigS"] = sTo, sSo
        preds_native[spec["name"]] = pack

    write_nc(pred_nc, xb, preds_native, sss, sst, ssh)
    if args.mode == "predict":
        print(f"wrote predictions {pred_nc}", flush=True)
        return 0

    do_n2 = not args.no_n2
    stats = {
        "nc": str(Path(args.nc).resolve()),
        "zip": str(ZIP_DEFAULT),
        "n": n,
        "n_cycles": int(len(set(xb["analysis_date"]))),
        "n_ood": int(xb["ood"].sum()),
        "models": [s["name"] for s in specs],
        "track": {s["name"]: s.get("track", "map") for s in specs},
        "n_sat": {k: int(v.get("n_sat", 0)) for k, v in preds_native.items()},
        "slices": _score_preds(xb, preds_native, do_n2),
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2, default=str))
    plot_residuals(xb, preds_native, out_dir, "T")
    plot_residuals(xb, preds_native, out_dir, "S")
    plot_maps(xb, preds_native, sss, sst, ssh, out_dir)
    plot_ence_map(xb, preds_native, out_dir)
    write_md(_REPO / "reports" / "eval_xb_argo.md", stats)
    write_md(out_dir / "eval_xb_argo.md", stats)
    print(f"wrote {out_dir}", flush=True)
    all_ = stats["slices"]["all"]
    print(
        "T RMSE xb={:.3f} xa={:.3f} ".format(all_["xb"]["T"]["rmse"], all_["xa"]["T"]["rmse"])
        + " ".join(
            f"{m}={all_['models'][m]['T']['rmse']:.3f}"
            for m in stats["models"]
            if m in all_["models"] and all_["models"][m]["T"]["rmse"] is not None
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
