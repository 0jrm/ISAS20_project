#!/usr/bin/env python3
"""Hstack HeaveFast ablation extras onto the 9-d v2 cache. ONI/RONI still splice at train load."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_REPO = _ROOT.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

CHANNELS = ("sss", "sst", "ssh")
DLAT = np.array([-1.0, 0.0, 1.0])
DLON = np.array([-1.0, 0.0, 1.0])
T_OFF = np.array([-2, -1, 0], dtype=np.int64)
OP_NAMES = (
    "sst.grad_x@local",
    "sst.grad_y@local",
    "sst.grad_x@1.0deg",
    "sst.grad_y@1.0deg",
    "sss.grad_x@local",
    "sss.grad_y@local",
    "sss.grad_x@1.0deg",
    "sss.grad_y@1.0deg",
    "ssh.grad_x@local",
    "ssh.grad_y@local",
    "ssh.grad_x@1.0deg",
    "ssh.grad_y@1.0deg",
    "ssh.laplacian@1.0deg",
    "sst.tendency@7d",
    "ssh.tendency@7d",
    "ssh.geo_u@local",
    "ssh.geo_v@local",
    "ssh.geo_u@1.0deg",
    "ssh.geo_v@1.0deg",
)
WIND_ROOT = Path("/unity/g2/jmiranda/SubsurfaceFields/Data/Wind")
ARGO_H5 = _REPO / "data/NeSPReSO_v2_ARGO_GoM_sat/satellite_NeSPReSO_v2_ARGO_GoM.h5"
OPS_CACHE = _ROOT / "data/cache/train_ready_2ab55b15b14f.pkl"
CUBE_PATH = _ROOT / "data/cube/gom_cube.zarr"
V2_SRC = "/unity/g2/jmiranda/v2-nespreso/src"
DEFAULT_BASE = _REPO / "data/cache/train_ready_3adcff404b0b.pkl"
KIND_OUT = {
    "conv": "train_ready_heave_conv_3x3.pkl",
    "ops": "train_ready_heave_ops.pkl",
    "bathy": "train_ready_heave_bathy.pkl",
    "bathy_wind": "train_ready_heave_bathy_wind.pkl",
}
KIND_META = {
    "conv": {"pad": (1, 2), "shape": [3, 3, 3, 3], "cache_kind": "heave_conv"},
    "ops": {"pad": (0, 0), "shape": None, "cache_kind": "heave_ops"},
    "bathy": {"pad": (0, 0), "shape": None, "cache_kind": "heave_bathy"},
    "bathy_wind": {"pad": (0, 0), "shape": None, "cache_kind": "heave_bathy_wind"},
}


def fill_nan_from_center(vol: np.ndarray) -> np.ndarray:
    """NaN at (c,t,h,w) copies the same (c,t) center pixel; remaining NaN → 0."""
    vol = np.asarray(vol, dtype=np.float32)
    center = vol[:, :, :, 1:2, 1:2]
    filled = np.where(np.isfinite(vol), vol, center)
    return np.nan_to_num(filled, nan=0.0).astype(np.float32)


def flatten_cthw(vol: np.ndarray) -> np.ndarray:
    """C-order flatten of (N,C,T,H,W). W fastest, then H, then T, matching PatchConvMLP.view."""
    vol = np.asarray(vol, dtype=np.float32)
    n, c, t, h, w = vol.shape
    return np.ascontiguousarray(vol).reshape(n, c * t * h * w)


def _dates_from_juld(juld: np.ndarray) -> np.ndarray:
    if V2_SRC not in sys.path:
        sys.path.insert(0, V2_SRC)
    from nespreso.utils.time import datenum_to_datetime

    out = []
    for t in np.asarray(juld, dtype=np.float64).ravel():
        dt = datenum_to_datetime(float(t))
        out.append(np.datetime64(dt.date(), "D"))
    return np.asarray(out)


def _t0_indices(dates: np.ndarray) -> np.ndarray:
    from preproc.cube.cube_schema import TIME_END, TIME_START, time_indices_of_days

    days = np.clip(np.asarray(dates, dtype="datetime64[D]"), TIME_START, TIME_END)
    return time_indices_of_days(days)


def sample_conv_patch(lat: np.ndarray, lon: np.ndarray, dates: np.ndarray, cube_path: Path) -> np.ndarray:
    from preproc.features.sampler import CubeProvider, MissingCubePlaneError, sample_plane

    n = len(lat)
    provider = CubeProvider(cube_path)
    t0 = _t0_indices(dates)
    n_t = int(np.asarray(provider.root["coords/time"]).shape[0])
    print(f"[export] conv unique t0={len(np.unique(t0))} n={n}", file=sys.stderr, flush=True)
    offset_w = [
        {ch: provider.weights_for(ch, lat + dlat, lon + dlon) for ch in CHANNELS}
        for dlat in DLAT
        for dlon in DLON
    ]
    out = np.full((n, 3, 3, 3, 3), np.nan, dtype=np.float32)
    for t0_idx in np.unique(t0):
        rows = np.flatnonzero(t0 == t0_idx)
        for it, dt in enumerate(T_OFF):
            t_idx = int(np.clip(int(t0_idx) + int(dt), 0, n_t - 1))
            for ic, ch in enumerate(CHANNELS):
                try:
                    plane = provider.plane(ch, t_idx)
                except MissingCubePlaneError:
                    continue
                k = 0
                for ih in range(3):
                    for iw in range(3):
                        v, _m = sample_plane(offset_w[k][ch], plane)
                        out[rows, ic, it, ih, iw] = v[rows]
                        k += 1
    return flatten_cthw(fill_nan_from_center(out))


def sample_bathy_center(h5_path: Path, n: int) -> np.ndarray:
    import h5py

    with h5py.File(h5_path, "r") as f:
        elev = np.asarray(f["bathymetry/elevation"][:, 2, 2], dtype=np.float32)
    if elev.shape[0] != n:
        raise ValueError(f"bathymetry n={elev.shape[0]} != cache n={n}")
    return np.maximum(0.0, -elev).astype(np.float32).reshape(n, 1)


def sample_wind_t0(lat: np.ndarray, lon: np.ndarray, dates: np.ndarray, wind_root: Path) -> np.ndarray:
    import netCDF4 as nc

    n = len(lat)
    out = np.zeros((n, 3), dtype=np.float32)
    lon360 = np.asarray(lon, dtype=np.float64) % 360.0
    lat64 = np.asarray(lat, dtype=np.float64)
    days = np.asarray(dates, dtype="datetime64[D]")
    uniq = np.unique(days)
    print(f"[export] wind unique days={len(uniq)} n={n}", file=sys.stderr, flush=True)
    for day in uniq:
        ymd = str(day).replace("-", "")
        path = wind_root / f"NBSv02_wind_daily_{ymd}.nc"
        rows = np.flatnonzero(days == day)
        if not path.is_file():
            continue
        try:
            ds = nc.Dataset(path, "r")
        except OSError:
            print(f"[export] skip wind {path.name}", file=sys.stderr, flush=True)
            continue
        try:
            glat = np.asarray(ds.variables["lat"][:], dtype=np.float64)
            glon = np.asarray(ds.variables["lon"][:], dtype=np.float64)
            fields = []
            for name in ("u_wind", "v_wind", "windspeed"):
                arr = np.asarray(ds.variables[name][:], dtype=np.float32)
                while arr.ndim > 2:
                    arr = arr[0]
                fields.append(arr)
        finally:
            ds.close()
        i = np.clip(np.searchsorted(glat, lat64[rows]) - 1, 0, len(glat) - 2)
        j = np.clip(np.searchsorted(glon, lon360[rows]) - 1, 0, len(glon) - 2)
        dlat = glat[i + 1] - glat[i]
        dlon = glon[j + 1] - glon[j]
        ty = np.zeros(len(rows), dtype=np.float64)
        tx = np.zeros(len(rows), dtype=np.float64)
        np.divide(lat64[rows] - glat[i], dlat, out=ty, where=dlat != 0)
        np.divide(lon360[rows] - glon[j], dlon, out=tx, where=dlon != 0)
        ty = np.clip(ty, 0.0, 1.0)
        tx = np.clip(tx, 0.0, 1.0)
        for c, field in enumerate(fields):
            v00 = field[i, j]
            v01 = field[i, j + 1]
            v10 = field[i + 1, j]
            v11 = field[i + 1, j + 1]
            val = (1 - ty) * (1 - tx) * v00 + (1 - ty) * tx * v01 + ty * (1 - tx) * v10 + ty * tx * v11
            out[rows, c] = np.nan_to_num(val, nan=0.0).astype(np.float32)
    return out


def ops_from_cube_cache(base_juld: np.ndarray, ops_path: Path) -> np.ndarray:
    with open(ops_path, "rb") as f:
        other = pickle.load(f)
    names = list(other["feature_names"])
    missing = [n for n in OP_NAMES if n not in names]
    if missing:
        raise KeyError(f"operator cache missing {missing}")
    oj = np.asarray(other["JULD"], dtype=np.float64)
    bj = np.asarray(base_juld, dtype=np.float64)
    if oj.shape != bj.shape or not np.allclose(oj, bj):
        raise ValueError(f"operator cache JULD does not match base ({oj.shape} vs {bj.shape})")
    if oj.shape[0] != 4145:
        raise ValueError(f"expected 4145 profiles, got {oj.shape[0]}")
    idx = [names.index(n) for n in OP_NAMES]
    vals = np.asarray(other["inputs"], dtype=np.float32)
    feat = vals if vals.shape[1] == len(names) else vals[:, -len(names) :]
    return np.nan_to_num(feat[:, idx], nan=0.0).astype(np.float32)


def extras_for(kind: str, cache: dict) -> np.ndarray:
    n = cache["inputs"].shape[0]
    lat = np.asarray(cache["LAT"], dtype=np.float64).reshape(-1)[:n]
    lon = np.asarray(cache["LON"], dtype=np.float64).reshape(-1)[:n]
    dates = _dates_from_juld(cache["JULD"])
    if kind == "conv":
        return sample_conv_patch(lat, lon, dates, CUBE_PATH)
    if kind == "ops":
        return ops_from_cube_cache(cache["JULD"], OPS_CACHE)
    if kind == "bathy":
        return sample_bathy_center(ARGO_H5, n)
    if kind == "bathy_wind":
        return np.concatenate(
            [sample_bathy_center(ARGO_H5, n), sample_wind_t0(lat, lon, dates, WIND_ROOT)],
            axis=1,
        )
    raise ValueError(kind)


def _zscore_train(extras: np.ndarray, juld: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from base.split_utils import build_split_indices

    n = extras.shape[0]
    splits = build_split_indices(
        n,
        juld,
        {"split_mode": "chronological", "train_frac": 0.7, "val_frac": 0.15, "test_frac": 0.15},
        dataset_tag="argo_v2",
        v2_src=V2_SRC,
    )
    tr = np.asarray(splits["train"], dtype=int)
    mu = extras[tr].mean(axis=0)
    sd = extras[tr].std(axis=0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return ((extras - mu) / sd).astype(np.float32), mu.astype(np.float32), sd.astype(np.float32)


def write_kind(kind: str, base_path: Path, out_path: Path) -> Path:
    with open(base_path, "rb") as f:
        cache = pickle.load(f)
    extras = extras_for(kind, cache)
    extra_mu = extra_sd = None
    if kind in ("bathy", "bathy_wind"):
        extras, extra_mu, extra_sd = _zscore_train(extras, cache["JULD"])
    payload = dict(cache)
    payload["inputs"] = np.concatenate(
        [np.asarray(cache["inputs"], dtype=np.float32), extras], axis=1
    )
    if extra_mu is not None:
        payload["extra_mean"] = extra_mu
        payload["extra_std"] = extra_sd
    meta = KIND_META[kind]
    payload["spatial_pad"] = meta["pad"][0]
    payload["temporal_pad"] = meta["pad"][1]
    payload["sat_patch_shape"] = meta["shape"]
    payload["cache_kind"] = meta["cache_kind"]
    payload["dataset_tag"] = "argo_v2"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(payload, f, protocol=4)
    return out_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="HeaveFast ablation caches from the 9-d v2 pickle")
    p.add_argument("--kind", choices=tuple(KIND_META))
    p.add_argument("--all", action="store_true", help="write all four pickles next to --base")
    p.add_argument("--base", type=Path, default=DEFAULT_BASE)
    p.add_argument("--out", type=Path, help="output pickle (required with --kind)")
    args = p.parse_args(argv)
    if args.all:
        outdir = args.base.parent
        for kind, name in KIND_OUT.items():
            print(write_kind(kind, args.base, outdir / name))
        return 0
    if not args.kind or args.out is None:
        p.error("need --kind KIND --out PATH, or --all")
    print(write_kind(args.kind, args.base, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
