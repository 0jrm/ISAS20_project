"""Build the per-cast parquet and the narrow level table from xb/nes NetCDF."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from evalphys.constants import LC_LAT_RANGE, LC_LON_RANGE
from evalphys.profile_features import RULE_VERSION, band_rmse, shape_features

TRAIN_LAT = (19.23, 28.70)
TRAIN_LON = (-96.80, -82.67)
SKIP_NES_VARS = frozenset({"argo", "xb", "xa"})
MAP_SKIP = frozenset({"stoch_eof_pair", "stoch_eof_pair_mix"})


@dataclass(frozen=True)
class ProfileBundle:
    z: np.ndarray
    lon: np.ndarray
    lat: np.ndarray
    time_iso: np.ndarray
    analysis_date: np.ndarray
    t_argo: np.ndarray
    t_xb: np.ndarray
    s_argo: np.ndarray
    s_xb: np.ndarray
    valid_t: np.ndarray
    ssh_sat: np.ndarray
    products: dict[str, np.ndarray]


def _as_str(arr) -> np.ndarray:
    return np.array([str(x) for x in np.asarray(arr)])


def load_profile_bundle(xb_path: Path, nes_path: Path) -> ProfileBundle:
    from netCDF4 import Dataset

    xb = Dataset(str(xb_path))
    nes = Dataset(str(nes_path))
    products: dict[str, np.ndarray] = {}
    for name in nes.variables:
        if not name.startswith("T_"):
            continue
        stem = name[2:]
        if stem in SKIP_NES_VARS:
            continue
        products[stem] = np.array(nes[name][:], dtype=np.float64)
    bundle = ProfileBundle(
        z=np.array(xb["z"][:], dtype=np.float64),
        lon=np.array(xb["lon"][:], dtype=np.float64),
        lat=np.array(xb["lat"][:], dtype=np.float64),
        time_iso=_as_str(xb["time_iso"][:]),
        analysis_date=_as_str(xb["analysis_date"][:]),
        t_argo=np.array(xb["T_argo"][:], dtype=np.float64),
        t_xb=np.array(xb["T_xb"][:], dtype=np.float64),
        s_argo=np.array(xb["S_argo"][:], dtype=np.float64),
        s_xb=np.array(xb["S_xb"][:], dtype=np.float64),
        valid_t=np.array(xb["valid_T"][:]).astype(bool),
        ssh_sat=(
            np.array(nes["ssh_sat"][:], dtype=np.float64)
            if "ssh_sat" in nes.variables
            else np.full(np.array(xb["lon"][:]).shape[0], np.nan, dtype=np.float64)
        ),
        products=products,
    )
    xb.close()
    nes.close()
    return bundle


def region_flags(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ood = (
        (lat < TRAIN_LAT[0])
        | (lat > TRAIN_LAT[1])
        | (lon < TRAIN_LON[0])
        | (lon > TRAIN_LON[1])
    )
    in_bbox = ~ood
    lc = (
        (lat >= LC_LAT_RANGE[0])
        & (lat <= LC_LAT_RANGE[1])
        & (lon >= LC_LON_RANGE[0])
        & (lon <= LC_LON_RANGE[1])
    )
    return in_bbox, ood, lc


def era_label(analysis_date: np.ndarray) -> np.ndarray:
    return np.array([s[:4] for s in analysis_date])


def _prefix_features(prefix: str, feat: dict) -> dict[str, np.ndarray]:
    keys = ("z20_m", "thickness_15_25_m", "peak_dtdz_c_per_m", "peak_dtdz_z_m", "mld_m")
    return {f"{prefix}_{k}": feat[k] for k in keys}


def git_short_hash(repo: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit"


def build_cast_and_level_tables(
    bundle: ProfileBundle,
    *,
    models: list[str] | None = None,
    with_mld: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    names = [m for m in bundle.products if m not in MAP_SKIP]
    if models:
        names = [m for m in names if m in set(models)]
    n = bundle.lon.size
    in_bbox, ood, lc = region_flags(bundle.lat, bundle.lon)
    era = era_label(bundle.analysis_date)
    argo_feat = shape_features(
        bundle.z,
        bundle.t_argo,
        bundle.valid_t,
        salinity=bundle.s_argo if with_mld else None,
        lat=bundle.lat if with_mld else None,
        lon=bundle.lon if with_mld else None,
    )
    xb_feat = shape_features(
        bundle.z,
        bundle.t_xb,
        bundle.valid_t,
        salinity=bundle.s_xb if with_mld else None,
        lat=bundle.lat if with_mld else None,
        lon=bundle.lon if with_mld else None,
    )
    rmse_xb = band_rmse(bundle.t_xb, bundle.t_argo, bundle.valid_t, bundle.z, 50.0, 200.0)
    cast_frames = []
    level_frames = []
    z = bundle.z
    for model in names:
        t_nes = bundle.products[model]
        nes_feat = shape_features(bundle.z, t_nes, bundle.valid_t)
        rmse_nes = band_rmse(t_nes, bundle.t_argo, bundle.valid_t, bundle.z, 50.0, 200.0)
        row = {
            "cast_id": np.arange(n, dtype=np.int32),
            "model": np.full(n, model),
            "time_iso": bundle.time_iso,
            "analysis_date": bundle.analysis_date,
            "era": era,
            "lat": bundle.lat,
            "lon": bundle.lon,
            "ssh_sat": bundle.ssh_sat,
            "in_bbox": in_bbox,
            "ood": ood,
            "lc": lc,
            "rmse_t_50_200_xb": rmse_xb,
            "rmse_t_50_200_nes": rmse_nes,
            **_prefix_features("argo", argo_feat),
            **_prefix_features("xb", xb_feat),
            **_prefix_features("nes", nes_feat),
            "smoother": np.full(n, argo_feat["smoother"]),
            "rule_version": np.full(n, RULE_VERSION),
        }
        cast_frames.append(pd.DataFrame(row))
        level_frames.append(
            pd.DataFrame(
                {
                    "cast_id": np.repeat(np.arange(n, dtype=np.int32), z.size),
                    "model": np.full(n * z.size, model),
                    "z": np.tile(z, n),
                    "t_argo": bundle.t_argo.reshape(-1),
                    "t_xb": bundle.t_xb.reshape(-1),
                    "t_nes": t_nes.reshape(-1),
                    "valid_t": bundle.valid_t.reshape(-1),
                }
            )
        )
    return pd.concat(cast_frames, ignore_index=True), pd.concat(level_frames, ignore_index=True)


def woa_feature_table(long_df: pd.DataFrame, *, value_col: str = "t_woa") -> pd.DataFrame:
    wide = long_df.pivot(index="cast_id", columns="z", values=value_col).sort_index(axis=1)
    z = wide.columns.to_numpy(dtype=np.float64)
    t = wide.to_numpy(dtype=np.float64)
    feat = shape_features(z, t, np.isfinite(t))
    out = pd.DataFrame({"cast_id": wide.index.astype(np.int32)})
    for key in ("z20_m", "thickness_15_25_m", "peak_dtdz_c_per_m", "peak_dtdz_z_m"):
        out[f"woa_{key}"] = feat[key]
    return out


def parquet_stem(git_hash: str, rule: str = RULE_VERSION) -> str:
    return f"cast_features_{git_hash}_{rule}"
