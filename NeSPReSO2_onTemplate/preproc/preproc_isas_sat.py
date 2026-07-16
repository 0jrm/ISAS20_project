import os
import sys
import hashlib
import json
import pickle
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import h5py
import numpy as np
from sklearn.decomposition import PCA

from model.loss import OUTPUT_H5_VARS, get_pca_weights, true_profiles_numpy

# ponytail: patch flatten order = time (oldest..newest within window), then row-major (lat, lon) per slice
PATCH_ORDER_DOC = "time_major_row_major_lat_lon"

ENCODING_KEYS = ("timecos", "timesin", "latcos", "latsin", "loncos", "lonsin")
SCALAR_EXTRA_KEYS = ("basin_sss", "basin_sst", "basin_ssh", "bathy_depth")
CENTER_SCALAR_KEYS = ("center_sss", "center_sst", "center_ssh")
ENGINEERED_PATCH_KEYS = (
    "patch_ssh_gradient",
    "patch_ssh_laplacian",
    "patch_temporal_tendency",
    "patch_sst_gradient",
)
SAT_KEYS = ("sss", "sst", "ssh")
SURFACE_RESIDUAL_SAT_KEY = {"temperature": "sst", "salinity": "sss"}
NORMALIZATION_VERSION = "train_zscore_v1"
ERROR_NORM_E0 = 1e-6  # Phase 2.2 log-offset for positive right-skewed error channels


def sat_patch_center_index(spatial_pad: int, temporal_pad: int) -> int:
    """Index of center pixel at latest time within one flattened sat-var block."""
    per_var = sat_features_per_var(spatial_pad, temporal_pad)
    if per_var == 1:
        return 0
    h = w = 2 * spatial_pad + 1
    t_win = temporal_pad + 1
    cy = cx = spatial_pad
    return (t_win - 1) * (h * w) + cy * w + cx


def surface_residual_feature_col(
    variable: str,
    *,
    n_enc: int,
    spatial_pad: int,
    temporal_pad: int,
    input_params: Mapping[str, bool],
) -> int:
    """Column index in the feature matrix for scalar SST (T) or SSS (S) residual."""
    key = SURFACE_RESIDUAL_SAT_KEY[variable]
    col = count_scalar_dims(input_params) if input_params is not None else n_enc
    per_var = sat_features_per_var(spatial_pad, temporal_pad)
    center = sat_patch_center_index(spatial_pad, temporal_pad)
    for sat_key in SAT_KEYS:
        if not input_params.get(sat_key):
            continue
        if sat_key == key:
            return col + center
        col += per_var
    raise KeyError(f"surface residual sat key {key!r} not enabled in input_params")


def surface_residual_from_features(
    features,
    variable: str,
    *,
    n_enc: int,
    spatial_pad: int,
    temporal_pad: int,
    input_params: Mapping[str, bool],
):
    """Scalar SST/SSS per row from cache inputs (center pixel, latest time)."""
    col = surface_residual_feature_col(
        variable,
        n_enc=n_enc,
        spatial_pad=spatial_pad,
        temporal_pad=temporal_pad,
        input_params=input_params,
    )
    if hasattr(features, "select"):
        return features[:, col]
    return features[:, col]


def count_encoding_dims(input_params: Mapping[str, bool]) -> int:
    return sum(1 for k in ENCODING_KEYS if input_params.get(k))


def count_scalar_extra_dims(input_params: Mapping[str, bool]) -> int:
    return sum(1 for k in SCALAR_EXTRA_KEYS if input_params.get(k))


def count_scalar_dims(input_params: Mapping[str, bool]) -> int:
    """Harmonics + optional basin/bathy scalars."""
    return count_encoding_dims(input_params) + count_scalar_extra_dims(input_params)


def count_center_scalar_dims(input_params: Mapping[str, bool]) -> int:
    return sum(1 for k in CENTER_SCALAR_KEYS if input_params.get(k))


def count_base_dims(input_params: Mapping[str, bool]) -> int:
    """Point-encoder block: harmonics + optional center SSH/SST/SSS scalars."""
    return count_encoding_dims(input_params) + count_center_scalar_dims(input_params)


def count_engineered_patch_channels(input_params: Mapping[str, bool]) -> int:
    """Extra patch channel planes (each is one full spatio-temporal block)."""
    n = 0
    if input_params.get("patch_ssh_gradient"):
        n += 2
    if input_params.get("patch_ssh_laplacian"):
        n += 1
    if input_params.get("patch_temporal_tendency"):
        n += len(SAT_KEYS)
    if input_params.get("patch_sst_gradient"):
        n += 2
    return n


def count_residual_patch_channels(
    input_params: Mapping[str, bool],
    *,
    use_mask_channels: bool = False,
) -> int:
    """Patch encoder channels: center-relative sat vars + optional engineered planes."""
    n_sat = count_sat_vars(input_params)
    if use_mask_channels and n_sat > 0:
        n_sat *= 2
    n_eng = count_engineered_patch_channels(input_params)
    return n_sat + n_eng


def compute_argo_residual_input_dim(
    input_params: Mapping[str, bool],
    spatial_pad: int,
    temporal_pad: int,
    *,
    use_mask_channels: bool = False,
    include_bathy_scalar: bool = False,
) -> int:
    """Residual layout: base scalars + optional bathy + flattened patch block."""
    base = count_base_dims(input_params)
    if include_bathy_scalar and input_params.get("bathy_depth"):
        base += 1
    per_ch = sat_features_per_var(spatial_pad, temporal_pad)
    n_ch = count_residual_patch_channels(input_params, use_mask_channels=use_mask_channels)
    return base + n_ch * per_ch


def resolve_use_mask_channels(config: Mapping[str, Any]) -> bool:
    """Whether satellite patches include interleaved mask columns (6 ch) vs values only (3 ch)."""
    arch_args = (config.get("arch") or {}).get("args") or {}
    if "use_mask_channels" in arch_args:
        return bool(arch_args["use_mask_channels"])
    input_params = config.get("input_params") or {}
    if "use_mask_channels" in input_params:
        return bool(input_params["use_mask_channels"])
    # legacy alias
    return bool(input_params.get("patch_mask"))


def count_patch_channels(
    input_params: Mapping[str, bool],
    n_sat_vars: int = 3,
    *,
    use_mask_channels: bool = False,
) -> int:
    """Patch channel count; doubles when mask planes are included (value + mask per var)."""
    n = count_sat_vars(input_params, n_sat_vars)
    if use_mask_channels and n > 0:
        return 2 * n
    return n


def count_sat_vars(input_params: Mapping[str, bool], n_sat_vars: int = 3) -> int:
    n = sum(1 for k in SAT_KEYS if input_params.get(k))
    if n == 0 and input_params.get("sat"):
        n = n_sat_vars
    return n


def sat_features_per_var(spatial_pad: int, temporal_pad: int) -> int:
    if spatial_pad == 0 and temporal_pad == 0:
        return 1
    spatial_size = 2 * spatial_pad + 1
    temporal_size = temporal_pad + 1
    return spatial_size * spatial_size * temporal_size


def sat_patch_shape(
    spatial_pad: int,
    temporal_pad: int,
    n_sat_vars: int = 3,
    *,
    input_params: Mapping[str, bool] | None = None,
    use_mask_channels: bool = False,
) -> tuple[int, int, int, int] | None:
    """``(C, T, H, W)`` when patches enabled; else ``None`` for point mode."""
    if spatial_pad == 0 and temporal_pad == 0:
        return None
    h = w = 2 * spatial_pad + 1
    n_ch = (
        count_patch_channels(input_params, n_sat_vars, use_mask_channels=use_mask_channels)
        if input_params
        else n_sat_vars
    )
    return (n_ch, temporal_pad + 1, h, w)


def compute_input_dim(
    input_params: Mapping[str, bool],
    spatial_pad: int = 0,
    temporal_pad: int = 0,
    n_sat_vars: int = 3,
    *,
    use_mask_channels: bool = False,
) -> int:
    """Feature count including flattened satellite patches when pads are set."""
    n_scalar = count_scalar_dims(input_params)
    n_ch = count_patch_channels(input_params, n_sat_vars, use_mask_channels=use_mask_channels)
    per_ch = sat_features_per_var(spatial_pad, temporal_pad)
    if spatial_pad == 0 and temporal_pad == 0:
        return n_scalar + count_sat_vars(input_params, n_sat_vars)
    return n_scalar + n_ch * per_ch


def compute_argo_l4_input_dim(
    input_params: Mapping[str, bool],
    spatial_pad: int,
    temporal_pad: int,
    *,
    use_mask_channels: bool = False,
) -> int:
    return compute_input_dim(
        input_params, spatial_pad, temporal_pad, use_mask_channels=use_mask_channels
    )


def config_hash(config: Dict) -> str:
    """Stable hash for cache filenames (includes ``io.dataset_tag``)."""
    payload = {
        "input_params": config.get("input_params"),
        "io": config.get("io"),
        "outputs": config.get("outputs"),
        "use_mask_channels": resolve_use_mask_channels(config),
    }
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def write_train_cache(payload: Dict, cache_dir: str, chash: str) -> str:
    """Pickle train-ready cache; returns path."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"train_ready_{chash}.pkl")
    with open(cache_path, "wb") as f:
        pickle.dump(payload, f)
    n = payload["inputs"].shape[0]
    d = payload["inputs"].shape[1]
    tag = payload.get("dataset_tag", "?")
    print(f"Train-ready cache saved to {cache_path} (tag={tag}, N={n}, D={d})")
    return cache_path


def _center_indices(n: int) -> int:
    return n // 2


def extract_sat_values(
    arr: np.ndarray,
    spatial_pad: int,
    temporal_pad: int,
) -> np.ndarray:
    """
    Extract scalar (offset=0) or flattened patch features per station.

    ``arr`` shape: (N,), (N, H, W), (N, T), or (N, T, H, W).
    """
    if arr.ndim == 1:
        return arr.astype(np.float32)

    if arr.ndim == 2:
        if spatial_pad == 0:
            return arr[:, _center_indices(arr.shape[1])].astype(np.float32)
        return arr.astype(np.float32).reshape(arr.shape[0], -1)

    if arr.ndim == 3:
        cy, cx = _center_indices(arr.shape[1]), _center_indices(arr.shape[2])
        if spatial_pad == 0:
            return arr[:, cy, cx].astype(np.float32)
        patch = arr[:, cy - spatial_pad : cy + spatial_pad + 1, cx - spatial_pad : cx + spatial_pad + 1]
        return patch.reshape(arr.shape[0], -1).astype(np.float32)

    # (N, T, H, W)
    t_win = temporal_pad + 1
    t_slice = arr[:, -t_win:] if temporal_pad > 0 else arr[:, -1:]
    cy, cx = _center_indices(arr.shape[2]), _center_indices(arr.shape[3])
    if spatial_pad == 0:
        vals = t_slice[:, :, cy, cx]
        if temporal_pad == 0:
            return vals[:, -1].astype(np.float32)
        return vals.astype(np.float32).reshape(arr.shape[0], -1)

    patches = t_slice[
        :,
        :,
        cy - spatial_pad : cy + spatial_pad + 1,
        cx - spatial_pad : cx + spatial_pad + 1,
    ]
    return patches.reshape(arr.shape[0], -1).astype(np.float32)


def extract_sat_patches_with_mask(
    arr: np.ndarray,
    spatial_pad: int,
    temporal_pad: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (values, mask) flattened patches per station; mask is 1 where value is finite."""
    raw = extract_sat_values(arr, spatial_pad, temporal_pad)
    if raw.ndim == 1:
        mask = np.isfinite(raw).astype(np.float32)
        return np.nan_to_num(raw, nan=0.0).astype(np.float32), mask
    finite = np.isfinite(raw)
    values = np.where(finite, raw, 0.0).astype(np.float32)
    mask = finite.astype(np.float32)
    return values, mask


def build_argo_l4_input_matrix(
    juld: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    sat_vars: Dict[str, np.ndarray],
    input_params: Mapping[str, bool],
    *,
    basin_means: Dict[str, np.ndarray] | None = None,
    bathy_depth: np.ndarray | None = None,
    use_mask_channels: bool = False,
) -> np.ndarray:
    """Scalars (harmonics + basin + bathy) + flattened satellite patch columns.

    When ``use_mask_channels`` is true, each variable block is pixel-interleaved:
    ``val_0, mask_0, val_1, mask_1, ...`` (SSS, then SST, then SSH). Otherwise
    only value columns are emitted (NaN → 0).
    """
    cols = []
    day = juld % 365

    if input_params.get("timecos"):
        cols.append(np.cos(2 * np.pi * day / 365))
    if input_params.get("timesin"):
        cols.append(np.sin(2 * np.pi * day / 365))
    if input_params.get("latcos"):
        cols.append(np.cos(2 * np.pi * lat / 180))
    if input_params.get("latsin"):
        cols.append(np.sin(2 * np.pi * lat / 180))
    if input_params.get("loncos"):
        cols.append(np.cos(2 * np.pi * lon / 360))
    if input_params.get("lonsin"):
        cols.append(np.sin(2 * np.pi * lon / 360))

    basin_means = basin_means or {}
    if input_params.get("basin_sss"):
        cols.append(np.asarray(basin_means.get("sss", np.zeros(len(juld))), dtype=np.float32))
    if input_params.get("basin_sst"):
        sst_mean = np.asarray(basin_means.get("sst", np.zeros(len(juld))), dtype=np.float64)
        cols.append((sst_mean - 273.15).astype(np.float32))
    if input_params.get("basin_ssh"):
        cols.append(np.asarray(basin_means.get("ssh", np.zeros(len(juld))), dtype=np.float32))
    if input_params.get("bathy_depth"):
        if bathy_depth is None:
            raise ValueError("bathy_depth requested but not provided")
        cols.append(np.asarray(bathy_depth, dtype=np.float32))

    for key in SAT_KEYS:
        if not input_params.get(key):
            continue
        if key not in sat_vars:
            raise KeyError(f"Missing satellite variable '{key}' in sat_vars")
        val = sat_vars[key]
        if key == "sst" and val.ndim == 1:
            val = val - 273.15
        elif key == "sst":
            val = val - 273.15
        if val.ndim > 1:
            finite = np.isfinite(val)
            val = np.where(finite, val, 0.0).astype(np.float32)
            if use_mask_channels:
                mask = finite.astype(np.float32)
                for j in range(val.shape[1]):
                    cols.append(val[:, j])
                    cols.append(mask[:, j])
            else:
                for j in range(val.shape[1]):
                    cols.append(val[:, j])
        elif val.ndim == 1:
            cols.append(val)
        else:
            for j in range(val.shape[1]):
                cols.append(val[:, j])

    if not cols:
        raise ValueError("No input features selected in input_params")
    stacked = np.column_stack(cols).astype(np.float32)
    return np.nan_to_num(stacked, nan=0.0)


def _reshape_sat_patch_block(
    flat: np.ndarray,
    spatial_pad: int,
    temporal_pad: int,
) -> np.ndarray:
    """``(N, T, H, W)`` from flattened satellite variable block."""
    t_win = temporal_pad + 1
    h = w = 2 * spatial_pad + 1
    n = flat.shape[0]
    return flat.reshape(n, t_win, h, w).astype(np.float32)


def _flatten_sat_patch_block(vol: np.ndarray) -> np.ndarray:
    return vol.reshape(vol.shape[0], -1).astype(np.float32)


def _center_value_at_latest(vol: np.ndarray, spatial_pad: int) -> np.ndarray:
    """Scalar center pixel at latest time, shape ``(N,)``."""
    cy = cx = spatial_pad
    return vol[:, -1, cy, cx].astype(np.float32)


def _make_center_relative(vol: np.ndarray, spatial_pad: int) -> np.ndarray:
    """Subtract latest-time center scalar from every pixel (center == 0 exactly)."""
    center = _center_value_at_latest(vol, spatial_pad)[:, np.newaxis, np.newaxis, np.newaxis]
    return (vol - center).astype(np.float32)


def _spatial_gradients_latest(vol: np.ndarray, spatial_pad: int) -> tuple[np.ndarray, np.ndarray]:
    """Spatial gradients on latest time slice, broadcast across T."""
    latest = vol[:, -1]
    gy, gx = np.gradient(latest, axis=(1, 2))
    t = vol.shape[1]
    gx_full = np.repeat(gx[:, np.newaxis, :, :], t, axis=1)
    gy_full = np.repeat(gy[:, np.newaxis, :, :], t, axis=1)
    return gx_full.astype(np.float32), gy_full.astype(np.float32)


def _spatial_laplacian_latest(vol: np.ndarray, spatial_pad: int) -> np.ndarray:
    latest = vol[:, -1]
    gy, gx = np.gradient(latest, axis=(1, 2))
    _, gxx = np.gradient(gx, axis=(1, 2))
    gyy, _ = np.gradient(gy, axis=(1, 2))
    lap = gxx + gyy
    t = vol.shape[1]
    return np.repeat(lap[:, np.newaxis, :, :], t, axis=1).astype(np.float32)


def _temporal_tendency(vol: np.ndarray) -> np.ndarray:
    """Last minus previous time slice; earliest step uses zero tendency."""
    out = np.zeros_like(vol, dtype=np.float32)
    if vol.shape[1] > 1:
        out[:, -1] = vol[:, -1] - vol[:, -2]
    return out


def _prepare_sat_volumes(
    sat_vars: Dict[str, np.ndarray],
    input_params: Mapping[str, bool],
    spatial_pad: int,
    temporal_pad: int,
) -> Dict[str, np.ndarray]:
    """Normalize units and reshape satellite variables to ``(N, T, H, W)``."""
    out: Dict[str, np.ndarray] = {}
    for key in SAT_KEYS:
        if not input_params.get(key):
            continue
        if key not in sat_vars:
            raise KeyError(f"Missing satellite variable '{key}' in sat_vars")
        val = np.asarray(sat_vars[key], dtype=np.float32)
        if key == "sst":
            val = val - 273.15
        if val.ndim == 1:
            raise ValueError(f"Residual model requires patch data for {key}, got scalars")
        finite = np.isfinite(val)
        val = np.where(finite, val, 0.0).astype(np.float32)
        out[key] = _reshape_sat_patch_block(val, spatial_pad, temporal_pad)
    return out


def _engineered_patch_planes(
    volumes: Dict[str, np.ndarray],
    input_params: Mapping[str, bool],
    spatial_pad: int,
) -> list[np.ndarray]:
  planes: list[np.ndarray] = []
  ssh = volumes.get("ssh")
  sst = volumes.get("sst")
  if input_params.get("patch_ssh_gradient") and ssh is not None:
    gx, gy = _spatial_gradients_latest(ssh, spatial_pad)
    planes.extend([gx, gy])
  if input_params.get("patch_ssh_laplacian") and ssh is not None:
    planes.append(_spatial_laplacian_latest(ssh, spatial_pad))
  if input_params.get("patch_temporal_tendency"):
    for key in SAT_KEYS:
      if key in volumes:
        planes.append(_temporal_tendency(volumes[key]))
  if input_params.get("patch_sst_gradient") and sst is not None:
    gx, gy = _spatial_gradients_latest(sst, spatial_pad)
    planes.extend([gx, gy])
  return planes


def build_residual_feature_names(
    input_params: Mapping[str, bool],
    spatial_pad: int,
    temporal_pad: int,
    *,
    include_bathy_scalar: bool = False,
    center_relative: bool = True,
) -> list[str]:
    names: list[str] = []
    if input_params.get("timecos"):
        names.append("timecos")
    if input_params.get("timesin"):
        names.append("timesin")
    if input_params.get("latcos"):
        names.append("latcos")
    if input_params.get("latsin"):
        names.append("latsin")
    if input_params.get("loncos"):
        names.append("loncos")
    if input_params.get("lonsin"):
        names.append("lonsin")
    if input_params.get("center_sss"):
        names.append("center_sss")
    if input_params.get("center_sst"):
        names.append("center_sst")
    if input_params.get("center_ssh"):
        names.append("center_ssh")
    if include_bathy_scalar and input_params.get("bathy_depth"):
        names.append("bathy_depth")
    per_var = sat_features_per_var(spatial_pad, temporal_pad)
    prefix = "rel_" if center_relative else "raw_"
    for key in SAT_KEYS:
        if not input_params.get(key):
            continue
        for i in range(per_var):
            names.append(f"{prefix}{key}_{i}")
    if input_params.get("patch_ssh_gradient"):
        for i in range(per_var):
            names.extend([f"ssh_grad_x_{i}", f"ssh_grad_y_{i}"])
    if input_params.get("patch_ssh_laplacian"):
        for i in range(per_var):
            names.append(f"ssh_laplacian_{i}")
    if input_params.get("patch_temporal_tendency"):
        for key in SAT_KEYS:
            if input_params.get(key):
                for i in range(per_var):
                    names.append(f"{key}_dt_{i}")
    if input_params.get("patch_sst_gradient"):
        for i in range(per_var):
            names.extend([f"sst_grad_x_{i}", f"sst_grad_y_{i}"])
    return names


def build_argo_residual_input_matrix(
    juld: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    sat_vars: Dict[str, np.ndarray],
    input_params: Mapping[str, bool],
    *,
    spatial_pad: int = 2,
    temporal_pad: int = 6,
    bathy_depth: np.ndarray | None = None,
    center_relative: bool = True,
    include_bathy_scalar: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Base scalars + center-relative patch block (+ optional engineered planes)."""
    cols: list[np.ndarray] = []
    day = juld % 365

    if input_params.get("timecos"):
        cols.append(np.cos(2 * np.pi * day / 365))
    if input_params.get("timesin"):
        cols.append(np.sin(2 * np.pi * day / 365))
    if input_params.get("latcos"):
        cols.append(np.cos(2 * np.pi * lat / 180))
    if input_params.get("latsin"):
        cols.append(np.sin(2 * np.pi * lat / 180))
    if input_params.get("loncos"):
        cols.append(np.cos(2 * np.pi * lon / 360))
    if input_params.get("lonsin"):
        cols.append(np.sin(2 * np.pi * lon / 360))

    volumes = _prepare_sat_volumes(sat_vars, input_params, spatial_pad, temporal_pad)
    if input_params.get("center_sss") and "sss" in volumes:
        cols.append(_center_value_at_latest(volumes["sss"], spatial_pad))
    if input_params.get("center_sst") and "sst" in volumes:
        cols.append(_center_value_at_latest(volumes["sst"], spatial_pad))
    if input_params.get("center_ssh") and "ssh" in volumes:
        cols.append(_center_value_at_latest(volumes["ssh"], spatial_pad))

    if include_bathy_scalar and input_params.get("bathy_depth"):
        if bathy_depth is None:
            raise ValueError("bathy_depth requested but not provided")
        cols.append(np.asarray(bathy_depth, dtype=np.float32))

    base_dim = len(cols)
    patch_offset = base_dim
    patch_planes: list[np.ndarray] = []
    for key in SAT_KEYS:
        if key not in volumes:
            continue
        vol = volumes[key]
        if center_relative:
            vol = _make_center_relative(vol, spatial_pad)
        patch_planes.append(vol)
    patch_planes.extend(_engineered_patch_planes(volumes, input_params, spatial_pad))

    for plane in patch_planes:
        cols.append(_flatten_sat_patch_block(plane))

    if not cols:
        raise ValueError("No input features selected in input_params")
    stacked = np.column_stack(cols).astype(np.float32)
    meta = {
        "base_dim": int(base_dim),
        "patch_offset": int(patch_offset),
        "center_relative": bool(center_relative),
        "include_bathy_scalar": bool(include_bathy_scalar),
    }
    return np.nan_to_num(stacked, nan=0.0), meta


def apply_train_standardization(
    inputs: np.ndarray,
    juld: np.ndarray,
    config: Mapping[str, Any],
    feature_names: list[str] | None = None,
    *,
    dataset_tag: str,
    v2_src: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Z-score inputs using train-split statistics only."""
    from base.split_utils import build_split_indices

    dl_cfg = dict((config.get("data_loader") or {}).get("args") or {})
    n = inputs.shape[0]
    splits = build_split_indices(
        n,
        juld,
        dl_cfg,
        dataset_tag=dataset_tag,
        v2_src=v2_src,
    )
    tr = np.asarray(splits["train"], dtype=int)
    X = np.asarray(inputs, dtype=np.float32)
    mu = X[tr].mean(axis=0)
    sd = X[tr].std(axis=0)
    sd = np.maximum(sd, 1e-6)
    standardized = ((X - mu) / sd).astype(np.float32)
    meta = {
        "mean": mu.astype(np.float32),
        "std": sd.astype(np.float32),
        "feature_names": list(feature_names or []),
        "normalization_version": NORMALIZATION_VERSION,
        "split": "train_chronological",
        "train_indices": tr,
    }
    return standardized, meta


def build_error_input_block(
    err_raw: dict[str, np.ndarray],
    juld: np.ndarray,
    config: Mapping[str, Any],
    *,
    dataset_tag: str,
    v2_src: str | None = None,
    e0: float = ERROR_NORM_E0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Log-zscore error channels + missing masks (Phase 2.2).

    Returns ``(inputs_err, err_missing, meta)`` aligned on stations.
    ``err_raw`` maps channel name → (N,) or (N, F) float arrays (NaN = missing).
    """
    from base.split_utils import build_split_indices

    if not err_raw:
        raise ValueError("err_raw must be non-empty")
    names = list(err_raw.keys())
    cols = []
    miss_cols = []
    for name in names:
        arr = np.asarray(err_raw[name], dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[:, None]
        miss = ~np.isfinite(arr)
        miss_cols.append(miss.astype(np.float32))
        cols.append(arr)
    raw = np.concatenate(cols, axis=1)
    missing = np.concatenate(miss_cols, axis=1)

    dl_cfg = dict((config.get("data_loader") or {}).get("args") or {})
    n = raw.shape[0]
    splits = build_split_indices(
        n, juld, dl_cfg, dataset_tag=dataset_tag, v2_src=v2_src
    )
    tr = np.asarray(splits["train"], dtype=int)
    fill = np.nanpercentile(raw[tr], 90, axis=0)
    fill = np.where(np.isfinite(fill), fill, e0)
    filled = np.where(np.isfinite(raw), raw, fill[None, :])
    log_e = np.log(np.maximum(filled, 0.0) + e0)
    mu = log_e[tr].mean(axis=0)
    sd = np.maximum(log_e[tr].std(axis=0), 1e-6)
    inputs_err = ((log_e - mu) / sd).astype(np.float32)
    meta = {
        "input_error_channels": names,
        "mean_log": mu.astype(np.float32),
        "std_log": sd.astype(np.float32),
        "fill_p90": fill.astype(np.float32),
        "e0": float(e0),
        "normalization": "log_zscore_train",
        "train_indices": tr,
        "feature_widths": {
            name: int(np.asarray(err_raw[name]).reshape(n, -1).shape[1]) for name in names
        },
    }
    return inputs_err, missing.astype(np.float32), meta


def residual_sat_patch_shape(
    input_params: Mapping[str, bool],
    spatial_pad: int,
    temporal_pad: int,
    *,
    use_mask_channels: bool = False,
) -> tuple[int, int, int, int]:
    """``(C, T, H, W)`` for residual patch encoder."""
    h = w = 2 * spatial_pad + 1
    t = temporal_pad + 1
    c = count_residual_patch_channels(input_params, use_mask_channels=use_mask_channels)
    return (c, t, h, w)


def build_input_matrix(
    juld: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    sat_vars: Dict[str, np.ndarray],
    input_params: Mapping[str, bool],
) -> np.ndarray:
    """Vectorized v2 ``prepare_inputs`` (offset already applied in ``sat_vars``)."""
    cols = []
    day = juld % 365

    if input_params.get("timecos"):
        cols.append(np.cos(2 * np.pi * day / 365))
    if input_params.get("timesin"):
        cols.append(np.sin(2 * np.pi * day / 365))
    if input_params.get("latcos"):
        cols.append(np.cos(2 * np.pi * lat / 180))
    if input_params.get("latsin"):
        cols.append(np.sin(2 * np.pi * lat / 180))
    if input_params.get("loncos"):
        cols.append(np.cos(2 * np.pi * lon / 360))
    if input_params.get("lonsin"):
        cols.append(np.sin(2 * np.pi * lon / 360))

    for key in ("sss", "sst", "ssh"):
        if not input_params.get(key):
            continue
        if key not in sat_vars:
            raise KeyError(f"Missing satellite variable '{key}' in groups/io config")
        val = sat_vars[key]
        if key == "sst":
            val = val - 273.15
        if val.ndim == 1:
            cols.append(val)
        else:
            for j in range(val.shape[1]):
                cols.append(val[:, j])

    if not cols:
        raise ValueError("No input features selected in input_params")

    stacked = np.column_stack(cols).astype(np.float32)
    return np.nan_to_num(stacked, nan=0.0)


def _resolve_h5_paths(io_cfg: Dict) -> tuple[str, str]:
    data_path = io_cfg["data_path"]
    sat_file = io_cfg.get("sat_file", "satellite_*.h5")
    prof_file = io_cfg.get("prof_file", "profiles_*.h5")
    if "*" in sat_file:
        import glob

        matches = sorted(glob.glob(os.path.join(data_path, sat_file)))
        if not matches:
            raise FileNotFoundError(f"No satellite file matching {sat_file} in {data_path}")
        sat_path = matches[0]
    else:
        sat_path = os.path.join(data_path, sat_file)
    prof_path = os.path.join(data_path, prof_file)
    return sat_path, prof_path


def build_train_cache(config: Dict, force: bool = False) -> str:
    """
    Read HDF5 once, build inputs + per-output PCA, pickle train-ready cache.

    Returns path to the pickle file.
    """
    io_cfg = config["io"]
    spatial_pad = int(io_cfg.get("spatial_pad", 0))
    temporal_pad = int(io_cfg.get("temporal_pad", 0))
    outputs = OrderedDict(config["outputs"])
    input_params = config["input_params"]
    groups = io_cfg.get("groups", {
        "sss": {"h5_group": "sss", "h5_var": "sos"},
        "sst": {"h5_group": "ostia", "h5_var": "analysed_sst"},
        "ssh": {"h5_group": "ssh", "h5_var": "adt"},
    })

    cache_dir = io_cfg.get("cache_dir", "data/cache")
    os.makedirs(cache_dir, exist_ok=True)
    chash = config_hash(config)
    cache_path = os.path.join(cache_dir, f"train_ready_{chash}.pkl")
    if os.path.exists(cache_path) and not force:
        return cache_path

    sat_path, prof_path = _resolve_h5_paths(io_cfg)  # trust boundary: paths from config only
    bbox = io_cfg.get("BBox")

    with h5py.File(prof_path, "r") as pf:
        model = pf["model"]
        lat = model["LATITUDE"][:]
        lon = model["LONGITUDE"][:]
        juld = model["JULD"][:]
        if bbox is not None:
            min_lat, max_lat, min_lon, max_lon = bbox
            mask = (lat >= min_lat) & (lat <= max_lat) & (lon >= min_lon) & (lon <= max_lon)
            station_indices = np.where(mask)[0]
        else:
            station_indices = np.arange(len(lat))

        lat = lat[station_indices]
        lon = lon[station_indices]
        juld = juld[station_indices]

        profile_data = {}
        for name in outputs:
            h5_var = OUTPUT_H5_VARS.get(name, name.upper())
            profile_data[name] = model[h5_var][station_indices].astype(np.float64)

        pres = model.get("DEPH")
        if pres is not None:
            pres = pres[0].astype(np.float32)
        else:
            pres = np.arange(profile_data[list(outputs.keys())[0]].shape[1], dtype=np.float32)

    sat_vars = {}
    with h5py.File(sat_path, "r") as sf:
        for key, spec in groups.items():
            if not input_params.get(key):
                continue
            d = sf[spec["h5_group"]][spec["h5_var"]][station_indices]
            sat_vars[key] = extract_sat_values(d, spatial_pad, temporal_pad)

    inputs = build_input_matrix(juld, lat, lon, sat_vars, input_params)

    pca_models = {}
    pcs_by_name = {}
    target_cols = []
    if not io_cfg.get("anomaly_targets"):
        for name, n_comp in outputs.items():
            profiles_fit = profile_data[name]
            profiles_fit = np.nan_to_num(profiles_fit, nan=0.0)
            pca = PCA(n_components=n_comp)
            pcs = pca.fit_transform(profiles_fit)
            pca_models[name] = pca
            pcs_by_name[name] = pcs.T
            target_cols.append(pcs.astype(np.float32))

        targets = np.hstack(target_cols).astype(np.float32)
        weights = get_pca_weights(pca_models, pcs_by_name, list(outputs.keys()))
    else:
        targets = None
        weights = None
    # ponytail: depth-major (n_z, N) matches v2 ``TEMP``/``SAL`` layout for eval scripts.
    # Keep NaN for missing/deep levels so eval RMSE can mask them; PCA fit above uses its
    # own nan_to_num. NEVER nan_to_num here — 0.0 fill contaminates profile-space RMSE.
    profiles = {
        name: np.ascontiguousarray(profile_data[name].T.astype(np.float32))
        for name in outputs
    }
    dataset_tag = io_cfg.get("dataset_tag", "isas20")
    n_sat = count_sat_vars(input_params)
    patch_shape = sat_patch_shape(spatial_pad, temporal_pad, n_sat)

    anom = None
    if io_cfg.get("anomaly_targets"):
        from preproc.climatology import build_anomaly_targets_block

        anom = build_anomaly_targets_block(
            profiles,
            lat.astype(np.float32),
            lon.astype(np.float32),
            juld.astype(np.float32),
            pres,
            outputs,
            config,
        )
        targets = anom["targets"]
        pca_models = anom["pca_models"]
        pcs_by_name = anom["pcs_by_name"]
        weights = anom["weights"]

    payload = {
        "inputs": inputs,
        "targets": targets,
        "pca_models": pca_models,
        "pcs_by_name": pcs_by_name,
        "outputs": outputs,
        "weights": weights,
        "profiles": profiles,
        "true_profiles": true_profiles_numpy(targets, pca_models, outputs),
        "LAT": lat.astype(np.float32),
        "LON": lon.astype(np.float32),
        "PRES": pres,
        "JULD": juld.astype(np.float32),
        "station_indices": station_indices,
        "input_params": dict(input_params),
        "spatial_pad": spatial_pad,
        "temporal_pad": temporal_pad,
        "sat_patch_shape": patch_shape,
        "patch_order": PATCH_ORDER_DOC,
        "config_hash": chash,
        "dataset_tag": dataset_tag,
        "min_depth": 0,
        "max_depth": int(profile_data[list(outputs.keys())[0]].shape[1] - 1),
    }
    if anom is not None:
        payload.update(
            {
                k: anom[k]
                for k in (
                    "climatology",
                    "clim_profiles",
                    "anomaly_targets",
                    "ssh_obs_adt",
                    "ssh_obs_sla",
                    "clim_steric",
                    "steric_calibration",
                )
                if k in anom
            }
        )

    return write_train_cache(payload, cache_dir, chash)


# --- legacy helpers (unchanged API) ---

def load_config(config_path: Optional[str] = None) -> Dict:
    if config_path is None:
        return {
            "data_path": "/unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project/data/NeSPReSO_v2_GoM_sat",
            "BBox": None,
            "input_params": {
                "bathymetry": ["elevation"],
                "ssh": ["adt", "sla", "ugos", "vgos"],
                "ostia": ["analysed_sst"],
                "sss": ["sos"],
                "wind": ["windspeed", "u_wind", "v_wind"],
            },
            "input_processing": "norm",
            "spatial_step": 1,
            "spatial_pad": 2,
            "temporal_pad": 3,
            "output_params": {"temperature": ["TEMP"], "salinity": ["PSAL"]},
            "output_processing": "norm",
        }

    with open(config_path, "r") as f:
        return json.load(f)


def normalize_ignore_nan(arr, axis=None):
    if axis is not None:
        n_valid = np.sum(~np.isnan(arr), axis=axis, keepdims=True)
    else:
        n_valid = np.sum(~np.isnan(arr))

    mean = np.nanmean(arr, axis=axis, keepdims=True)
    mean = np.where(n_valid > 0, mean, 0)
    std = np.nanstd(arr, axis=axis, keepdims=True)
    std = np.where(n_valid > 1, std, 1)
    normed = (arr - mean) / std
    return np.where(np.isnan(normed), 0, normed), mean, std


def get_bbox_mask(lat, lon, bbox):
    if bbox is None:
        return np.ones_like(lat, dtype=bool)
    min_lat, max_lat, min_lon, max_lon = bbox
    return (lat >= min_lat) & (lat <= max_lat) & (lon >= min_lon) & (lon <= max_lon)


def reshape_time(data, temporal_pad):
    N, T = data.shape[:2]
    return data[:, : temporal_pad + 1, ...]


def preprocess_data(config: Dict) -> Dict:
    """Legacy multi-dict preprocess (kept for exploratory notebooks)."""
    data_path = config["data_path"]
    sat_file = os.path.join(data_path, "satellite_NeSPReSO_v1_global.h5")
    prof_file = os.path.join(data_path, "profiles_NeSPReSO_v1_global.h5")

    with h5py.File(prof_file, "r") as pf:
        station_lat = pf["model"]["LATITUDE"][:]
        station_lon = pf["model"]["LONGITUDE"][:]
        bbox_mask = get_bbox_mask(station_lat, station_lon, config["BBox"])
        station_indices = np.where(bbox_mask)[0]

    processed = {"inputs": {}, "outputs": {}, "masks": {}, "station_indices": station_indices, "stats": {}}

    with h5py.File(sat_file, "r") as sf:
        for group, vars in config["input_params"].items():
            g = sf[group]
            for var in vars:
                d = g[var][:]
                d = d[station_indices]
                mask = np.isnan(d)
                if config["input_processing"] == "norm":
                    d, input_mean, input_std = normalize_ignore_nan(d)
                    processed["stats"][f"{group}_{var}_mean"] = input_mean
                    processed["stats"][f"{group}_{var}_std"] = input_std
                if d.ndim >= 3 and d.shape[1] >= config["temporal_pad"]:
                    d = reshape_time(d, config["temporal_pad"])
                    mask = reshape_time(mask, config["temporal_pad"])
                processed["inputs"][f"{group}_{var}"] = d
                processed["masks"][f"{group}_{var}"] = mask

    with h5py.File(prof_file, "r") as pf:
        for group, vars in config["output_params"].items():
            g = pf["model"]
            for var in vars:
                d = g[var][:]
                d = d[station_indices]
                mask = np.isnan(d)
                if config["output_processing"] == "norm":
                    d, output_mean, output_std = normalize_ignore_nan(d, axis=0)
                    processed["stats"][f"{group}_{var}_mean"] = output_mean
                    processed["stats"][f"{group}_{var}_std"] = output_std
                processed["outputs"][f"{group}_{var}"] = d
                processed["masks"][f"{group}_{var}"] = mask

    return processed


def save_processed_data(processed: Dict, save_path: str) -> None:
    import torch

    tensor_processed = {}
    for key, value in processed.items():
        if key == "station_indices":
            tensor_processed[key] = torch.tensor(value)
        elif isinstance(value, dict):
            tensor_processed[key] = {
                k: torch.tensor(v) if isinstance(v, np.ndarray) else v for k, v in value.items()
            }
        elif isinstance(value, np.ndarray):
            tensor_processed[key] = torch.tensor(value)
        else:
            tensor_processed[key] = value

    with open(save_path, "wb") as f:
        pickle.dump(tensor_processed, f)
    print(f"Preprocessed data saved to {save_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "cache":
        from parse_config import read_json, validate_config

        cfg_path = sys.argv[2] if len(sys.argv) > 2 else "config/isas/config_isas.json"
        cfg = read_json(cfg_path)
        validate_config(cfg)
        build_train_cache(cfg, force="--force" in sys.argv)
    else:
        config_path = sys.argv[1] if len(sys.argv) > 1 else None
        config = load_config(config_path)
        save_path = os.path.join(config["data_path"], "preprocessed_satellite_data.pkl")
        save_processed_data(preprocess_data(config), save_path)
