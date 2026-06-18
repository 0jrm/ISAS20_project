"""Flatten mask-native L3 bundles into PatchConvMLP patch inputs."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from preproc.l3_rasterize import FEATURE_NAMES, IDX_MASK, empty_bundle, l3_geometry
from preproc.preproc_isas_sat import PATCH_ORDER_DOC, count_encoding_dims

# ponytail: fixed transforms; upgrade to fit stats from train split
AGE_SCALE_HOURS = 336.0


def l3_variable_names(l3_cfg: Mapping[str, Any]) -> list[str]:
    return list(l3_cfg["variables"].keys())


def l3_feature_names(l3_cfg: Mapping[str, Any]) -> list[str]:
    return list(l3_cfg.get("features", FEATURE_NAMES))


def l3_n_channels(l3_cfg: Mapping[str, Any]) -> int:
    return len(l3_variable_names(l3_cfg)) * len(l3_feature_names(l3_cfg))


def l3_sat_patch_shape(l3_cfg: Mapping[str, Any]) -> tuple[int, int, int, int]:
    """``(C, T, H, W)`` for PatchConvMLP when L3 enabled."""
    spatial_pad, temporal_pad, _ = l3_geometry(l3_cfg)
    t = temporal_pad + 1
    h = w = 2 * spatial_pad + 1
    return (l3_n_channels(l3_cfg), t, h, w)


def compute_l3_input_dim(
    input_params: Mapping[str, bool],
    l3_cfg: Mapping[str, Any],
) -> int:
    c, t, h, w = l3_sat_patch_shape(l3_cfg)
    return count_encoding_dims(input_params) + c * t * h * w


def _feature_index(name: str) -> int:
    return FEATURE_NAMES.index(name)


def normalize_feature_plane(name: str, plane: np.ndarray) -> np.ndarray:
    """Per-feature transforms; masks stay binary."""
    out = np.asarray(plane, dtype=np.float32).copy()
    if name == "mask":
        return (out > 0).astype(np.float32)
    if name == "age":
        return np.clip(out / AGE_SCALE_HOURS, 0.0, 1.0).astype(np.float32)
    if name == "uncertainty":
        return np.nan_to_num(out, nan=0.0).astype(np.float32)
    if name == "count":
        return np.log1p(np.maximum(out, 0.0)).astype(np.float32)
    return out


def bundle_to_channels(bundle: np.ndarray, feature_names: Sequence[str]) -> np.ndarray:
    """``(len(features), T, H, W)`` from full bundle ``(5, T, H, W)``."""
    planes = [normalize_feature_plane(f, bundle[_feature_index(f)]) for f in feature_names]
    return np.stack(planes, axis=0).astype(np.float32)


def flatten_l3_tensors(
    tensors: Mapping[str, np.ndarray] | None,
    l3_cfg: Mapping[str, Any],
) -> np.ndarray:
    """Flatten to ``C*T*H*W`` row vector (time-major within each channel)."""
    vars_ = l3_variable_names(l3_cfg)
    feats = l3_feature_names(l3_cfg)
    c, t, h, w = l3_sat_patch_shape(l3_cfg)
    if tensors is None:
        stack = np.zeros((c, t, h, w), dtype=np.float32)
    else:
        channels = []
        for var in vars_:
            bundle = tensors.get(var)
            if bundle is None:
                bundle = empty_bundle(t, h, w)
            channels.append(bundle_to_channels(bundle, feats))
        stack = np.concatenate(channels, axis=0)
    assert stack.shape == (c, t, h, w), stack.shape
    return stack.reshape(-1)


def build_l3_input_rows(
    base_inputs: np.ndarray,
    l3_tensors: Sequence[Mapping[str, np.ndarray] | None],
    l3_cfg: Mapping[str, Any],
    *,
    n_enc: int,
) -> np.ndarray:
    enc = np.asarray(base_inputs[:, :n_enc], dtype=np.float32)
    sat = np.stack([flatten_l3_tensors(t, l3_cfg) for t in l3_tensors], axis=0)
    return np.hstack([enc, sat]).astype(np.float32)


def l3_channel_metadata(l3_cfg: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "patch_order": PATCH_ORDER_DOC,
        "variables": l3_variable_names(l3_cfg),
        "features": l3_feature_names(l3_cfg),
        "patch_shape": list(l3_sat_patch_shape(l3_cfg)),
        "age_scale_hours": AGE_SCALE_HOURS,
    }
