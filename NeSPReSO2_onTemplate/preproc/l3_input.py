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
    vars_ = l3_variable_names(l3_cfg)
    feats = l3_feature_names(l3_cfg)
    flat = [f"{var}/{feat}" for var in vars_ for feat in feats]
    return {
        "patch_order": PATCH_ORDER_DOC,
        "variables": vars_,
        "features": feats,
        "flat_channel_names": flat,
        "patch_shape": list(l3_sat_patch_shape(l3_cfg)),
        "age_scale_hours": AGE_SCALE_HOURS,
    }


def verify_l3_cache_layout(cache: Mapping[str, Any], l3_cfg: Mapping[str, Any]) -> None:
    """Fail fast when cache channel layout disagrees with config (train/eval honesty)."""
    if not cache.get("l3_enabled"):
        return
    meta = cache.get("l3_channel_metadata") or {}
    expected = l3_channel_metadata(l3_cfg)
    for key in ("variables", "features", "flat_channel_names", "patch_shape"):
        if meta.get(key) != expected.get(key):
            raise ValueError(
                f"L3 cache layout mismatch on {key!r}: cache={meta.get(key)!r} config={expected.get(key)!r}"
            )
    input_params = cache.get("input_params") or {}
    expected_dim = compute_l3_input_dim(input_params, l3_cfg)
    actual_dim = int(np.asarray(cache["inputs"]).shape[1])
    if actual_dim != expected_dim:
        raise ValueError(f"L3 cache input dim {actual_dim} != expected {expected_dim}")


def sync_arch_with_io(config: Mapping[str, Any]) -> int:
    """Align ``arch.args`` with io path (L3 mask-native or legacy L4/point). Returns expected input dim."""
    io_cfg = config.get("io", {}) or {}
    l3_cfg = io_cfg.get("l3") or {}
    input_params = config.get("input_params", {}) or {}
    arch = config["arch"]
    arch_args = arch.setdefault("args", {})

    if l3_cfg.get("enabled"):
        expected_dim = compute_l3_input_dim(input_params, l3_cfg)
        arch_args["n_enc"] = count_encoding_dims(input_params)
        c, t, h, w = l3_sat_patch_shape(l3_cfg)
        arch_args["n_sat"] = l3_n_channels(l3_cfg)
        arch_args["patch_shape"] = [c, t, h, w]
    elif arch.get("type") == "FieldUNet":
        field_cfg = io_cfg.get("field") or {}
        n_static = 3
        n_time = 4
        arch_args["in_channels"] = int(field_cfg.get("in_channels", n_static + n_time))
        arch_args["out_channels"] = sum(config["outputs"].values())
        arch_args["base_width"] = int(field_cfg.get("base_width", 32))
        expected_dim = arch_args["in_channels"]
    elif io_cfg.get("cache_kind") == "point_cube":
        expected_dim = int(arch_args.get("input_dim", 9))
        arch_args["n_enc"] = int(arch_args.get("n_enc", 6))
        arch_args["n_sat"] = int(arch_args.get("n_sat", 3))
        arch_args["patch_shape"] = None
    elif arch.get("type") in ("PatchConvMLP", "PatchMaskConvMLP"):
        from preproc.preproc_isas_sat import (
            compute_input_dim,
            count_patch_channels,
            count_scalar_dims,
            resolve_use_mask_channels,
            sat_patch_shape,
        )

        spatial_pad = int(io_cfg.get("spatial_pad", 0))
        temporal_pad = int(io_cfg.get("temporal_pad", 0))
        use_mask_channels = resolve_use_mask_channels(config)
        arch_args["use_mask_channels"] = use_mask_channels
        expected_dim = compute_input_dim(
            input_params, spatial_pad, temporal_pad, use_mask_channels=use_mask_channels
        )
        arch_args["n_enc"] = count_scalar_dims(input_params)
        n_ch = count_patch_channels(input_params, use_mask_channels=use_mask_channels)
        patch_shape = sat_patch_shape(
            spatial_pad,
            temporal_pad,
            3,
            input_params=input_params,
            use_mask_channels=use_mask_channels,
        )
        arch_args["n_sat"] = n_ch if patch_shape else count_patch_channels(
            input_params, use_mask_channels=use_mask_channels
        )
        arch_args["patch_shape"] = list(patch_shape) if patch_shape else None
    elif arch.get("type") == "ResidualPatchModel":
        from preproc.preproc_isas_sat import (
            compute_argo_residual_input_dim,
            count_base_dims,
            count_encoding_dims,
            residual_sat_patch_shape,
            resolve_use_mask_channels,
        )

        spatial_pad = int(io_cfg.get("spatial_pad", 0))
        temporal_pad = int(io_cfg.get("temporal_pad", 0))
        use_mask_channels = resolve_use_mask_channels(config)
        include_bathy_scalar = bool(io_cfg.get("include_bathy_scalar", False))
        expected_dim = compute_argo_residual_input_dim(
            input_params,
            spatial_pad,
            temporal_pad,
            use_mask_channels=use_mask_channels,
            include_bathy_scalar=include_bathy_scalar,
        )
        patch_shape = residual_sat_patch_shape(
            input_params,
            spatial_pad,
            temporal_pad,
            use_mask_channels=use_mask_channels,
        )
        base_dim = int(arch_args.get("base_dim", 0)) or (
            count_base_dims(input_params)
            + (1 if include_bathy_scalar and input_params.get("bathy_depth") else 0)
        )
        arch_args["base_dim"] = base_dim
        arch_args["patch_offset"] = base_dim
        arch_args["n_enc"] = count_encoding_dims(input_params)
        arch_args["n_sat"] = 3
        arch_args["patch_shape"] = list(patch_shape)
    elif arch.get("type") == "PointAnchoredResidual":
        features = config.get("features") or {}
        from preproc.features.sampler import expand_feature_names

        feat_dim = len(expand_feature_names(features))
        base_dim = int(arch_args.get("base_dim", 9))
        arch_args["base_dim"] = base_dim
        arch_args["feat_dim"] = feat_dim
        arch_args["feat_offset"] = base_dim
        arch_args["n_enc"] = 6
        arch_args["n_sat"] = 3
        expected_dim = base_dim + feat_dim
    else:
        from preproc.preproc_isas_sat import compute_input_dim

        spatial_pad = int(io_cfg.get("spatial_pad", 0))
        temporal_pad = int(io_cfg.get("temporal_pad", 0))
        expected_dim = compute_input_dim(input_params, spatial_pad, temporal_pad)

    arch_args["input_dim"] = expected_dim
    arch_args["output_dim"] = sum(config["outputs"].values())
    return expected_dim
