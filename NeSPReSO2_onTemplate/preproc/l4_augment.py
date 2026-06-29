"""L4 auxiliary / mask-simulation augmentation (Phase 4).

L4 fields may only enter training when explicitly labeled synthetic or auxiliary.
ponytail: single-pixel noise + mask replay; upgrade to spatially correlated noise.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any, Mapping

import numpy as np

from preproc.l3_rasterize import (
    IDX_AGE,
    IDX_COUNT,
    IDX_MASK,
    IDX_UNC,
    IDX_VALUE,
    PatchGrid,
    TargetPoint,
    empty_bundle,
)

SOURCE_REAL_L3 = 0
SOURCE_SYNTH_L4_MASKED = 1
SOURCE_L4_AUX = 2
SOURCE_MISSING = 3

VALID_L4_MODES = ("mask_augment",)


class SourceFlag(IntEnum):
    REAL_L3 = SOURCE_REAL_L3
    SYNTH_L4_MASKED = SOURCE_SYNTH_L4_MASKED
    L4_AUX = SOURCE_L4_AUX
    MISSING = SOURCE_MISSING


def apply_l4_mask_augment(
    l4_field: np.ndarray,
    observation_mask: np.ndarray,
    *,
    noise_scale: float = 1.0,
    err_field: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Mask L4 with real observation geometry; return (values, mask, source_flag)."""
    rng = rng or np.random.default_rng()
    field = np.asarray(l4_field, dtype=np.float64)
    mask = (np.asarray(observation_mask) > 0).astype(np.float32)
    if err_field is not None:
        sigma = np.maximum(np.asarray(err_field, dtype=np.float64), 1e-6)
        noise = rng.normal(0.0, noise_scale * sigma)
    else:
        noise = rng.normal(0.0, noise_scale, size=field.shape)
    values = np.where(mask > 0, field + noise, 0.0).astype(np.float32)
    flag = int(SourceFlag.SYNTH_L4_MASKED)
    return values, mask, flag


def apply_mask_dropouts(
    mask: np.ndarray,
    l4_cfg: Mapping[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    """Mission / time-window dropout on an observation mask (T,H,W) or (H,W)."""
    out = np.asarray(mask, dtype=np.float32).copy()
    if float(l4_cfg.get("mission_dropout_prob", 0.0)) > 0 and rng.random() < float(l4_cfg["mission_dropout_prob"]):
        out[...] = 0.0
        return out
    tw = float(l4_cfg.get("time_window_dropout_prob", 0.0))
    if tw <= 0 or out.ndim < 3:
        return out
    for t in range(out.shape[0]):
        if rng.random() < tw:
            out[t] = 0.0
    return out


def observation_mask_from_bundle(bundle: np.ndarray) -> np.ndarray:
    """Real L3 mask library plane(s) from a feature bundle."""
    return (np.asarray(bundle[IDX_MASK], dtype=np.float32) > 0).astype(np.float32)


def augment_bundle_from_l4(
    l3_mask_bundle: np.ndarray,
    l4_values: np.ndarray,
    l4_errors: np.ndarray | None,
    l4_cfg: Mapping[str, Any],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Build synthetic sparse bundle from L4 + real L3 mask geometry.

    Returns (bundle, source_flag_plane) with source_flag_plane shape (T,H,W).
    """
    l3_mask_bundle = np.asarray(l3_mask_bundle, dtype=np.float32)
    l4_values = np.asarray(l4_values, dtype=np.float64)
    n_time, height, width = l4_values.shape
    out = empty_bundle(n_time, height, width)
    source = np.full((n_time, height, width), SOURCE_MISSING, dtype=np.float32)
    noise_scale = float(l4_cfg.get("noise_scale", 1.0))
    mask_lib = observation_mask_from_bundle(l3_mask_bundle)
    mask_lib = apply_mask_dropouts(mask_lib, l4_cfg, rng)

    for t in range(n_time):
        err_t = l4_errors[t] if l4_errors is not None else None
        vals, m, flag = apply_l4_mask_augment(
            l4_values[t],
            mask_lib[t] if mask_lib.ndim == 3 else mask_lib,
            noise_scale=noise_scale,
            err_field=err_t,
            rng=rng,
        )
        out[IDX_VALUE, t] = vals
        out[IDX_MASK, t] = m
        out[IDX_AGE, t] = l3_mask_bundle[IDX_AGE, t]
        out[IDX_COUNT, t] = m
        if err_t is not None:
            out[IDX_UNC, t] = np.where(m > 0, np.nan_to_num(err_t, nan=0.0), np.nan).astype(np.float32)
        source[t] = np.where(m > 0, float(flag), float(SOURCE_MISSING))
    return out, source


def apply_l4_to_sample(
    sample: dict[str, Any],
    *,
    target: TargetPoint,
    grid: PatchGrid,
    raw_root: Any,
    windows_hours: list[float],
    l4_cfg: Mapping[str, Any],
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Apply configured L4 mask augmentation to an L3 processed sample."""
    from pathlib import Path

    from preproc.l4_rasterize import collect_l4_ssh_paths, sample_l4_ssh_patch

    mode = str(l4_cfg.get("mode", "mask_augment"))
    if mode not in VALID_L4_MODES:
        raise ValueError(f"io.l4.mode must be one of {VALID_L4_MODES}, got {mode!r}")

    raw_root = Path(raw_root)
    apply_vars = list(l4_cfg.get("apply_to", ("ssh",)))
    source_flags: dict[str, Any] = {}
    l4_sources: dict[str, list[str]] = {}

    for var in apply_vars:
        if var not in sample:
            continue
        mask_bundle = np.asarray(sample[var], dtype=np.float32)
        if var == "ssh":
            if raw_root.is_dir():
                paths = collect_l4_ssh_paths(raw_root, target.time, windows_hours)
                l4_sources[var] = [str(p) for p in paths]
                l4_vals, l4_errs = sample_l4_ssh_patch(paths, target, grid, windows_hours)
            else:
                l4_sources[var] = []
                t, h, w = mask_bundle.shape[1:]
                l4_vals = np.zeros((t, h, w), dtype=np.float64)
                l4_errs = None
            aug_bundle, src_plane = augment_bundle_from_l4(mask_bundle, l4_vals, l4_errs, l4_cfg, rng)
            sample[var] = aug_bundle
            source_flags[var] = src_plane
        # ponytail: wind/sst L4 paths deferred

    sample["l4_augment"] = {
        "mode": mode,
        "apply_to": apply_vars,
        "source_flags": source_flags,
        "config": augment_config_summary(dict(l4_cfg)),
    }
    sample["sources"]["l4_files"] = l4_sources
    return sample


def augment_config_summary(l4_cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "enabled": bool(l4_cfg.get("enabled", False)),
        "mode": str(l4_cfg.get("mode", "mask_augment")),
        "apply_to": list(l4_cfg.get("apply_to", ("ssh",))),
        "noise_scale": float(l4_cfg.get("noise_scale", 1.0)),
        "mission_dropout_prob": float(l4_cfg.get("mission_dropout_prob", 0.0)),
        "time_window_dropout_prob": float(l4_cfg.get("time_window_dropout_prob", 0.0)),
    }
