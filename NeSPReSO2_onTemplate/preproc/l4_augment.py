"""L4 auxiliary / mask-simulation augmentation (Phase 4 scaffold).

L4 fields may only enter training when explicitly labeled synthetic or auxiliary.
ponytail: single-pixel noise + mask replay; upgrade to spatially correlated noise.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any

import numpy as np

SOURCE_REAL_L3 = 0
SOURCE_SYNTH_L4_MASKED = 1
SOURCE_L4_AUX = 2
SOURCE_MISSING = 3


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


def augment_config_summary(l4_cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "enabled": bool(l4_cfg.get("enabled", False)),
        "noise_scale": float(l4_cfg.get("noise_scale", 1.0)),
        "mission_dropout_prob": float(l4_cfg.get("mission_dropout_prob", 0.0)),
        "time_window_dropout_prob": float(l4_cfg.get("time_window_dropout_prob", 0.0)),
    }
