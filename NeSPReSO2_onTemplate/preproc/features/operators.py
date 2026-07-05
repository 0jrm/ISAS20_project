"""Declarative feature operators on cube planes (Component B)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from scipy import ndimage

EARTH_RADIUS_M = 6_371_000.0
GRAVITY = 9.81


@dataclass(frozen=True)
class OperatorSpec:
    name: str
    version: int
    fn: Callable[..., np.ndarray | tuple[np.ndarray, ...]]
    units_fn: Callable[[str], str] | None = None


OPERATOR_REGISTRY: dict[str, OperatorSpec] = {}


def register(name: str, version: int = 1, units_fn: Callable[[str], str] | None = None):
    def decorator(fn: Callable[..., np.ndarray | tuple[np.ndarray, ...]]):
        OPERATOR_REGISTRY[name] = OperatorSpec(name=name, version=version, fn=fn, units_fn=units_fn)
        return fn

    return decorator


def list_operators() -> dict[str, int]:
    return {k: v.version for k, v in OPERATOR_REGISTRY.items()}


def sigma_pixels(scale_deg: float, grid_step_deg: float) -> float:
    return max(scale_deg / grid_step_deg, 0.5)


def metric_spacing_m(lats: np.ndarray, lons: np.ndarray) -> tuple[float, float]:
    """Mean dy, dx in metres for the grid."""
    dlat = float(np.mean(np.diff(lats))) if len(lats) > 1 else 0.05
    dlon = float(np.mean(np.diff(lons))) if len(lons) > 1 else 0.05
    lat_c = float(np.mean(lats))
    dy = math.radians(dlat) * EARTH_RADIUS_M
    dx = math.radians(dlon) * EARTH_RADIUS_M * math.cos(math.radians(lat_c))
    return dy, dx


def normalized_gaussian_filter(field: np.ndarray, sigma_pix: float) -> np.ndarray:
    """Gaussian smooth with land/ocean mask normalization."""
    mask = np.isfinite(field).astype(np.float64)
    values = np.where(np.isfinite(field), field, 0.0).astype(np.float64)
    num = ndimage.gaussian_filter(values, sigma=sigma_pix, mode="nearest")
    den = ndimage.gaussian_filter(mask, sigma=sigma_pix, mode="nearest")
    out = np.full_like(field, np.nan, dtype=np.float64)
    valid = den > 1e-6
    out[valid] = num[valid] / den[valid]
    return out.astype(np.float32)


def normalized_gaussian_derivative(
    field: np.ndarray,
    sigma_pix: float,
    axis: int,
    spacing_m: float,
) -> np.ndarray:
    mask = np.isfinite(field).astype(np.float64)
    values = np.where(np.isfinite(field), field, 0.0).astype(np.float64)
    smoothed = ndimage.gaussian_filter1d(values, sigma=sigma_pix, axis=axis, mode="nearest")
    grad = np.gradient(smoothed, spacing_m, axis=axis)
    weight = ndimage.gaussian_filter1d(mask, sigma=sigma_pix, axis=axis, mode="nearest")
    out = np.full_like(field, np.nan, dtype=np.float64)
    valid = weight > 0.5
    out[valid] = grad[valid]
    return out.astype(np.float32)


@register("value", version=1, units_fn=lambda u: u)
def op_value(field: np.ndarray, *, lats: np.ndarray, lons: np.ndarray, scale_deg: float, grid_step_deg: float) -> np.ndarray:
    sig = sigma_pixels(scale_deg, grid_step_deg)
    return normalized_gaussian_filter(field, sig)


@register("value_centered", version=1, units_fn=lambda u: u)
def op_value_centered(
    field: np.ndarray,
    *,
    lats: np.ndarray,
    lons: np.ndarray,
    scale_deg: float,
    grid_step_deg: float,
    basin_value: float,
) -> np.ndarray:
    """Smoothed value at scale minus the day's basin mean.

    Isolates the local anomaly (e.g. eddy SSH) by removing the spatially
    constant basin-wide signal — the ``adt - daily basin mean`` centering the
    golden v2 model was handed. Because ``basin_value`` is constant over the
    plane, subtracting it before point-sampling is exact (== subtract after).
    Spatial derivatives (grad/laplacian/geo_uv) are unchanged by this constant
    offset, so only value and tendency benefit from centering.
    """
    sig = sigma_pixels(scale_deg, grid_step_deg)
    return (normalized_gaussian_filter(field, sig) - np.float32(basin_value)).astype(np.float32)


@register("grad", version=2, units_fn=lambda u: f"{u}/m")
def op_grad(
    field: np.ndarray,
    *,
    lats: np.ndarray,
    lons: np.ndarray,
    scale_deg: float,
    grid_step_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    sig = sigma_pixels(scale_deg, grid_step_deg)
    dy, dx = metric_spacing_m(lats, lons)
    gx = normalized_gaussian_derivative(field, sig, axis=1, spacing_m=dx)
    gy = normalized_gaussian_derivative(field, sig, axis=0, spacing_m=dy)
    return gx, gy


@register("laplacian", version=1, units_fn=lambda u: f"{u}/m^2")
def op_laplacian(
    field: np.ndarray,
    *,
    lats: np.ndarray,
    lons: np.ndarray,
    scale_deg: float,
    grid_step_deg: float,
) -> np.ndarray:
    sig = sigma_pixels(scale_deg, grid_step_deg)
    smooth = normalized_gaussian_filter(field, sig)
    dy, dx = metric_spacing_m(lats, lons)
    d2x = np.gradient(np.gradient(smooth, dx, axis=1), dx, axis=1)
    d2y = np.gradient(np.gradient(smooth, dy, axis=0), dy, axis=0)
    return (d2x + d2y).astype(np.float32)


@register("tendency", version=1, units_fn=lambda u: f"{u}/d")
def op_tendency(stack: np.ndarray, *, window_days: int = 7) -> np.ndarray:
    """Vectorized LSQ slope per pixel over time stack (days axis 0)."""
    y = np.asarray(stack, dtype=np.float64)
    t = np.arange(y.shape[0], dtype=np.float64)
    t_mean = t.mean()
    tc = t - t_mean
    denom = float(np.dot(tc, tc))
    if denom < 1e-12:
        return np.full(y.shape[1:], np.nan, dtype=np.float32)
    mask = np.isfinite(y)
    count = mask.sum(axis=0)
    y_mean = np.where(mask, y, 0.0).sum(axis=0) / np.maximum(count, 1)
    ym = np.where(mask, y - y_mean[None, :, :], 0.0)
    num = np.tensordot(tc, ym, axes=(0, 0))
    out = num / denom
    out[count < max(3, window_days // 2)] = np.nan
    return out.astype(np.float32)


@register("geo_uv", version=1, units_fn=lambda _u: "m/s")
def op_geo_uv(
    field: np.ndarray,
    *,
    lats: np.ndarray,
    lons: np.ndarray,
    scale_deg: float,
    grid_step_deg: float,
    profile_lat: float,
) -> tuple[np.ndarray, np.ndarray]:
    gx, gy = op_grad(field, lats=lats, lons=lons, scale_deg=scale_deg, grid_step_deg=grid_step_deg)
    f = 2.0 * 7.2921e-5 * math.sin(math.radians(profile_lat))
    if abs(f) < 1e-8:
        f = 1e-8
    u = (-GRAVITY / f) * gy
    v = (GRAVITY / f) * gx
    return u.astype(np.float32), v.astype(np.float32)


def apply_operator(
    name: str,
    field: np.ndarray,
    *,
    lats: np.ndarray | None = None,
    lons: np.ndarray | None = None,
    scale_deg: float = 1.0,
    grid_step_deg: float = 0.05,
    stack: np.ndarray | None = None,
    window_days: int = 7,
    profile_lat: float | None = None,
    basin_value: float | None = None,
) -> np.ndarray | tuple[np.ndarray, ...]:
    spec = OPERATOR_REGISTRY[name]
    if name == "tendency":
        if stack is None:
            raise ValueError("tendency requires stack")
        return spec.fn(stack, window_days=window_days)
    if lats is None or lons is None:
        raise ValueError(f"{name} requires lats and lons")
    if name == "value_centered":
        if basin_value is None:
            raise ValueError("value_centered requires basin_value")
        return spec.fn(
            field,
            lats=lats,
            lons=lons,
            scale_deg=scale_deg,
            grid_step_deg=grid_step_deg,
            basin_value=basin_value,
        )
    if name == "geo_uv":
        if profile_lat is None:
            raise ValueError("geo_uv requires profile_lat")
        return spec.fn(
            field,
            lats=lats,
            lons=lons,
            scale_deg=scale_deg,
            grid_step_deg=grid_step_deg,
            profile_lat=profile_lat,
        )
    return spec.fn(field, lats=lats, lons=lons, scale_deg=scale_deg, grid_step_deg=grid_step_deg)
