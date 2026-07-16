"""Probabilistic calibration metrics — numpy reference + torch mirror for training."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy import stats

from evalphys.constants import DEPTH_BAND_LABELS, DEPTH_BANDS, ENCE_MAX, SIGMA_MIN_DEFAULT

_INV_SQRT_PI = 1.0 / np.sqrt(np.pi)
_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)


def _align(*arrays: np.ndarray) -> list[np.ndarray]:
    shapes = {a.shape for a in arrays}
    if len(shapes) != 1:
        raise ValueError(f"shape mismatch: {shapes}")
    return [np.asarray(a, dtype=np.float64) for a in arrays]


def gaussian_crps(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray, *, sigma_min: float = SIGMA_MIN_DEFAULT) -> np.ndarray:
    """Closed-form Gaussian CRPS; returns array same shape as inputs."""
    mu, sigma, y = _align(mu, sigma, y)
    s = np.maximum(sigma, float(sigma_min))
    z = (y - mu) / s
    phi = stats.norm.pdf(z)
    Phi = stats.norm.cdf(z)
    return s * (z * (2.0 * Phi - 1.0) + 2.0 * phi - _INV_SQRT_PI)


def gaussian_crps_torch(mu, sigma, y, *, sigma_min: float = SIGMA_MIN_DEFAULT):
    """Torch mirror of :func:`gaussian_crps` (same formula)."""
    import torch

    s = torch.clamp(sigma, min=float(sigma_min))
    z = (y - mu) / s
    phi = torch.exp(-0.5 * z * z) * _INV_SQRT_2PI
    Phi = 0.5 * (1.0 + torch.erf(z / np.sqrt(2.0)))
    return s * (z * (2.0 * Phi - 1.0) + 2.0 * phi - _INV_SQRT_PI)


def pit_histogram(
    mu: np.ndarray,
    sigma: np.ndarray,
    y: np.ndarray,
    *,
    n_bins: int = 20,
    sigma_min: float = SIGMA_MIN_DEFAULT,
) -> dict[str, Any]:
    mu, sigma, y = _align(mu, sigma, y)
    s = np.maximum(sigma, float(sigma_min))
    pit = stats.norm.cdf((y - mu) / s)
    valid = np.isfinite(pit)
    pit = pit[valid]
    counts, edges = np.histogram(pit, bins=n_bins, range=(0.0, 1.0))
    freq = counts / counts.sum() if counts.sum() else np.zeros(n_bins)
    target = 1.0 / n_bins
    sup_dev = float(np.max(np.abs(freq - target))) if counts.sum() else None
    return {
        "n_bins": n_bins,
        "counts": counts.tolist(),
        "bin_edges": edges.tolist(),
        "frequencies": freq.tolist(),
        "sup_bin_deviation": sup_dev,
        "n_points": int(valid.sum()),
    }


def spread_skill(
    mu: np.ndarray,
    sigma: np.ndarray,
    y: np.ndarray,
    *,
    n_bins: int = 10,
    sigma_min: float = SIGMA_MIN_DEFAULT,
) -> dict[str, Any]:
    mu, sigma, y = _align(mu, sigma, y)
    s = np.maximum(sigma, float(sigma_min))
    err = mu - y
    valid = np.isfinite(err) & np.isfinite(s)
    e, s_flat = err[valid].ravel(), s[valid].ravel()
    if e.size < n_bins:
        return {"status": "unavailable", "reason": "too few points"}
    order = np.argsort(s_flat)
    bins = np.array_split(order, n_bins)
    mean_sigma, rmse = [], []
    for b in bins:
        if b.size == 0:
            continue
        mean_sigma.append(float(np.mean(s_flat[b])))
        rmse.append(float(np.sqrt(np.mean(e[b] ** 2))))
    mean_sigma_a = np.asarray(mean_sigma)
    rmse_a = np.asarray(rmse)
    if mean_sigma_a.size < 2 or np.ptp(mean_sigma_a) == 0:
        slope = None
    else:
        slope = float(np.polyfit(mean_sigma_a, rmse_a, 1)[0])
    if e.size > 2 and np.ptp(s_flat) > 0:
        spearman = float(stats.spearmanr(s_flat, np.abs(e)).statistic)
    else:
        spearman = None
    return {
        "status": "ok",
        "n_bins": n_bins,
        "mean_sigma_by_bin": mean_sigma,
        "rmse_by_bin": rmse,
        "slope_rmse_vs_sigma": slope,
        "spearman_sigma_abs_error": spearman,
    }


def ence(
    mu: np.ndarray,
    sigma: np.ndarray,
    y: np.ndarray,
    *,
    n_bins: int = 10,
    sigma_min: float = SIGMA_MIN_DEFAULT,
) -> dict[str, Any]:
    """ENCE (Levi et al.): mean_b |RMSE_b - RMV_b| / RMV_b."""
    mu, sigma, y = _align(mu, sigma, y)
    s = np.maximum(sigma, float(sigma_min))
    err = mu - y
    valid = np.isfinite(err) & np.isfinite(s)
    e, s_flat = err[valid].ravel(), s[valid].ravel()
    if e.size < 2:
        return {"ence": None, "bins": [], "threshold": ENCE_MAX}
    order = np.argsort(s_flat)
    bins = np.array_split(order, min(n_bins, max(1, len(order))))
    terms, rows = [], []
    for b in bins:
        if b.size == 0:
            continue
        rmv = float(np.sqrt(np.mean(s_flat[b] ** 2)))
        rmse = float(np.sqrt(np.mean(e[b] ** 2)))
        rows.append({"n": int(b.size), "rmv": rmv, "rmse": rmse})
        if rmv > 0:
            terms.append(abs(rmse - rmv) / rmv)
    return {"ence": float(np.mean(terms)) if terms else None, "bins": rows, "threshold": ENCE_MAX}


def season_from_juld(juld: np.ndarray, *, dataset_tag: str = "argo_v2") -> np.ndarray:
    """DJF/MAM/JJA/SON labels from cache JULD."""
    from base.split_utils import sample_dates

    d = sample_dates(np.asarray(juld, dtype=np.float64), dataset_tag=dataset_tag)
    month = d.astype("datetime64[M]").astype(int) % 12 + 1
    out = np.full(month.shape, "UNK", dtype=object)
    out[(month == 12) | (month <= 2)] = "DJF"
    out[(month >= 3) & (month <= 5)] = "MAM"
    out[(month >= 6) & (month <= 8)] = "JJA"
    out[(month >= 9) & (month <= 11)] = "SON"
    return out


def apply_strata(
    metric_fn: Callable[..., dict[str, Any]],
    *,
    strata: Mapping[str, np.ndarray | None],
    **kwargs: Any,
) -> dict[str, Any]:
    """Run ``metric_fn`` on full data and per-stratum subsets.

    ``strata`` keys: ``depth_band`` (uses level axis), ``season`` (per profile),
    ``input_error_tercile`` (per profile; skipped when None).
    """
    full = metric_fn(**kwargs)
    out: dict[str, Any] = {"all": full}
    n_prof = None
    for k in ("mu", "y", "T_pred"):
        if k in kwargs and kwargs[k] is not None:
            a = np.asarray(kwargs[k])
            n_prof = a.shape[0] if a.ndim >= 2 else a.size
            break
    season = strata.get("season")
    if season is not None and n_prof is not None:
        by_season: dict[str, Any] = {}
        for label in ("DJF", "MAM", "JJA", "SON"):
            m = season == label
            if not np.any(m):
                continue
            sliced = {key: (val[m] if isinstance(val, np.ndarray) and val.shape[0] == n_prof else val)
                        for key, val in kwargs.items()}
            by_season[label] = metric_fn(**sliced)
        out["by_season"] = by_season
    terc = strata.get("input_error_tercile")
    if terc is not None and n_prof is not None:
        by_terc: dict[str, Any] = {}
        for label in ("low", "mid", "high"):
            m = terc == label
            if not np.any(m):
                continue
            sliced = {key: (val[m] if isinstance(val, np.ndarray) and val.shape[0] == n_prof else val)
                        for key, val in kwargs.items()}
            by_terc[label] = metric_fn(**sliced)
        out["by_input_error_tercile"] = by_terc
    return out


def _mean_metric(x: np.ndarray, depth: np.ndarray | None, bands: Sequence[tuple[float, float]] = DEPTH_BANDS) -> dict[str, float | None]:
    if depth is None:
        return {"overall": float(np.nanmean(x))}
    z = np.asarray(depth, dtype=np.float64).reshape(-1)
    out: dict[str, float | None] = {"overall": float(np.nanmean(x))}
    for label, (lo, hi) in zip(DEPTH_BAND_LABELS, bands):
        if np.isfinite(hi):
            m = (z >= lo) & (z < hi)
        else:
            m = z >= lo
        if x.ndim == 2:
            sel = x[:, m]
        else:
            sel = x[m]
        out[label] = float(np.nanmean(sel)) if np.any(np.isfinite(sel)) else None
    return out


def summarize_calibration(
    mu: np.ndarray,
    sigma: np.ndarray,
    y: np.ndarray,
    *,
    depth: np.ndarray | None = None,
    sigma_min: float = SIGMA_MIN_DEFAULT,
) -> dict[str, Any]:
    crps = gaussian_crps(mu, sigma, y, sigma_min=sigma_min)
    return {
        "crps": _mean_metric(crps, depth),
        "pit": pit_histogram(mu, sigma, y, sigma_min=sigma_min),
        "spread_skill": spread_skill(mu, sigma, y, sigma_min=sigma_min),
        "ence": ence(mu, sigma, y, sigma_min=sigma_min),
    }
