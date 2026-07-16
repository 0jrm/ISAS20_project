"""Readiness diagnostics on saved temperature/salinity profile predictions."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "notebooks") not in sys.path:
    sys.path.insert(0, str(_ROOT / "notebooks"))

import gsw_torch as gsw  # noqa: E402 — differentiable GSW bindings; use as ``gsw`` throughout

DEFAULT_STABILITY_TOL_KGM3 = 0.01
_GSW_DTYPE = torch.float64
# ponytail: gsw_torch vs reference gsw differ by ~1e-5 kg/m³ on σ₀ (verified in selfcheck)
_GSW_REF_ATOL_KGM3 = 1e-4


def _git_commit() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_ROOT.parent,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _as_depth_major(arr: np.ndarray, n_depth: int, n_samples: int) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    if a.shape == (n_depth, n_samples):
        return a
    if a.shape == (n_samples, n_depth):
        return a.T
    raise ValueError(f"expected ({n_depth}, {n_samples}) or transpose, got {a.shape}")


def _pressure_grid(
    pressure: np.ndarray | None,
    *,
    n_depth: int,
    n_samples: int,
) -> np.ndarray:
    if pressure is None:
        z = np.linspace(0.0, 1800.0, n_depth, dtype=np.float64)
        return np.broadcast_to(z[:, None], (n_depth, n_samples)).copy()
    p = np.asarray(pressure, dtype=np.float64)
    if p.ndim == 1:
        if p.shape[0] != n_depth:
            raise ValueError(f"pressure length {p.shape[0]} != n_depth {n_depth}")
        return np.broadcast_to(p[:, None], (n_depth, n_samples)).copy()
    return _as_depth_major(p, n_depth, n_samples)


def _valid_interface_mask(temp: np.ndarray, sal: np.ndarray) -> np.ndarray:
    """Boolean (n_depth-1, n_samples): True where both adjacent levels are finite."""
    ok = np.isfinite(temp) & np.isfinite(sal)
    return ok[:-1] & ok[1:]


def static_stability_diagnostic(
    temperature: np.ndarray,
    salinity: np.ndarray,
    pressure: np.ndarray | None,
    latitude: np.ndarray,
    longitude: np.ndarray,
    *,
    tol_kgm3: float = DEFAULT_STABILITY_TOL_KGM3,
) -> dict[str, Any]:
    """RC-1: flag σ₀ inversions (lighter water below denser water)."""
    lat = np.asarray(latitude, dtype=np.float64).ravel()
    lon = np.asarray(longitude, dtype=np.float64).ravel()
    n_samples = lat.shape[0]
    temp_arr = np.asarray(temperature, dtype=np.float64)
    if temp_arr.ndim != 2:
        raise ValueError("temperature must be 2-D (depth × samples or samples × depth)")
    if temp_arr.shape[1] == n_samples:
        n_depth = temp_arr.shape[0]
    elif temp_arr.shape[0] == n_samples:
        n_depth = temp_arr.shape[1]
    else:
        raise ValueError(f"temperature shape {temp_arr.shape} inconsistent with n_samples={n_samples}")
    temp = _as_depth_major(temp_arr, n_depth, n_samples)
    sal = _as_depth_major(salinity, n_depth, n_samples)
    pres = _pressure_grid(pressure, n_depth=n_depth, n_samples=n_samples)
    if lon.shape[0] != n_samples:
        raise ValueError("latitude/longitude length must match n_samples")

    interface_depth_m = 0.5 * (pres[:-1, 0] + pres[1:, 0])

    lon_t = torch.as_tensor(lon, dtype=_GSW_DTYPE)
    lat_t = torch.as_tensor(lat, dtype=_GSW_DTYPE)
    temp_t = torch.as_tensor(temp, dtype=_GSW_DTYPE)
    sal_t = torch.as_tensor(sal, dtype=_GSW_DTYPE)
    pres_t = torch.as_tensor(pres, dtype=_GSW_DTYPE)

    sa = gsw.SA_from_SP(sal_t, pres_t, lon_t, lat_t)
    ct = gsw.CT_from_t(sa, temp_t, pres_t)
    sigma0 = gsw.sigma0(sa, ct).detach().cpu().numpy()
    delta = np.diff(sigma0, axis=0)
    valid = _valid_interface_mask(temp, sal)
    viol = valid & (delta < -float(tol_kgm3))

    profile_flags = np.any(viol, axis=0)
    violation_count_by_interface = viol.sum(axis=1).astype(int)
    total_violations = int(viol.sum())
    total_interfaces_checked = int(valid.sum())

    magnitude_sum_by_interface = np.zeros(n_depth - 1, dtype=np.float64)
    magnitude_count_by_interface = np.zeros(n_depth - 1, dtype=int)
    if np.any(viol):
        mags = -delta[viol]
        idx_k, _idx_j = np.where(viol)
        for k, mag in zip(idx_k, mags):
            magnitude_sum_by_interface[k] += float(mag)
            magnitude_count_by_interface[k] += 1

    checked_profiles = int(np.sum(np.any(valid, axis=0)))
    violation_rate = float(np.mean(profile_flags)) if n_samples else 0.0
    interface_violation_rate = (
        float(total_violations / total_interfaces_checked) if total_interfaces_checked else 0.0
    )
    mean_mag_by_interface = np.full(n_depth - 1, np.nan, dtype=np.float64)
    nz = magnitude_count_by_interface > 0
    mean_mag_by_interface[nz] = magnitude_sum_by_interface[nz] / magnitude_count_by_interface[nz]

    return {
        "method": "gsw_torch.sigma0_inversion",
        "tol_kgm3": float(tol_kgm3),
        "n_profiles": int(n_samples),
        "n_profiles_checked": checked_profiles,
        "violation_rate": violation_rate,
        "interface_violation_rate": interface_violation_rate,
        "total_violations": int(total_violations),
        "total_interfaces_checked": int(total_interfaces_checked),
        "profile_flags": profile_flags.astype(int).tolist(),
        "interface_depth_m": interface_depth_m.tolist(),
        "violation_count_by_interface": violation_count_by_interface.tolist(),
        "mean_violation_magnitude_by_interface_kgm3": [
            float(v) if np.isfinite(v) else None for v in mean_mag_by_interface
        ],
    }


def steric_ssh_diagnostic(
    temperature: np.ndarray,
    salinity: np.ndarray,
    pressure: np.ndarray | None,
    latitude: np.ndarray,
    longitude: np.ndarray,
    ssh: np.ndarray | None = None,
    *,
    clim_steric: np.ndarray | None = None,
    calibration: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """RC-2: calibrated steric height anomaly vs observed SLA."""
    from model.steric import steric_height_anomaly
    import torch

    lat = np.asarray(latitude, dtype=np.float64).ravel()
    lon = np.asarray(longitude, dtype=np.float64).ravel()
    n_samples = lat.shape[0]
    temp = _as_depth_major(np.asarray(temperature, dtype=np.float64), temperature.shape[0], n_samples)
    sal = _as_depth_major(np.asarray(salinity, dtype=np.float64), salinity.shape[0], n_samples)
    pres = _pressure_grid(pressure, n_depth=temp.shape[0], n_samples=n_samples)

    temp_t = torch.as_tensor(temp, dtype=_GSW_DTYPE)
    sal_t = torch.as_tensor(sal, dtype=_GSW_DTYPE)
    pres_t = torch.as_tensor(pres, dtype=_GSW_DTYPE)
    lat_t = torch.as_tensor(lat, dtype=_GSW_DTYPE)
    lon_t = torch.as_tensor(lon, dtype=_GSW_DTYPE)

    with torch.no_grad():
        steric = steric_height_anomaly(temp_t, sal_t, pres_t, lat_t, lon_t, subsample_dz=5).numpy()

    # RC-2 is only meaningful against the train-split affine fit that absorbs deep steric,
    # barotropic and DUACS offsets. Without clim_steric + calibration, pred_sla would be an
    # absolute steric height compared to an anomaly — a plausible-looking, meaningless number.
    if clim_steric is None or calibration is None:
        return {
            "status": "unavailable",
            "reason": "cache lacks clim_steric and/or steric_calibration (rebuild with steric_at_build)",
            "method": "gsw_torch.steric_height_anomaly",
            "n_profiles": int(n_samples),
            "steric_height_m": steric.tolist(),
        }

    cal = calibration
    alpha = float(cal.get("alpha", 1.0))
    beta = float(cal.get("beta", 0.0))
    steric_anom = steric - np.asarray(clim_steric, dtype=np.float64).ravel()
    pred_sla = alpha * steric_anom + beta

    out: dict[str, Any] = {
        "status": "ok",
        "method": "gsw_torch.steric_height_anomaly",
        "calibration": {k: float(v) for k, v in cal.items()},
        "n_profiles": int(n_samples),
        "steric_height_m": steric.tolist(),
        "calibrated_sla_m": pred_sla.tolist(),
    }
    if ssh is not None:
        obs = np.asarray(ssh, dtype=np.float64).ravel()
        valid = np.isfinite(obs) & np.isfinite(pred_sla)
        if valid.sum() > 1:
            out["rmse_m"] = float(np.sqrt(np.mean((pred_sla[valid] - obs[valid]) ** 2)))
            out["correlation"] = float(np.corrcoef(pred_sla[valid], obs[valid])[0, 1])
        else:
            out["rmse_m"] = None
            out["correlation"] = None
    return out


def ensemble_crps(members: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Empirical CRPS per point for an ensemble; ``members`` is (N, ...), ``target`` (...).

    Uses the sorted-ensemble identity
    ``CRPS = mean|x_i - y| - (1/N²) Σ_i (2i - N - 1) x_(i)``  (i 1-indexed on sorted members),
    which is the O(N log N) form of the pairwise double sum. Degenerates to |x - y| (MAE)
    for a zero-spread ensemble, as CRPS should.
    """
    x = np.asarray(members, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    n = x.shape[0]
    if n < 2:
        raise ValueError("CRPS needs at least 2 ensemble members")
    term1 = np.mean(np.abs(x - y[None, ...]), axis=0)
    xs = np.sort(x, axis=0)
    coef = (2.0 * np.arange(1, n + 1) - n - 1).reshape(-1, *([1] * (xs.ndim - 1)))
    term2 = np.sum(coef * xs, axis=0) / (n * n)
    return term1 - term2


def uncertainty_calibration_hook(
    ensemble_mean: np.ndarray | None = None,
    ensemble_spread: np.ndarray | None = None,
    target: np.ndarray | None = None,
    depth: np.ndarray | None = None,
    subset_metadata: Mapping[str, Any] | None = None,
    *,
    members: np.ndarray | None = None,
    n_members: int | None = None,
    n_bins: int = 10,
) -> dict[str, Any]:
    """RC-4 — spread-error ratio, ENCE, reliability by depth, CRPS.

    Arrays are (n_depth, n_samples); ``depth`` is (n_depth,). ``members`` (N, n_depth,
    n_samples) is optional and only needed for CRPS. Returns ``not_implemented`` when no
    ensemble is supplied, preserving the N=0 contract.
    """
    if ensemble_mean is None or ensemble_spread is None or target is None:
        return {
            "status": "not_implemented",
            "reason": "ensemble outputs not available in current training path",
            "expected_fields": ["ensemble_mean", "ensemble_spread", "target", "depth", "subset_metadata"],
        }

    mean = np.asarray(ensemble_mean, dtype=np.float64)
    spread = np.asarray(ensemble_spread, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    if not (mean.shape == spread.shape == truth.shape):
        raise ValueError(f"shape mismatch: mean {mean.shape}, spread {spread.shape}, target {truth.shape}")

    err = mean - truth
    valid = np.isfinite(err) & np.isfinite(spread)
    if valid.sum() < 2:
        return {"status": "unavailable", "reason": "fewer than 2 finite ensemble/target points"}

    e, s = err[valid], spread[valid]
    rmse = float(np.sqrt(np.mean(e**2)))
    rms_spread = float(np.sqrt(np.mean(s**2)))
    ratio = float(rms_spread / rmse) if rmse > 0 else None

    out: dict[str, Any] = {
        "status": "ok",
        "method": "mc_dropout",
        "n_members": int(n_members) if n_members is not None else None,
        "n_points": int(valid.sum()),
        "rmse": rmse,
        "rms_spread": rms_spread,
        "spread_error_ratio": ratio,
        "mean_spread": float(np.mean(s)),
        "mean_abs_error": float(np.mean(np.abs(e))),
    }

    # A perfectly reliable N-member ensemble scores sqrt(N/(N+1)), not 1: the ensemble mean
    # carries σ²/N of sampling error on top of the σ² it is trying to represent. Report the
    # target so a finite-N ratio is not misread as under-dispersion.
    if n_members is not None and n_members > 1:
        ideal = float(np.sqrt(n_members / (n_members + 1.0)))
        out["finite_n_ideal_ratio"] = ideal
        out["spread_error_ratio_corrected"] = float(ratio / ideal) if ratio is not None else None

    # ENCE (Levi et al. 2022) over equal-count bins of predicted spread.
    order = np.argsort(s)
    bins = np.array_split(order, min(n_bins, max(1, len(order))))
    ence_terms, bin_rows = [], []
    for b in bins:
        if b.size == 0:
            continue
        rmv_b = float(np.sqrt(np.mean(s[b] ** 2)))
        rmse_b = float(np.sqrt(np.mean(e[b] ** 2)))
        bin_rows.append({"n": int(b.size), "rmv": rmv_b, "rmse": rmse_b,
                         "ratio": float(rmv_b / rmse_b) if rmse_b > 0 else None})
        if rmv_b > 0:
            ence_terms.append(abs(rmv_b - rmse_b) / rmv_b)
    out["ence"] = float(np.mean(ence_terms)) if ence_terms else None
    out["calibration_bins"] = bin_rows

    # Does a larger predicted spread actually track a larger error? (rank, so it is
    # insensitive to the systematic under-scaling the spread-error ratio already reports)
    if len(s) > 2 and np.ptp(s) > 0:
        sr = _rankdata(s)
        ar = _rankdata(np.abs(e))
        out["spread_error_rank_corr"] = float(np.corrcoef(sr, ar)[0, 1])

    # ⚠️ The pooled rank correlation above is confounded by depth: spread and |error| are
    # both large near the surface and both small at depth, so pooling over all (depth,
    # sample) points scores high even if the spread carries no per-profile information.
    # Recompute WITHIN each depth level (across samples) and report the distribution — that
    # is the only version that answers "at a given depth, does a wider spread mean a larger
    # error on THIS profile?", which is what DA would actually consume.
    if mean.ndim == 2 and mean.shape[1] > 2:
        per_depth = []
        for k in range(mean.shape[0]):
            vk = valid[k]
            if vk.sum() <= 2:
                continue
            sk, ek = spread[k][vk], np.abs(err[k][vk])
            if np.ptp(sk) == 0 or np.ptp(ek) == 0:
                continue
            per_depth.append(float(np.corrcoef(_rankdata(sk), _rankdata(ek))[0, 1]))
        if per_depth:
            arr = np.asarray(per_depth, dtype=np.float64)
            out["spread_error_rank_corr_within_depth"] = {
                "median": float(np.median(arr)),
                "q25": float(np.percentile(arr, 25)),
                "q75": float(np.percentile(arr, 75)),
                "n_depth_levels": int(arr.size),
                "note": "pooled rank corr is depth-confounded; this is the depth-controlled version",
            }

    # Reliability by depth: (n_depth, n_samples) reduced across samples.
    if mean.ndim == 2:
        with np.errstate(invalid="ignore"):
            rmse_d = np.sqrt(np.nanmean(np.where(valid, err, np.nan) ** 2, axis=1))
            spread_d = np.sqrt(np.nanmean(np.where(valid, spread, np.nan) ** 2, axis=1))
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio_d = np.where(rmse_d > 0, spread_d / rmse_d, np.nan)
        out["reliability_by_depth"] = {
            "depth_m": (np.asarray(depth, dtype=np.float64).ravel().tolist()
                        if depth is not None else None),
            "rmse": _nan_to_none(rmse_d),
            "rms_spread": _nan_to_none(spread_d),
            "spread_error_ratio": _nan_to_none(ratio_d),
        }

    if members is not None:
        m = np.asarray(members, dtype=np.float64)
        if m.shape[1:] != truth.shape:
            raise ValueError(f"members {m.shape} inconsistent with target {truth.shape}")
        crps = ensemble_crps(m, truth)
        out["crps"] = float(np.nanmean(np.where(valid, crps, np.nan)))
        # CRPS of the deterministic ensemble mean = MAE; the ratio shows what the spread buys.
        out["crps_over_mae"] = (float(out["crps"] / out["mean_abs_error"])
                                if out["mean_abs_error"] > 0 else None)

    if subset_metadata:
        out["subset_metadata"] = dict(subset_metadata)
    return out


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average-tie ranks (scipy.stats.rankdata equivalent, kept dependency-free)."""
    a = np.asarray(a, dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    ranks[order] = np.arange(1, len(a) + 1, dtype=np.float64)
    sorted_a = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = 0.5 * (i + 1 + j + 1)
        i = j + 1
    return ranks


def _nan_to_none(a: np.ndarray) -> list:
    return [None if not np.isfinite(v) else float(v) for v in np.asarray(a).ravel()]


def readiness_report(
    temperature: np.ndarray,
    salinity: np.ndarray,
    *,
    pressure: np.ndarray | None = None,
    latitude: np.ndarray | None = None,
    longitude: np.ndarray | None = None,
    ssh: np.ndarray | None = None,
    clim_steric: np.ndarray | None = None,
    steric_calibration: Mapping[str, float] | None = None,
    tol_kgm3: float = DEFAULT_STABILITY_TOL_KGM3,
    metadata: Mapping[str, Any] | None = None,
    uncertainty: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    temp = np.asarray(temperature, dtype=np.float64)
    if temp.ndim != 2:
        raise ValueError("temperature must be 2-D (depth × samples or samples × depth)")
    n_samples = temp.shape[1] if temp.shape[0] > temp.shape[1] else temp.shape[0]
    if latitude is None or longitude is None:
        raise ValueError("latitude and longitude are required for GSW static-stability diagnostics")

    report: dict[str, Any] = {
        "version": 1,
        "timestamp": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "git_commit": _git_commit(),
        "metadata": dict(metadata or {}),
        "static_stability": static_stability_diagnostic(
            temperature,
            salinity,
            pressure,
            latitude,
            longitude,
            tol_kgm3=tol_kgm3,
        ),
        "steric_ssh": steric_ssh_diagnostic(
            temperature,
            salinity,
            pressure,
            latitude,
            longitude,
            ssh=ssh,
            clim_steric=clim_steric,
            calibration=steric_calibration,
        ),
        "uncertainty_calibration": (
            uncertainty_calibration_hook(**dict(uncertainty)) if uncertainty
            else uncertainty_calibration_hook()
        ),
    }
    return report


def readiness_from_checkpoint(
    config,
    checkpoint_path: str,
    *,
    split: str = "test",
    tol_kgm3: float = DEFAULT_STABILITY_TOL_KGM3,
    mc_samples: int = 0,
    mc_variable: str = "temperature",
) -> dict[str, Any]:
    from collections import OrderedDict

    from model.loss import reconstruct_physical_profiles
    from nb_metrics import depth_meters, profiles_depth_major, run_inference

    inf = run_inference(config, checkpoint_path, split=split, mc_samples=mc_samples)
    idx = np.asarray(inf["indices"], dtype=int)
    outputs: OrderedDict = inf["outputs"]
    cache = inf["cache"]

    # Anomaly caches predict anomaly PCs: climatology must be added back before GSW, or σ₀ is
    # computed on ~0 °C / ~0 PSU and the whole report is silently wrong. No-op on raw caches.
    clim_profiles = cache.get("clim_profiles")
    pred = reconstruct_physical_profiles(
        inf["pcs"], inf["pca_models"], outputs, clim_profiles=clim_profiles, indices=idx
    )
    z = depth_meters(cache)
    lat = np.asarray(cache["LAT"], dtype=np.float64)[idx]
    lon = np.asarray(cache["LON"], dtype=np.float64)[idx]

    def _sel(key: str) -> np.ndarray | None:
        arr = cache.get(key)
        return None if arr is None else np.asarray(arr, dtype=np.float64)[idx]

    meta = {
        "checkpoint": str(checkpoint_path),
        "cache": str(inf["cache_path"]),
        "dataset_tag": inf["dataset_tag"],
        "split": split,
        "n_samples": int(inf["n_samples"]),
        "variables": list(outputs.keys()),
        "anomaly_cache": clim_profiles is not None,
    }

    uncertainty = None
    if mc_samples > 0:
        if mc_variable not in outputs:
            raise ValueError(f"mc_variable {mc_variable!r} not in outputs {list(outputs)}")
        # Every member goes through the same physical reconstruction as the deterministic
        # prediction — on an anomaly cache the climatology add-back is what makes the mean
        # and the target comparable at all (it cancels in the spread, not in the error).
        members = np.stack(
            [
                reconstruct_physical_profiles(
                    inf["mc_pcs"][m], inf["pca_models"], outputs,
                    clim_profiles=clim_profiles, indices=idx,
                )[mc_variable]
                for m in range(mc_samples)
            ],
            axis=0,
        )
        # cache["profiles"] is physical and depth-major; cache["true_profiles"] is anomalies
        # and sample-major. profiles_depth_major reads the former — the only correct target
        # for physically-reconstructed members.
        truth = profiles_depth_major(cache, mc_variable)[:, idx]
        uncertainty = {
            "ensemble_mean": members.mean(axis=0),
            "ensemble_spread": members.std(axis=0, ddof=1),
            "target": truth,
            "depth": z,
            "members": members,
            "n_members": int(mc_samples),
            "subset_metadata": {
                "variable": mc_variable,
                "split": split,
                "n_samples": int(inf["n_samples"]),
                "dropout_active": True,
                "anomaly_cache": clim_profiles is not None,
            },
        }
        meta["mc_samples"] = int(mc_samples)
        meta["mc_variable"] = mc_variable

    return readiness_report(
        pred["temperature"],
        pred["salinity"],
        pressure=z,
        latitude=lat,
        longitude=lon,
        ssh=_sel("ssh_obs_sla"),
        clim_steric=_sel("clim_steric"),
        steric_calibration=cache.get("steric_calibration"),
        tol_kgm3=tol_kgm3,
        metadata=meta,
        uncertainty=uncertainty,
    )


def to_markdown(report: Mapping[str, Any]) -> str:
    stab = report["static_stability"]
    meta = report.get("metadata") or {}
    lines = [
        "# Readiness diagnostics",
        "",
        f"- **Checkpoint:** `{meta.get('checkpoint', 'n/a')}`",
        f"- **Cache:** `{meta.get('cache', 'n/a')}`",
        f"- **Split:** {meta.get('split', 'n/a')} (n={stab['n_profiles']})",
        "",
        "## Static stability (σ₀ inversions)",
        "",
        f"- Method: `{stab['method']}` (tol={stab['tol_kgm3']} kg/m³)",
        f"- Profile violation rate: **{100 * stab['violation_rate']:.2f}%**",
        f"- Interface violation rate: **{100 * stab['interface_violation_rate']:.2f}%**",
        f"- Total violations: {stab['total_violations']} / {stab['total_interfaces_checked']} interfaces",
        "",
        "| Interface depth (m) | Violation count | Mean |Δσ₀| (kg/m³) |",
        "|---:|---:|---:|",
    ]
    for z, cnt, mag in zip(
        stab["interface_depth_m"],
        stab["violation_count_by_interface"],
        stab["mean_violation_magnitude_by_interface_kgm3"],
    ):
        if cnt == 0:
            continue
        mag_s = f"{mag:.4f}" if mag is not None else "n/a"
        lines.append(f"| {z:.0f} | {cnt} | {mag_s} |")
    steric = report["steric_ssh"]
    lines.extend(["", "## Steric SSH vs observed SLA (RC-2)", "", f"- Status: `{steric['status']}`"])
    if steric["status"] == "ok":
        cal = steric.get("calibration", {})
        rmse, corr = steric.get("rmse_m"), steric.get("correlation")
        lines.extend(
            [
                f"- Calibration: alpha={cal.get('alpha', float('nan')):.4f}, "
                f"beta={cal.get('beta', float('nan')):.4f}, r_train={cal.get('r_train', float('nan')):.4f}",
                f"- RMSE vs observed SLA: **{rmse:.4f} m**" if rmse is not None else "- RMSE: n/a",
                f"- Correlation: **{corr:.4f}**" if corr is not None else "- Correlation: n/a",
            ]
        )
    else:
        lines.append(f"- Reason: {steric.get('reason', 'n/a')}")
    unc = report["uncertainty_calibration"]
    lines.extend(["", "## Uncertainty calibration (RC-4)", "", f"- Status: `{unc['status']}`"])
    if unc["status"] == "ok":
        sub = unc.get("subset_metadata") or {}
        ratio = unc.get("spread_error_ratio")
        ideal = unc.get("finite_n_ideal_ratio")
        corrected = unc.get("spread_error_ratio_corrected")
        lines.extend(
            [
                f"- Method: `{unc['method']}` ({unc.get('n_members')} members, "
                f"variable `{sub.get('variable', 'n/a')}`, {unc['n_points']} points)",
                f"- RMSE (ensemble mean): **{unc['rmse']:.4f}**",
                f"- RMS spread: **{unc['rms_spread']:.4f}**",
                f"- **Spread-error ratio: {ratio:.4f}**" if ratio is not None else "- Spread-error ratio: n/a",
            ]
        )
        if ideal is not None:
            lines.append(
                f"  - a perfectly reliable {unc['n_members']}-member ensemble scores "
                f"{ideal:.4f}; ratio/ideal = **{corrected:.4f}**"
            )
        if ratio is not None:
            verdict = ("**over-confident** (spread ≪ error)" if ratio < 0.8
                       else "**under-confident** (spread ≫ error)" if ratio > 1.25
                       else "approximately calibrated")
            lines.append(f"  - reading: {verdict}")
        if unc.get("ence") is not None:
            lines.append(f"- ENCE: **{unc['ence']:.4f}** ({len(unc.get('calibration_bins', []))} equal-count spread bins)")
        if unc.get("spread_error_rank_corr") is not None:
            lines.append(
                f"- Spread↔|error| rank correlation (pooled): {unc['spread_error_rank_corr']:.4f} "
                "— ⚠️ depth-confounded, see below"
            )
        wd = unc.get("spread_error_rank_corr_within_depth")
        if wd:
            lines.append(
                f"- Spread↔|error| rank correlation **within depth**: median "
                f"**{wd['median']:.4f}** (IQR {wd['q25']:.4f}–{wd['q75']:.4f}, "
                f"{wd['n_depth_levels']} levels) — the depth-controlled number, and the one DA cares about"
            )
        if unc.get("crps") is not None:
            lines.append(f"- CRPS: **{unc['crps']:.4f}** (MAE = {unc['mean_abs_error']:.4f}, "
                         f"CRPS/MAE = {unc['crps_over_mae']:.4f})")
        bins = unc.get("calibration_bins") or []
        if bins:
            lines.extend(["", "| spread bin | n | RMV | RMSE | RMV/RMSE |", "|---:|---:|---:|---:|---:|"])
            for i, b in enumerate(bins):
                r = f"{b['ratio']:.3f}" if b.get("ratio") is not None else "n/a"
                lines.append(f"| {i + 1} | {b['n']} | {b['rmv']:.4f} | {b['rmse']:.4f} | {r} |")
    else:
        lines.append(f"- Reason: {unc.get('reason', 'n/a')}")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="NeSPReSO readiness diagnostics on saved predictions")
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-r", "--resume", required=True, help="checkpoint .pth")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--tol", type=float, default=DEFAULT_STABILITY_TOL_KGM3, help="σ₀ inversion tolerance (kg/m³)")
    parser.add_argument("-d", "--device", default=None)
    parser.add_argument("--out", default=None, help="write JSON report")
    parser.add_argument("--md-out", default=None, help="write Markdown summary")
    parser.add_argument(
        "--mc-samples", type=int, default=0,
        help="MC-dropout members for RC-4 uncertainty calibration (0 = off, ~50 typical)",
    )
    parser.add_argument(
        "--mc-variable", default="temperature", help="variable to calibrate (default: temperature)"
    )
    args = parser.parse_args(argv)
    if args.device:
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    from parse_config import ConfigParser, validate_config
    from base.util import read_json

    cfg = read_json(args.config)
    validate_config(cfg)
    config = ConfigParser(cfg)
    report = readiness_from_checkpoint(
        config, args.resume, split=args.split, tol_kgm3=args.tol,
        mc_samples=args.mc_samples, mc_variable=args.mc_variable,
    )
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")
    if args.md_out:
        Path(args.md_out).write_text(to_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
