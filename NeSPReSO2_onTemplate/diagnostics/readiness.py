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

import gsw_torch as gsw  # noqa: E402 — API matches gsw; returns torch.Tensor

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
) -> dict[str, Any]:
    """RC-2 placeholder — steric height vs SSH consistency (not wired yet)."""
    _ = (temperature, salinity, pressure, latitude, longitude, ssh)
    return {
        "status": "not_implemented",
        "reason": "steric SSH consistency requires reviewed methodology and matched SSH targets",
    }


def uncertainty_calibration_hook(
    ensemble_mean: np.ndarray | None = None,
    ensemble_spread: np.ndarray | None = None,
    target: np.ndarray | None = None,
) -> dict[str, Any]:
    """RC-4 placeholder — spread-error ratio / ENCE when ensemble outputs exist."""
    _ = (ensemble_mean, ensemble_spread, target)
    return {
        "status": "not_implemented",
        "reason": "ensemble outputs not available in current training path",
        "expected_fields": ["ensemble_mean", "ensemble_spread", "target", "depth", "subset_metadata"],
    }


def readiness_report(
    temperature: np.ndarray,
    salinity: np.ndarray,
    *,
    pressure: np.ndarray | None = None,
    latitude: np.ndarray | None = None,
    longitude: np.ndarray | None = None,
    ssh: np.ndarray | None = None,
    tol_kgm3: float = DEFAULT_STABILITY_TOL_KGM3,
    metadata: Mapping[str, Any] | None = None,
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
            temperature, salinity, pressure, latitude, longitude, ssh=ssh
        ),
        "uncertainty_calibration": uncertainty_calibration_hook(),
    }
    return report


def readiness_from_checkpoint(
    config,
    checkpoint_path: str,
    *,
    split: str = "test",
    tol_kgm3: float = DEFAULT_STABILITY_TOL_KGM3,
) -> dict[str, Any]:
    from collections import OrderedDict

    from nb_metrics import depth_meters, pcs_to_profiles_depth_major, run_inference

    inf = run_inference(config, checkpoint_path, split=split)
    idx = np.asarray(inf["indices"], dtype=int)
    outputs: OrderedDict = inf["outputs"]
    pred = pcs_to_profiles_depth_major(inf["pcs"], inf["pca_models"], outputs)
    cache = inf["cache"]
    z = depth_meters(cache)
    lat = np.asarray(cache["LAT"], dtype=np.float64)[idx]
    lon = np.asarray(cache["LON"], dtype=np.float64)[idx]

    meta = {
        "checkpoint": str(checkpoint_path),
        "cache": str(inf["cache_path"]),
        "dataset_tag": inf["dataset_tag"],
        "split": split,
        "n_samples": int(inf["n_samples"]),
        "variables": list(outputs.keys()),
    }
    return readiness_report(
        pred["temperature"],
        pred["salinity"],
        pressure=z,
        latitude=lat,
        longitude=lon,
        tol_kgm3=tol_kgm3,
        metadata=meta,
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
    lines.extend(
        [
            "",
            "## Steric SSH / uncertainty",
            "",
            f"- Steric SSH: `{report['steric_ssh']['status']}`",
            f"- Uncertainty calibration: `{report['uncertainty_calibration']['status']}`",
        ]
    )
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
        config, args.resume, split=args.split, tol_kgm3=args.tol
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
