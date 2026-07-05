#!/usr/bin/env python3
"""M2 spot-check: cube-sampled SST/SSS/SSH vs reference scalars at profile locations."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from preproc.features.export_feature_cache import POINT_CUBE_FEATURES  # noqa: E402
from preproc.features.sampler import CubeProvider  # noqa: E402

CHANNELS = ("sss", "sst", "ssh")
CUBE_FEATURE = {
    "sss": "sss.value@local",
    "sst": "sst.value@local",
    "ssh": "ssh.value@local",
}
REF_COL = {"sss": 6, "sst": 7, "ssh": 8}


def _load_profiles_from_pickle(pickle_path: Path, v2_src: str | None, max_samples: int | None) -> dict[str, np.ndarray]:
    if v2_src:
        sys.path.insert(0, str(v2_src))
    from astropy.time import Time
    from nespreso.data.pickle_compat import load_dataset_pickle
    from nespreso.utils.time import datenum_to_datetime

    data = load_dataset_pickle(str(pickle_path))
    ds = data["full_dataset"]
    n = len(ds)
    if max_samples is not None:
        n = min(n, int(max_samples))
    lat = np.asarray(ds.LAT[:n], dtype=np.float64)
    lon = np.asarray(ds.LON[:n], dtype=np.float64)
    juld = np.asarray(ds.TIME[:n], dtype=np.float64)
    dates_jd = np.array([Time(datenum_to_datetime(float(t))).jd for t in juld], dtype=np.float64)
    input_params = {
        "timecos": True,
        "timesin": True,
        "latcos": True,
        "latsin": True,
        "loncos": True,
        "lonsin": True,
        "sss": True,
        "sst": True,
        "ssh": True,
        "sat": True,
    }
    ds.input_params = input_params
    rows = []
    for i in range(n):
        x, _ = ds[i]
        rows.append(x.numpy() if hasattr(x, "numpy") else np.asarray(x, dtype=np.float32))
    inputs = np.stack(rows).astype(np.float32)
    return {
        "n": n,
        "lat": lat,
        "lon": lon,
        "juld": juld,
        "dates_jd": dates_jd,
        "reference": {ch: inputs[:, REF_COL[ch]] for ch in CHANNELS},
    }


def _channel_stats(ref: np.ndarray, cube: np.ndarray) -> dict[str, Any]:
    mask = np.isfinite(ref) & np.isfinite(cube)
    n = int(mask.sum())
    if n == 0:
        return {"n_finite": 0, "bias": None, "slope": None, "intercept": None, "correlation": None}
    r = ref[mask]
    c = cube[mask]
    bias = float(np.mean(r - c))
    lr = stats.linregress(c, r)
    corr = float(np.corrcoef(c, r)[0, 1]) if n > 1 else None
    return {
        "n_finite": n,
        "bias": bias,
        "slope": float(lr.slope),
        "intercept": float(lr.intercept),
        "correlation": corr,
        "mean_reference": float(np.mean(r)),
        "mean_cube": float(np.mean(c)),
        "rmse": float(np.sqrt(np.mean((r - c) ** 2))),
    }


def _example_profiles(
    lat: np.ndarray,
    lon: np.ndarray,
    juld: np.ndarray,
    ref_by_ch: dict[str, np.ndarray],
    cube_by_ch: dict[str, np.ndarray],
    *,
    n_examples: int = 4,
    seed: int = 0,
) -> list[dict[str, Any]]:
    n = len(lat)
    mask = np.ones(n, dtype=bool)
    for ch in CHANNELS:
        mask &= np.isfinite(ref_by_ch[ch]) & np.isfinite(cube_by_ch[ch])
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    rng = np.random.default_rng(seed)
    pick = idx if idx.size <= n_examples else rng.choice(idx, size=n_examples, replace=False)
    examples = []
    for i in pick:
        ex = {
            "index": int(i),
            "lat": float(lat[i]),
            "lon": float(lon[i]),
            "juld": float(juld[i]),
            "reference": {ch: float(ref_by_ch[ch][i]) for ch in CHANNELS},
            "cube": {ch: float(cube_by_ch[ch][i]) for ch in CHANNELS},
            "diff": {ch: float(ref_by_ch[ch][i] - cube_by_ch[ch][i]) for ch in CHANNELS},
        }
        examples.append(ex)
    return examples


def run_spotcheck(
    cube_path: Path,
    reference: Path,
    *,
    v2_src: str | None = None,
    max_samples: int | None = 500,
    n_examples: int = 4,
    seed: int = 0,
) -> dict[str, Any]:
    ref_path = Path(reference)
    if ref_path.suffix == ".pkl":
        prof = _load_profiles_from_pickle(ref_path, v2_src, max_samples)
    else:
        raise ValueError(f"unsupported --reference type: {ref_path} (use v2 .pkl)")

    provider = CubeProvider(cube_path)
    table = provider.sample(POINT_CUBE_FEATURES, prof["lat"], prof["lon"], prof["dates_jd"])
    name_to_j = {n: i for i, n in enumerate(table.names)}
    cube_by_ch = {ch: table.values[:, name_to_j[CUBE_FEATURE[ch]]].astype(np.float64) for ch in CHANNELS}

    channels = {}
    for ch in CHANNELS:
        channels[ch] = _channel_stats(prof["reference"][ch], cube_by_ch[ch])

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cube_path": str(cube_path.resolve()),
        "reference": str(ref_path.resolve()),
        "n_profiles": int(prof["n"]),
        "channels": channels,
        "example_profiles": _example_profiles(
            prof["lat"],
            prof["lon"],
            prof["juld"],
            prof["reference"],
            cube_by_ch,
            n_examples=n_examples,
            seed=seed,
        ),
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="M2 cube-vs-reference spot check")
    parser.add_argument("--cube-path", required=True, help="path to gom_cube.zarr")
    parser.add_argument("--reference", required=True, help="reference pickle (v2 dataset)")
    parser.add_argument("--out", required=True, help="output JSON path")
    parser.add_argument("--v2-src", default="/unity/g2/jmiranda/v2-nespreso/src")
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--n-examples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    report = run_spotcheck(
        Path(args.cube_path),
        Path(args.reference),
        v2_src=args.v2_src,
        max_samples=args.max_samples,
        n_examples=args.n_examples,
        seed=args.seed,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
