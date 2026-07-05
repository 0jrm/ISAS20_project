#!/usr/bin/env python3
"""Benchmark + golden-output regression harness for CubeProvider.sample().

Fixed synthetic profile subset (seeded RNG) + a representative feature spec
(all operator kinds: value, grad, laplacian, tendency, geo_uv). Used as the
safety net for the sampler hot-path refactor in PLAN_datacube_speed.md:

    python3 scripts/bench_datacube_speed.py --save-golden   # before refactor
    python3 scripts/bench_datacube_speed.py --check-golden  # after refactor

``--check-golden`` fails loudly if values or valid_mask drift beyond
``atol=1e-6`` (rtol=1e-5), which is the tolerance the plan requires for
pure-performance refactors.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from astropy.time import Time

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from preproc.cube.cube_schema import TIME_END, TIME_START, default_cube_path  # noqa: E402
from preproc.features.sampler import CubeProvider  # noqa: E402

BENCH_CHANNELS = ("sst", "sss", "ssh")

FEATURE_SPEC = {
    "scalars": [
        "timecos", "timesin", "latcos", "latsin", "loncos", "lonsin",
        "basin_sss", "basin_sst", "basin_ssh", "bathy_depth",
    ],
    "operators": [
        {"op": "value", "channels": ["sst", "sss", "ssh"], "scales": ["local"]},
        {"op": "grad", "channels": ["sst", "sss", "ssh"], "scales": ["local", "1.0deg"]},
        {"op": "laplacian", "channels": ["ssh"], "scales": ["1.0deg"]},
        {"op": "tendency", "channels": ["sst", "ssh"], "window_days": 7},
        {"op": "geo_uv", "channels": ["ssh"], "scales": ["local", "1.0deg"]},
    ],
}

GOLDEN_PATH = _ROOT / "tests" / "golden" / "sampler_golden_v1.npz"


def _day_is_usable(provider: CubeProvider, t_idx: int) -> bool:
    """Reject cube days with a fully-NaN plane on any bench channel (data gap, not a refactor concern)."""
    for ch in BENCH_CHANNELS:
        try:
            plane = provider.plane(ch, t_idx)
        except Exception:
            return False
        if not np.isfinite(plane).any():
            return False
    return True


def make_profiles(
    provider: CubeProvider, n: int, seed: int = 0, profiles_per_day: int = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``profiles_per_day > 1`` clusters multiple synthetic profiles onto the same cube day.

    Realistic ARGO exports have several profiles per day; that's the exact
    scenario the old per-profile ``geo_uv`` cache key (keyed on rounded lat)
    made pathological — one plane recompute per profile instead of per day.
    """
    rng = np.random.default_rng(seed)
    lats = rng.uniform(19.5, 29.5, size=n)
    lons = rng.uniform(-96.5, -82.5, size=n)

    # Keep >=10 days clear of TIME_START so tendency windows never need clamping,
    # and >=2 days clear of TIME_END/whitelisted-missing tail.
    t_lo = 10
    t_hi = int((TIME_END - TIME_START) / np.timedelta64(1, "D")) - 2

    n_days = max(1, -(-n // profiles_per_day))  # ceil(n / profiles_per_day)
    candidates = rng.permutation(np.arange(t_lo, t_hi + 1))
    usable_t_idx: list[int] = []
    for t_idx in candidates:
        if _day_is_usable(provider, int(t_idx)):
            usable_t_idx.append(int(t_idx))
        if len(usable_t_idx) >= n_days:
            break
    if len(usable_t_idx) < n_days:
        raise RuntimeError(f"only found {len(usable_t_idx)}/{n_days} usable cube days in range")

    # Not shuffled: for profiles_per_day=1 this must reproduce the exact
    # (profile index -> day) association used before clustering support was added,
    # so the original golden file stays valid.
    day_for_profile = np.array(usable_t_idx * profiles_per_day)[:n]
    days = TIME_START + day_for_profile.astype("timedelta64[D]")
    dates_jd = np.array([Time(str(d)).jd for d in days], dtype=np.float64)
    return lats, lons, dates_jd


def run(n_profiles: int, seed: int, profiles_per_day: int = 1) -> tuple[dict, float]:
    cube_path = default_cube_path(_ROOT)
    provider = CubeProvider(cube_path)
    lats, lons, dates_jd = make_profiles(provider, n_profiles, seed, profiles_per_day)

    t0 = time.perf_counter()
    table = provider.sample(FEATURE_SPEC, lats, lons, dates_jd)
    elapsed = time.perf_counter() - t0

    payload = {
        "names": table.names,
        "values": table.values,
        "valid_mask": table.valid_mask,
        "lats": lats,
        "lons": lons,
        "dates_jd": dates_jd,
    }
    return payload, elapsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-profiles", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--profiles-per-day", type=int, default=1)
    parser.add_argument("--save-golden", action="store_true")
    parser.add_argument("--check-golden", action="store_true")
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-5)
    args = parser.parse_args(argv)

    global GOLDEN_PATH
    if args.profiles_per_day != 1:
        GOLDEN_PATH = GOLDEN_PATH.with_name(f"sampler_golden_ppd{args.profiles_per_day}.npz")

    payload, elapsed = run(args.n_profiles, args.seed, args.profiles_per_day)
    print(f"[bench] sample() over {args.n_profiles} profiles, {len(payload['names'])} features: {elapsed:.3f}s")

    if args.save_golden:
        GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            GOLDEN_PATH,
            names=np.array(payload["names"]),
            values=payload["values"],
            valid_mask=payload["valid_mask"],
            lats=payload["lats"],
            lons=payload["lons"],
            dates_jd=payload["dates_jd"],
        )
        meta_path = GOLDEN_PATH.with_suffix(".meta.json")
        meta_path.write_text(json.dumps({"n_profiles": args.n_profiles, "seed": args.seed, "elapsed_s": elapsed}, indent=2))
        print(f"[bench] saved golden output to {GOLDEN_PATH}")

    if args.check_golden:
        if not GOLDEN_PATH.is_file():
            print(f"[bench] no golden file at {GOLDEN_PATH}; run --save-golden first", file=sys.stderr)
            return 1
        golden = np.load(GOLDEN_PATH, allow_pickle=False)
        names_match = list(golden["names"]) == list(payload["names"])
        if not names_match:
            print("[bench] FAIL: feature name/order mismatch", file=sys.stderr)
            return 1
        ok_valid = np.array_equal(golden["valid_mask"], payload["valid_mask"])
        ok_values = np.allclose(
            golden["values"], payload["values"], atol=args.atol, rtol=args.rtol, equal_nan=True
        )
        if ok_valid and ok_values:
            print("[bench] PASS: values and valid_mask match golden within tolerance")
            return 0
        if not ok_valid:
            n_diff = int(np.sum(golden["valid_mask"] != payload["valid_mask"]))
            print(f"[bench] FAIL: valid_mask differs in {n_diff} cells", file=sys.stderr)
        if not ok_values:
            diff = np.abs(golden["values"] - payload["values"])
            worst = np.nanmax(diff)
            j = int(np.nanargmax(np.where(np.isnan(diff), -1, diff)))
            name = payload["names"][j % len(payload["names"])]
            print(f"[bench] FAIL: values differ, max |diff|={worst:.3e} (near feature {name!r})", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
