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

Loop-ready evaluator (PLAN-agentic-ai-experiment.md Track A)
-----------------------------------------------------------
This is the fitness function for a search loop, so it is built to be hard to
fool and hard to misread:

* ``--repeat N`` — report min/median/sigma. A single shot cannot tell a real
  speedup from timing noise.
* ``--json`` — machine-readable ``{elapsed_min, elapsed_median, sigma, golden}``.
* ``--cold``/``--warm`` — in-process cache state is explicit and reported.
* ``--pin-caches`` — cache sizes come from *here*, not from constructor defaults
  inside the mutable file, so a candidate cannot buy speed with memory.
* ``--max-rss-mb`` — memory ceiling, asserted.
* Any traceback is a hard reject (exit 2), never a stack dump mistaken for noise.

Exit codes: ``0`` pass, ``1`` golden drift / ceiling breach, ``2`` error.

    python3 scripts/bench_datacube_speed.py --cascade --json    # loop entry point
"""

from __future__ import annotations

import argparse
import json
import resource
import statistics
import sys
import time
import traceback
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

GOLDEN_DIR = _ROOT / "tests" / "golden"

# Cache sizes the evaluator pins. These live here, not in sampler.py's constructor defaults,
# because sampler.py is the mutable file in the search: a candidate that quietly raises its own
# cache ceilings would buy wall-clock with RAM and look like an algorithmic win.
PINNED_CACHES = {"plane_cache_size": 512, "stack_cache_size": 128, "derived_cache_size": 512}
DEFAULT_MAX_RSS_MB = 8192

# (profiles_per_day, n_profiles). Stage 1 is the cheap filter; stage 2 is the full gate.
# n_profiles MUST match what each golden was saved with (ppd50 was saved at 500, not 300) or the
# shape guard rejects every candidate for the wrong reason. Cross-check tests/golden/*.meta.json.
GOLDEN_N_PROFILES = {1: 300, 5: 300, 50: 500}
CASCADE_STAGE1 = [(50, GOLDEN_N_PROFILES[50])]
CASCADE_STAGE2 = [(1, GOLDEN_N_PROFILES[1]), (5, GOLDEN_N_PROFILES[5]), (50, GOLDEN_N_PROFILES[50])]


def golden_path_for(profiles_per_day: int) -> Path:
    """Golden file for a config. Explicit argument — never a mutated global."""
    if profiles_per_day == 1:
        return GOLDEN_DIR / "sampler_golden_v1.npz"
    return GOLDEN_DIR / f"sampler_golden_ppd{profiles_per_day}.npz"


def cube_data_revision() -> int | None:
    """``data_revision`` of the cube on disk, or None if unreadable."""
    try:
        import zarr

        return int(zarr.open(str(default_cube_path(_ROOT)), mode="r").attrs["data_revision"])
    except Exception:
        return None


def golden_data_revision(golden_path: Path) -> int | None:
    meta = golden_path.with_suffix(".meta.json")
    if not meta.is_file():
        return None
    try:
        return json.loads(meta.read_text()).get("data_revision")
    except Exception:
        return None


def _peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


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

    Note this scans planes to find usable days, which warms the OS page cache
    before any timing starts. See ``run()``'s ``warm`` argument.
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


def run(
    n_profiles: int,
    seed: int,
    profiles_per_day: int = 1,
    *,
    repeat: int = 1,
    warm: bool = False,
    pin_caches: bool = True,
) -> tuple[dict, list[float]]:
    """Time ``sample()`` ``repeat`` times. Returns (payload from last run, elapsed list).

    ``warm=False`` (default) builds a fresh provider for each timed call, so in-process caches
    start empty and the measurement is not inflated by ``make_profiles``' plane scan. The OS
    page cache stays warm either way — dropping it needs root, so we report rather than claim.
    """
    cube_path = default_cube_path(_ROOT)
    cache_kwargs = dict(PINNED_CACHES) if pin_caches else {}

    setup_provider = CubeProvider(cube_path, **cache_kwargs)
    lats, lons, dates_jd = make_profiles(setup_provider, n_profiles, seed, profiles_per_day)

    elapsed: list[float] = []
    table = None
    for _ in range(max(1, repeat)):
        provider = setup_provider if warm else CubeProvider(cube_path, **cache_kwargs)
        t0 = time.perf_counter()
        table = provider.sample(FEATURE_SPEC, lats, lons, dates_jd)
        elapsed.append(time.perf_counter() - t0)

    payload = {
        "names": table.names,
        "values": table.values,
        "valid_mask": table.valid_mask,
        "lats": lats,
        "lons": lons,
        "dates_jd": dates_jd,
    }
    return payload, elapsed


def check_golden(payload: dict, golden_path: Path, *, atol: float, rtol: float) -> tuple[bool, str]:
    """Compare payload to golden. Returns (ok, reason). Shape/name drift is a clean FAIL."""
    if not golden_path.is_file():
        return False, f"no golden file at {golden_path}; run --save-golden first"

    # Provenance before values. A cube rebuild silently invalidates every golden: rev 2 -> rev 3
    # (a decode fix) left these files asserting a 3 degC Gulf of Mexico for ten days, and the only
    # symptom was every candidate failing. Compare revisions first so that rot is a one-line
    # diagnosis instead of an archaeology exercise.
    cube_rev, golden_rev = cube_data_revision(), golden_data_revision(golden_path)
    if cube_rev is not None and golden_rev is not None and cube_rev != golden_rev:
        return False, (
            f"STALE GOLDEN: built against cube data_revision={golden_rev}, cube on disk is "
            f"data_revision={cube_rev}. The cube was rebuilt; re-derive the golden from a "
            f"trusted sampler (git stash any local sampler edits first) — do NOT regenerate it "
            f"from a modified sampler, that launders the change into the baseline."
        )
    if golden_rev is None:
        print(
            f"[bench] warning: {golden_path.name} has no data_revision in its .meta.json — "
            f"provenance unverifiable; re-save it to enable the staleness check.",
            file=sys.stderr,
        )

    golden = np.load(golden_path, allow_pickle=False)

    if list(golden["names"]) != list(payload["names"]):
        return False, "feature name/order mismatch"
    # Guard shapes before comparing: an off-golden --n-profiles used to surface as a raw
    # ValueError traceback, which reads like a crash rather than "you asked the wrong question".
    if golden["values"].shape != payload["values"].shape:
        return False, (
            f"shape mismatch: golden {golden['values'].shape} vs run {payload['values'].shape} "
            f"— --n-profiles/--profiles-per-day do not match this golden file"
        )
    if golden["valid_mask"].shape != payload["valid_mask"].shape:
        return False, f"valid_mask shape mismatch: {golden['valid_mask'].shape} vs {payload['valid_mask'].shape}"

    if not np.array_equal(golden["valid_mask"], payload["valid_mask"]):
        n_diff = int(np.sum(golden["valid_mask"] != payload["valid_mask"]))
        return False, f"valid_mask differs in {n_diff} cells"
    if not np.allclose(golden["values"], payload["values"], atol=atol, rtol=rtol, equal_nan=True):
        diff = np.abs(golden["values"] - payload["values"])
        worst = float(np.nanmax(diff))
        j = int(np.nanargmax(np.where(np.isnan(diff), -1, diff)))
        name = payload["names"][j % len(payload["names"])]
        return False, f"values differ, max |diff|={worst:.3e} (near feature {name!r})"
    return True, "values and valid_mask match golden within tolerance"


def eval_config(
    n_profiles: int,
    profiles_per_day: int,
    *,
    seed: int,
    repeat: int,
    warm: bool,
    pin_caches: bool,
    atol: float,
    rtol: float,
    do_check: bool,
) -> dict:
    """One (n_profiles, profiles_per_day) config. Never raises — errors become a result."""
    label = f"ppd{profiles_per_day}"
    try:
        payload, elapsed = run(
            n_profiles, seed, profiles_per_day, repeat=repeat, warm=warm, pin_caches=pin_caches
        )
    except Exception:
        return {
            "config": label,
            "status": "error",
            "golden": "fail",
            "traceback": traceback.format_exc(limit=6),
        }

    out: dict = {
        "config": label,
        "status": "ok",
        "n_profiles": n_profiles,
        "profiles_per_day": profiles_per_day,
        "n_features": len(payload["names"]),
        "repeat": len(elapsed),
        "elapsed_min": min(elapsed),
        "elapsed_median": statistics.median(elapsed),
        "elapsed_all": elapsed,
        "sigma": statistics.stdev(elapsed) if len(elapsed) > 1 else 0.0,
        "cache_state": "warm" if warm else "cold",
        "page_cache": "warm (cannot drop without root)",
        "caches_pinned": PINNED_CACHES if pin_caches else "constructor defaults",
        "peak_rss_mb": round(_peak_rss_mb(), 1),
    }
    out["sigma_pct"] = round(100 * out["sigma"] / out["elapsed_median"], 2) if out["elapsed_median"] else 0.0

    if do_check:
        ok, reason = check_golden(payload, golden_path_for(profiles_per_day), atol=atol, rtol=rtol)
        out["golden"] = "pass" if ok else "fail"
        out["golden_reason"] = reason
    else:
        out["golden"] = "skipped"
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-profiles", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--profiles-per-day", type=int, default=1)
    parser.add_argument("--save-golden", action="store_true")
    parser.add_argument("--check-golden", action="store_true")
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--repeat", type=int, default=1, help="timed repeats; report min/median/sigma")
    parser.add_argument("--warm", action="store_true", help="reuse one provider (in-process caches warm)")
    parser.add_argument("--cold", action="store_true", help="fresh provider per repeat (default)")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("--cascade", action="store_true", help="stage 1 (ppd50) filter, then full gate")
    parser.add_argument("--no-pin-caches", action="store_true", help="use sampler.py constructor defaults")
    parser.add_argument("--max-rss-mb", type=float, default=DEFAULT_MAX_RSS_MB)
    args = parser.parse_args(argv)

    if args.warm and args.cold:
        parser.error("--warm and --cold are mutually exclusive")
    warm = bool(args.warm)
    pin_caches = not args.no_pin_caches

    if args.save_golden:
        gp = golden_path_for(args.profiles_per_day)
        try:
            payload, elapsed = run(
                args.n_profiles, args.seed, args.profiles_per_day, repeat=1, warm=warm,
                pin_caches=pin_caches,
            )
        except Exception:
            traceback.print_exc()
            return 2
        gp.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            gp,
            names=np.array(payload["names"]),
            values=payload["values"],
            valid_mask=payload["valid_mask"],
            lats=payload["lats"],
            lons=payload["lons"],
            dates_jd=payload["dates_jd"],
        )
        gp.with_suffix(".meta.json").write_text(
            json.dumps(
                {
                    "n_profiles": args.n_profiles,
                    "seed": args.seed,
                    "elapsed_s": elapsed[0],
                    # Provenance: pins the golden to the cube build it was derived from.
                    "data_revision": cube_data_revision(),
                    "profiles_per_day": args.profiles_per_day,
                },
                indent=2,
            )
        )
        print(f"[bench] saved golden output to {gp}")
        return 0

    if args.cascade:
        stages = [("stage1", CASCADE_STAGE1), ("stage2", CASCADE_STAGE2)]
    else:
        stages = [("single", [(args.profiles_per_day, args.n_profiles)])]

    do_check = args.check_golden or args.cascade
    results: list[dict] = []
    verdict = "pass"
    for stage_name, configs in stages:
        for ppd, n in configs:
            r = eval_config(
                n, ppd, seed=args.seed, repeat=args.repeat, warm=warm, pin_caches=pin_caches,
                atol=args.atol, rtol=args.rtol, do_check=do_check,
            )
            r["stage"] = stage_name
            results.append(r)
            if r["status"] == "error":
                verdict = "error"
                break
            if r.get("golden") == "fail":
                verdict = "fail"
                break
        if verdict != "pass":
            break  # cascade: a stage-1 reject never pays for stage 2

    peak = _peak_rss_mb()
    rss_ok = peak <= args.max_rss_mb
    if not rss_ok and verdict == "pass":
        verdict = "fail"

    summary = {
        "verdict": verdict,
        "peak_rss_mb": round(peak, 1),
        "max_rss_mb": args.max_rss_mb,
        "rss_within_ceiling": rss_ok,
        "results": results,
    }

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        for r in results:
            if r["status"] == "error":
                print(f"[bench] {r['config']}: ERROR\n{r.get('traceback','')}", file=sys.stderr)
                continue
            print(
                f"[bench] {r['config']}: {r['n_profiles']} profiles, {r['n_features']} features | "
                f"min {r['elapsed_min']:.3f}s median {r['elapsed_median']:.3f}s "
                f"sigma {r['sigma']:.3f}s ({r['sigma_pct']}%) | {r['cache_state']} | golden {r['golden']}"
            )
            if r.get("golden") == "fail":
                print(f"[bench] FAIL: {r['golden_reason']}", file=sys.stderr)
        if not rss_ok:
            print(f"[bench] FAIL: peak RSS {peak:.0f} MB exceeds ceiling {args.max_rss_mb:.0f} MB", file=sys.stderr)
        print(f"[bench] verdict: {verdict}")

    return {"pass": 0, "fail": 1, "error": 2}[verdict]


if __name__ == "__main__":
    raise SystemExit(main())
