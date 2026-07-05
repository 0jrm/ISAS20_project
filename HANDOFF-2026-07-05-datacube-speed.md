# Session handoff — datacube speed plan execution

**Branch:** `residual_cube`
**Date:** 2026-07-05
**Source plan:** [`PLAN_datacube_speed.md`](PLAN_datacube_speed.md) (repo root) — read that first, this handoff assumes it.
**Code home:** `NeSPReSO2_onTemplate/` (all paths below are relative to that dir unless stated otherwise)
**Conda env:** `nespreso` for all python commands.

User asked to execute the full plan (Phases 0–5). This session completed Phase 0 and got Phase 1 numerically verified except for one loose end (see "Immediate next step" below) — **stopped mid-command, nothing is broken, just needs one more verification run.**

---

## State of the working tree right now

Uncommitted, not yet committed (user said only commit when asked):

- `NeSPReSO2_onTemplate/preproc/cube/cube_schema.py` — modified (vectorized `time_index_of`)
- `NeSPReSO2_onTemplate/preproc/features/sampler.py` — modified (Phase 1 hot-path rewrite)
- `NeSPReSO2_onTemplate/scripts/bench_datacube_speed.py` — new file (Phase 0 benchmark harness)
- `NeSPReSO2_onTemplate/tests/golden/` — new dir, golden `.npz` files (see below; **one may need to be regenerated**, see next-step)
- `PLAN_datacube_speed.md` — pre-existing untracked file, not touched by me

No commits made. No cube data rebuilt. Nothing destructive happened.

---

## Immediate next step (was mid-flight when interrupted)

I had just fixed a bug in `scripts/bench_datacube_speed.py`'s `make_profiles()` — it had an
`rng.shuffle(day_for_profile)` call that silently broke reproducibility of the *original*
golden file (`tests/golden/sampler_golden_v1.npz`, generated before clustering support existed).
I removed that shuffle line (see diff already applied — `day_for_profile` is no longer shuffled).

**Run this first, in `NeSPReSO2_onTemplate/`, before anything else:**

```bash
conda activate nespreso
python3 scripts/bench_datacube_speed.py --n-profiles 300 --check-golden
```

Expect: `[bench] PASS: values and valid_mask match golden within tolerance`. This checks the
Phase 1 sampler rewrite against the *original pre-refactor* golden output (300 synthetic
profiles, each on a distinct day, full operator coverage: value/grad/laplacian/tendency/geo_uv).

If it passes (it should — the shuffle bug was in the harness, not `sampler.py`; a very similar
test already passed with the fix in place before this got interrupted), then:

**Regenerate the two clustered golden files**, because they were captured *before* the shuffle
fix, using the *old* (pre-Phase-1) sampler — the shuffle bug means their day/profile pairing no
longer matches what the fixed harness now produces. Do this via `git stash` (only the two
sampler files, not the new benchmark script):

```bash
cd /unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project
git stash push -- NeSPReSO2_onTemplate/preproc/features/sampler.py NeSPReSO2_onTemplate/preproc/cube/cube_schema.py
cd NeSPReSO2_onTemplate
python3 scripts/bench_datacube_speed.py --n-profiles 300 --profiles-per-day 5  --save-golden
python3 scripts/bench_datacube_speed.py --n-profiles 500 --profiles-per-day 50 --save-golden
cd ..
git stash pop
cd NeSPReSO2_onTemplate
python3 scripts/bench_datacube_speed.py --n-profiles 300 --profiles-per-day 5  --check-golden
python3 scripts/bench_datacube_speed.py --n-profiles 500 --profiles-per-day 50 --check-golden
python3 -m pytest tests/test_sampler.py tests/test_cube_validate.py -q
```

All three `--check-golden` invocations (default + ppd5 + ppd50) and both test files must pass.
Once green, Phase 1 is done — check off task #2, move to Phase 2. Each `--save-golden`/`--check-golden`
call takes 30s–5min depending on clustering (unclustered is slowest: ~3 min per side).

---

## What's done: Phase 0 (benchmark harness)

New file `scripts/bench_datacube_speed.py`. Loads the real `data/cube/gom_cube.zarr`, builds a
feature spec covering all operator kinds (mirrors `config/argo/config_argo_residual_cube.json`'s
`features` block: value/grad/laplacian/tendency/geo_uv across sst/sss/ssh), generates a seeded
synthetic profile set, calls `CubeProvider.sample()`, times it, and can save/check a golden
`.npz` of `(values, valid_mask)` at `atol=1e-6, rtol=1e-5`.

Key knobs:
- `--n-profiles N` — profile count
- `--profiles-per-day K` — cluster profiles onto `ceil(N/K)` unique cube days instead of N
  distinct days (real ARGO exports have several profiles/day — this is the scenario that made
  the old `geo_uv` cache-key bug pathological; all-unique-days doesn't exercise it much)
- `--save-golden` / `--check-golden` — golden path auto-suffixes by `profiles_per_day` (default
  → `tests/golden/sampler_golden_v1.npz`; ppd=5 → `..._ppd5.npz`; ppd=50 → `..._ppd50.npz`)

Gotcha discovered along the way: the cube has at least one **fully-NaN plane** for `sst` at
`t_idx=682` (2016-11-13) that is *not* in `ALLOWED_MISSING_DAYS`/`missing_days` attrs — a
pre-existing data-quality gap, unrelated to this speed work. The benchmark's `_day_is_usable()`
filters these out by construction (checks `isfinite(plane).any()` for sst/sss/ssh before picking
a day). Worth a separate investigation at some point (not started).

## What's done: Phase 1 (sampler hot path)

All in `preproc/features/sampler.py` + one function in `preproc/cube/cube_schema.py`.

**1. Fixed the geo_uv cache-key bug (the plan's "worst offender").** Old code cached the
derived plane per `(name, ch, t_idx, scale_lbl, round(lat_i, 3))` — since `geo_uv`'s `name`
already encodes direction and the key included the profile's own rounded latitude, the full-grid
smoothed gradient was recomputed on almost every profile. Fix: `grad` and `geo_uv` now share one
cache key `("grad", ch, t_idx, scale_lbl)` storing the raw `(gx, gy)` tuple (no lat dependence);
`f(lat)` is applied *after* sampling, per profile, vectorized over whichever group of profiles
shares that `t_idx`. This also means `grad` and `geo_uv` features on the same channel/scale now
share one computation (e.g. `ssh.grad_x@local` and `ssh.geo_u@local` reuse the same `gx,gy`).

**2. Batched sparse sampling by `t_idx` instead of per-row `getrow(i)`.** `CubeProvider.sample()`
now groups all profile indices by cube day once (`_group_by_t_idx`, a `np.argsort` + boundary
scan), and for every `(feature, day)` pair does one batched `sample_plane(weights[idx_arr], plane)`
sparse matmul over the whole group instead of one `.getrow(i)` per profile. This is the structural
fix for "O(N_profiles × F) instead of O(unique_days × F)".

**3. Vectorized time-index lookup.** `cube_schema.time_index_of` now does array arithmetic
(`(day - TIME_START).astype(int)` + bounds check) instead of rebuilding the full time axis and
doing `np.where` per call; added `time_indices_of_days()` (vectorized) which it wraps for the
scalar case. `sampler.py` added `time_indices_of_jd()` which does one batched
`astropy.time.Time(jd_array, format="jd").datetime64` conversion for the whole profile array
instead of one `Time(...).datetime` per profile.

**4. Basin mask caching.** `CubeProvider._basin_mask_for()` caches `basin_mask(...)` per
`(channel, basin_cfg-as-sorted-tuple)` instead of rebuilding the lat/lon meshgrid + mask on every
`basin_mean()` call.

**5. Bounded LRU caches.** Added `_LRUDict(OrderedDict)` (evicts least-recently-used past
`maxsize`) for `plane_cache`, `stack_cache`, `derived_planes` inside `sample()`. Sizes are
constructor params on `CubeProvider(cube_path, plane_cache_size=512, stack_cache_size=128,
derived_cache_size=512)` so a full-dataset export can size them to available RAM. **Gotcha:**
the standard "override `__getitem__`+`__setitem__`, evict via `popitem(last=False)`" pattern is
**broken** for `OrderedDict` subclasses — `popitem`/`pop` internally do `self[key]` which
re-enters the overridden `__getitem__` and corrupts the linked list (raised a spurious
`KeyError`). Fixed by evicting via `oldest = next(iter(self)); del self[oldest]` instead (the
documented-safe recipe from the stdlib docs).

**6. Tendency stacks read as one chunk-aligned slab.** `_read_tendency_stack()` reads
`root[ch][t_idx-window+1:t_idx+1]` in one zarr call when the range is fully in-bounds (checking
`_assert_plane_available` per day first, to preserve the whitelisted-missing-day exception
behavior exactly), falling back to the old per-day clamped loop only near the very start of the
cube time axis (boundary case, rare).

**Correctness bug found and fixed during verification** (this is why Phase 0's golden harness
mattered): my first geo_uv rewrite computed `valid[idx_arr, j] = gx_valid & gy_valid` for *both*
`geo_u` and `geo_v`. That's wrong — `geo_v = (g/f)*gx` depends only on `gx`, and `geo_u =
(-g/f)*gy` depends only on `gy`; the original single-plane cache's validity mask tracked
`isfinite()` of whichever *one* plane (`u` or `v`) was actually stored, not both. Fixed: `geo_u`
validity now tracks `gy_valid` only, `geo_v` tracks `gx_valid` only. This was caught by a 1-cell
mismatch in a clustered-profile golden check (a coastal cell right at the `wsum >= 0.5` boundary,
where the two masks' validity happened to differ) — the unclustered golden check didn't catch it
because it happened not to land on such a boundary cell.

**Measured speedup so far** (before the shuffle-bug/regeneration above, so take as directional,
not final): unclustered 300 profiles ~271s → ~184s (only getrow/time-index/mask-cache gains — no
day-sharing to exploit since every profile has a unique day, worst case for this fix); 5
profiles/day (300 profiles, 60 days): 61s → 28s; 50 profiles/day (500 profiles, 10 days): 22s →
4.7s. Real ARGO exports (~10k profiles, thousands of unique days over 2015–2022, several
profiles/day on average) should land in the favorable regime — expect an order of magnitude or
more, consistent with the plan's prediction, once re-verified with the corrected harness.

---

## Not started: Phases 2–5

Per the plan (`PLAN_datacube_speed.md`), still to do:

- **Phase 2 — cube build (`preproc/cube/build_cube.py`).** Single NetCDF open per worker task
  (currently opens 3x: `_indices_for_file`, `_read_hyperslab_plane`, `_slice_coords`); precompute
  bilinear regrid weights once per product instead of rebuilding `RegularGridInterpolator` +
  meshgrid + nearest-fallback interpolator per day in `_regrid_plane`; chunk-aligned buffered
  zarr writes instead of `zarr_arr[t_idx, :, :] = plane` one day at a time (64–128× write
  amplification per the plan). Regression gate: `python3 -m preproc.cube.build_cube --validate`
  (`validate_cube.py`) plus `tests/test_cube_validate.py`. This phase **touches on-disk cube
  writes** — rebuild only against a scratch/test cube path first, not `data/cube/gom_cube.zarr`,
  before trusting it against the real one.

- **Phase 3 — L3/L4 rasterization** (`preproc/l3_rasterize.py`, `preproc/l4_rasterize.py`).
  Vectorize `rasterize_era5_wind_for_target` (currently triple nested Python loop over
  time×lat×lon), `sample_l4_ssh_patch` (per-cell `np.argmin` via `_nearest_index`, re-reads the
  same file per time bin — should read once and use `searchsorted`), `bin_observations`
  (currently a dict-of-lists per cell, could be `np.add.at` histogramming). Parallelize
  `build_l3_processed_batch` (in `preproc/export_l3_cache.py`) across samples via
  `ProcessPoolExecutor`. **Caveat found this session:** per `HANDOFF.md`, this repo currently has
  **no raw L3/L4 netCDF data downloaded** (`data/raw/` missing → rasterization produces
  all-mask-zero bundles, the documented fast path). This phase can be implemented and unit-tested
  against synthetic in-memory arrays, but can't be validated end-to-end against real raw files in
  this environment. Confirm with the user whether to still invest here now vs. wait until L3/L4
  raw downloads land (see `HANDOFF.md` "Next coding tasks" #1).

- **Phase 4 — auxiliary exporters.** `preproc/climatology.py::_ridge_solve` solves one ridge
  regression per depth level in a Python loop (`fit_climatology`); plan suggests factoring
  `XtX + αI` once and solving multi-RHS when NaN patterns allow (need to check per-depth-level
  validity masks — they can differ across depths, so this needs a "group depths by identical
  valid-mask" step, not a blanket batch). `preproc/basin_stats.py::compute_basin_daily_means`
  already dedupes to unique days (better than the plan's writeup implied) but processes them
  sequentially — file I/O bound, a `ThreadPoolExecutor` over unique days would help.
  `preproc/overlap.py::days_since_1950` loops per-element through
  `nespreso.utils.time.datenum_to_datetime` (external repo, `/unity/g2/jmiranda/v2-nespreso/src`)
  — this is a pure affine transform in disguise (`datenum - (366 + date(1950,1,1).toordinal())`)
  and can be vectorized directly, skipping the datetime round-trip entirely, **but** note the
  original code's `timedelta(days=datenum % 1)` rounds to microsecond precision, so a fully
  vectorized version will be *more* precise than the original by sub-microsecond amounts — verify
  this is within whatever tolerance matters for downstream matched-eval code before landing it.
  `preproc/ssh_obs.py::sample_ssh_obs` is I/O-bound against `retrieve_satellite_data`
  (`utils/retrieve_sat.py`) — not easily vectorizable without changing that API; lower priority.

- **Phase 5 — ingestion format split.** `data_loader/data_loaders.py::NeSPReSODataLoader.__init__`
  does one `pickle.load()` of the entire train-ready cache (includes eval-only `profiles`,
  `true_profiles`, `clim_profiles`, PCA models alongside `inputs`/`targets`/`JULD` needed by every
  consumer). Plan wants inputs/targets/JULD split into an mmap-able core (`.npz` or small zarr)
  with eval-only payloads lazily loaded from a sidecar, keeping the pickle writer as a
  compatibility path. **Not investigated yet this session** — need to find the cache *writer*
  side (likely `preproc/preproc_isas_sat.py::write_train_cache` per the Explore-agent's earlier
  research, or similar) before touching the loader, since both ends need to agree on format. This
  is explicitly the lowest-priority phase per the plan ("optional... fixed per-run cost, not
  per-sample").

---

## Design notes worth preserving for whoever picks this up

- **The operator-version constraint from the plan applies:** `PLAN_datacube_speed.md` says pure
  perf refactors must produce numerically identical outputs and must NOT bump
  `operator_versions` (cache-invalidation hash) unless the numerics genuinely changed. Everything
  landed in Phase 1 is designed to be bit-identical to the original (verified via golden-output
  diffing at `atol=1e-6`) — no operator version bumps were needed or made.
- `expand_feature_names()` in `sampler.py` still emits `geo_u`/`geo_v` as separate named features
  (unchanged) — only the *internal* derived-plane cache key was unified with `grad`'s.
  `apply_operator("geo_uv", ...)` (in `preproc/features/operators.py`) is no longer called from
  `sampler.py`'s hot path at all (it now calls `apply_operator("grad", ...)` directly and does the
  `f(lat)` scaling inline) — the `op_geo_uv` function in `operators.py` itself is untouched and
  still used by `apply_operator`'s public API / anything else that might call it directly (grep
  showed no other call sites, but didn't delete the function — it's cheap to keep and matches
  "don't touch things outside what's asked").
- The `CubeProvider.weights_for()` cache is keyed by channel only (not by the specific
  `lats`/`lons` array passed in) — a **pre-existing** latent bug if `sample()` is ever called
  twice on the same provider instance with different profile sets for the same channel (it'd
  silently reuse stale weights). Not introduced by this session, not fixed either — out of scope
  per the plan, but worth flagging if it ever bites someone.

## Useful commands to reorient in a fresh session

```bash
cd /unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project/NeSPReSO2_onTemplate
conda activate nespreso
git diff --stat                                   # see the two modified files
python3 -m pytest tests/test_sampler.py tests/test_cube_validate.py -q
python3 scripts/bench_datacube_speed.py --n-profiles 300 --check-golden
```
