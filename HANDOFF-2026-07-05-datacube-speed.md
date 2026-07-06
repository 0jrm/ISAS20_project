# Session handoff — datacube speed plan execution

**Branch:** `residual_cube`
**Date:** 2026-07-05
**Source plan:** [`PLAN_datacube_speed.md`](PLAN_datacube_speed.md) (repo root) — read that first, this handoff assumes it.
**Code home:** `NeSPReSO2_onTemplate/` (all paths below are relative to that dir unless stated otherwise)
**Conda env:** `nespreso` for all python commands. On this machine `conda activate nespreso` may not
put the right python first on `PATH` — if `import astropy`/`import zarr` fails, call the interpreter
directly: `/conda/jmiranda/miniconda/envs/nespreso/bin/python3`.

User asked to execute the full plan (Phases 0–5). **Phases 0, 1, and 2 are done, verified, committed,
and pushed to `origin/residual_cube`.** Phase 3 is skipped for now (no raw L3/L4 data in this
environment, per below). **Phase 4 is done and verified in this session but not yet committed** —
working tree has changes pending review/commit. Phase 5 not started.

```
d7519b7 Phase 2 datacube speed: single NetCDF open per read, chunk-aligned buffered writes
64053bd Phase 1 datacube speed: fix geo_uv cache-key bug, batch sampling by day
2cc6c2c cube first draft and runs
```

Phase 4 changes (uncommitted, `git diff --stat`):
```
NeSPReSO2_onTemplate/base/split_utils.py    | 26 +++++++++++++---
NeSPReSO2_onTemplate/preproc/basin_stats.py | 23 +++++++++++---
NeSPReSO2_onTemplate/preproc/climatology.py | 47 ++++++++++++++++++-----------
NeSPReSO2_onTemplate/preproc/overlap.py     | 34 +++++++++------------
NeSPReSO2_onTemplate/preproc/ssh_obs.py     | 23 +++++++++++---
```

---

## What's done: Phase 0 (benchmark harness) — committed in 64053bd

New file `scripts/bench_datacube_speed.py`. Loads the real `data/cube/gom_cube.zarr`, builds a
feature spec covering all operator kinds (mirrors `config/argo/config_argo_residual_cube.json`'s
`features` block: value/grad/laplacian/tendency/geo_uv across sst/sss/ssh), generates a seeded
synthetic profile set, calls `CubeProvider.sample()`, times it, and can save/check a golden `.npz` of
`(values, valid_mask)` at `atol=1e-6, rtol=1e-5`. Golden fixtures live in `tests/golden/`.

Key knobs: `--n-profiles N`, `--profiles-per-day K` (clusters profiles onto `ceil(N/K)` unique cube
days — real ARGO exports have several profiles/day; this is the scenario that exercises the old
geo_uv cache-key bug), `--save-golden` / `--check-golden`.

Gotcha discovered: the cube has at least one **fully-NaN plane** for `sst` at `t_idx=682`
(2016-11-13) that is *not* in `ALLOWED_MISSING_DAYS`/`missing_days` attrs — a pre-existing
data-quality gap, unrelated to this speed work, not investigated further. The benchmark's
`_day_is_usable()` filters it out by construction.

## What's done: Phase 1 (sampler hot path) — committed in 64053bd

All in `preproc/features/sampler.py` + `preproc/cube/cube_schema.py`. Fixed the geo_uv cache-key bug
(shared `grad`/`geo_uv` derived-plane cache keyed by `(name, ch, t_idx, scale_lbl)` with no
per-profile latitude in the key; `f(lat)` applied post-sample instead), batched sparse sampling by
`t_idx` instead of per-row `getrow(i)`, vectorized `time_index_of`, cached basin masks, added bounded
LRU caches for `plane_cache`/`stack_cache`/`derived_planes`, and read tendency stacks as one
chunk-aligned zarr slab when in-bounds.

Measured speedup (re-verified with corrected golden harness): unclustered 300 profiles ~271s → ~185s;
5 profiles/day (300 profiles, 60 days) 60s → 27s; 50 profiles/day (500 profiles, 10 days) 22s → 4.2s.
Verified bit-identical to pre-refactor output via golden `.npz` diffing (`atol=1e-6`), no
`operator_versions` bump. Full details/design notes are in the Phase 1 section of git history
(`git show 64053bd`) if you need them — not repeating here to keep this handoff focused on current
state.

## What's done: Phase 2 (cube build) — committed in d7519b7

All in `preproc/cube/build_cube.py`. Two changes landed, one attempted-and-reverted:

**1. Single NetCDF open per worker read.** `_worker_read_daily`/`_worker_read_ssh_day` used to call
`_indices_for_file` (open #1, reads lat/lon), `_read_hyperslab_plane` (open #2, reads data),
`_slice_coords` (open #3, re-reads lat/lon) — for SSH, `_worker_read_ssh_day` also opened once more
just to look up the time index, so 4 opens/day. Replaced with one `_read_daily_bundle()` that does
everything in a single `with nc.Dataset(path) as ds:` block. SSH still does 2 opens (one lightweight
one to resolve the per-day time index within the yearly file, one for `_read_daily_bundle`) — could
be reduced further by batching all of a yearly file's timestamps into one task instead of one task
per day, but that's a bigger structural change and wasn't attempted this session.

**2. Chunk-aligned buffered zarr writes.** `build_product`'s `ProcessPoolExecutor` loop used to do
`zarr_arr[t_idx, :, :] = plane` per day, forcing a decompress-modify-recompress of the whole
64-128-day time chunk on every single-day write (the plan's "~64× write amplification"). Now planes
are buffered in memory (a plain `dict[t_idx, plane]` — bounded by however many days a `build_product`
call actually touches, ~1.2GB peak for a full ~2600-day sst build, fine on this hardware) and flushed
via `_flush_chunked_writes()`: read the existing chunk slab once, overlay the new days at their local
offsets, write the slab back once. This preserves `--resume` correctness (previously-written days in
a partially-touched chunk survive) because it reads-before-writing rather than blind-overwriting.

**3. Attempted and reverted: precomputed regrid weights.** The plan's third Phase 2 item was to stop
rebuilding `RegularGridInterpolator` (bilinear + nearest-fallback) from scratch for every single day,
since the destination grid is fixed per product. I built a vectorized sparse-weight equivalent
(`build_regrid_weights`/`_regrid_plane_fast`, since removed) and it matched scipy bit-for-bit against
~24 randomly sampled real archive files across sst/sss/ssh — looked done. But testing against a real
end-to-end scratch cube build (not just isolated function calls) turned up silent mismatches for SSH
specifically, tracked down to an undocumented scipy behavior: `RegularGridInterpolator` sums all 4
bilinear corners **unconditionally** (so `0 * nan == nan` from a technically-zero-weight corner still
poisons the result), and at exact grid-line alignment it picks the bracket by treating the query
point as the **left** edge of the next cell, not the right edge of the previous one. Both are
undocumented implementation details, not part of scipy's public contract, and hand-replicating them
is fragile — the failure mode only shows up at exact grid alignment next to a NaN (coastline/land),
which random sampling can miss entirely (it did, initially). Given this is real science data
(SST/SSS/SSH), the risk of silently shifting values wasn't worth the speedup, so I reverted to the
original scipy-based `_regrid_plane`, unchanged from Phase 1. Saved the full postmortem to memory
(`scipy-rgi-bilinear-nan-edge-case` in the auto-memory system) so a future attempt doesn't repeat the
same random-sampling blind spot.

**Verification method used (repeatable if you touch this file again):** `git stash` the file to get
back the pre-Phase-2 code, seed a manifest that marks every date "written" except a handful of real
target dates spread across different products/time-chunks, build a scratch cube (not
`data/cube/gom_cube.zarr`) with old code, `git stash pop`, build a second scratch cube with new code,
diff the written planes at those exact `t_idx`s with `np.array_equal(..., equal_nan=True)`. This is
how the SSH regrid bug was actually caught — isolated unit-level comparisons had missed it.

```bash
cd /unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project
conda activate nespreso   # or use the direct interpreter path above
cd NeSPReSO2_onTemplate
python3 -m pytest tests/test_sampler.py tests/test_cube_validate.py -q   # 11 passed, unaffected by Phase 2
```

---

## Skipped: Phase 3

- **Phase 3 — L3/L4 rasterization** (`preproc/l3_rasterize.py`, `preproc/l4_rasterize.py`).
  Vectorize `rasterize_era5_wind_for_target` (triple nested Python loop over time×lat×lon),
  `sample_l4_ssh_patch` (per-cell `np.argmin` via `_nearest_index`, re-reads the same file per time
  bin), `bin_observations` (dict-of-lists per cell → `np.add.at` histogramming). Parallelize
  `build_l3_processed_batch` (`preproc/export_l3_cache.py`) via `ProcessPoolExecutor`.
  **Still true as of this handoff:** per `HANDOFF.md` "Next coding tasks" #1, this repo has **no raw
  L3/L4 netCDF data downloaded** (`data/raw/` still missing — checked again this session).
  Rasterization currently produces all-mask-zero bundles (the documented fast path). This phase can
  be implemented and unit-tested against synthetic in-memory arrays, but can't be validated
  end-to-end against real raw files in this environment yet. **Confirm with the user whether to
  invest here now (synthetic-only validation) or wait for the L3/L4 raw downloads.**

## Done: Phase 4 (auxiliary exporters) — uncommitted this session

All changes verified against real `data/cache/train_ready_*.pkl` caches (13 files, dataset_tag
`argo_cube`, 4145 real ARGO profile timestamps) by running old vs new implementations side by side
via `git stash` and diffing outputs — same methodology as Phase 2. No `operator_versions` bump: all
changes are bit-identical (or identical after the float32 cast the pipeline already applies).

**1. `base/split_utils.py::sample_dates`** — was a per-element Python loop
(`juld_to_datetime(v).date()` per sample, including an import + branch per call). Replaced with a
vectorized numpy `datetime64` affine transform: `isas20` is `epoch64 + juld` days; the MATLAB-datenum
branch (`argo_v2`/`argo_cube`/anything else) reduces to `datetime.fromordinal(1) + (d - 367)` days,
derived from `datetime.fromordinal(int(d) - 366) + timedelta(days=d % 1)` collapsing to a pure affine
shift for positive `d` (`int(d) + d % 1 == d`). No more `nespreso.utils.time` import needed for this
path. Verified 0 mismatches on all 13 real caches + exhaustive synthetic edge cases (near-midnight
fractional days, multiple leap/non-leap years). `v2_src` kept in the signature for compatibility but
unused.

**2. `preproc/climatology.py::_day_of_year_accurate`** — used to re-loop over the `sample_dates`
output doing `date.fromisoformat(str(d)[:10]).timetuple().tm_yday` per sample (a wasteful
datetime64 → string → date round trip). Now a direct vectorized `(dates - dates.astype('Y'))` day
offset. Verified against the old loop across every day of 10 years (leap and non-leap) — 0
mismatches.

**3. `preproc/overlap.py::days_since_1950`** — same affine-collapse idea as #1, applied directly
(this function bypasses `sample_dates`/`juld_to_datetime` and had its own loop + `_v2_datenum_to_datetime`
helper, now deleted as dead code along with the now-unused `sys`/`timedelta`/`_V2_SRC` imports).
Verified bit-identical (`max_abs_diff = 0.0`) on all 13 real caches.

**4. `preproc/ssh_obs.py`** — `sample_ssh_obs` called `_juld_to_astropy_jd` once per profile, each
call constructing its own `astropy.time.Time` object (~0.1ms overhead each, measured). **Did not**
apply the same affine-collapse trick here: verified empirically that astropy's UTC-scale JD is *not*
perfectly affine across leap-second days (e.g. 2015-06-30) — a naive `juld -> jd` formula disagreed
by up to ~0.94s there, a real algorithmic effect, not float noise. Instead added
`_juld_array_to_astropy_jd`, which builds the list of `datetime` objects (same per-element step as
before, still needed for correctness) but passes the whole list to **one** batched `Time(list).jd`
call instead of N separate `Time(scalar)` calls — ~50x faster construction, and bit-identical output
(same underlying erfa/leap-second-aware conversion, just batched). `sample_ssh_obs` now precomputes
the full JD array once instead of per-row inside the batch loop. `_juld_to_astropy_jd` (scalar) is
kept as-is since `selfcheck.py::test_ssh_obs_cached_smoke` calls it directly.

**5. `preproc/climatology.py::fit_climatology`** — replaced the per-depth-level `_ridge_solve` loop
(one `np.linalg.solve` per depth, `n_z` solves per variable) with `_ridge_solve_multi_depth`, which
groups depth levels by identical valid-mask (`np.packbits(np.isfinite(col)).tobytes()` as the group
key) and solves one shared `(XtX + alpha*I)` factorization per group as a multi-RHS system. Falls
back to one solve per group when every depth has a unique mask (no worse than before). Real cache
data (`true_profiles`) turned out to have zero NaNs (already gap-filled), so it exercises the
best-case single-group path there; separately stress-tested with synthetic data across 5 NaN
patterns (dense, uniform, per-depth-varying/simulating variable max-depth profiles, fully-unique
masks, sparse-train) — multi-RHS vs per-depth-loop differ by ~1e-16 (float64 solve-order noise from
LAPACK batching multiple RHS through one factorization) which **fully disappears** after the
`.astype(np.float32)` cast `fit_climatology` already applies to `coef` (confirmed bit-identical
post-cast across 5 random seeds). End-to-end `fit_climatology`/`eval_climatology` roundtrip on 3 real
caches: bit-identical to the old per-depth loop. The old `_ridge_solve` (single-depth) function was
deleted as dead code (no longer called, not referenced by any test).

**6. `preproc/basin_stats.py::compute_basin_daily_means`** — was a sequential double loop (unique
days × 3 products), each iteration doing an I/O-bound `xr.open_dataset` over NFS. Now flattens to a
`(day, product)` task list and runs it through a `ThreadPoolExecutor(max_workers=8)` (new keyword
arg, default 8); `retrieve_sat.select_candidate_file`/`get_product_files` are `functools.lru_cache`-backed
and read-only, so this is thread-safe. Verified identical output dict on real satellite files across
6 real dates (three from the file's own `_selfcheck()`, plus 3 more spanning different years) — 4x
wall-clock speedup (12.6s → 3.1s) in that test.

**Verification note:** `selfcheck.py` run top-to-bottom aborts early at `test_combined_pca_loss_v2`
(line 326, a `CombinedPCALoss` golden-value assertion) — confirmed **pre-existing** on the
unmodified `residual_cube` HEAD (reproduced by `git stash`-ing all Phase 4 changes and re-running
just that test), unrelated to this work. Since `selfcheck.py` has no per-test isolation, the 14
tests actually relevant to Phase 4 (`test_climatology_*`, `test_anomaly_cache_addback`,
`test_ssh_obs_cached_smoke`, `test_steric_*`, `test_field_date_split_disjoint`,
`test_field_cache_targets_match_v2`, `test_split_matches_torch_seed`,
`test_chronological_split_no_leakage`, `test_overlap_pairs`, `test_cache_schema_keys`) were run
individually and all pass. `pytest tests/test_sampler.py tests/test_cube_validate.py` (Phase 1/2's
regression gate) still passes, unaffected.

**Not committed yet** — working tree has the 5 modified files pending user review.

## Not started: Phase 5

- **Phase 5 — ingestion format split** (lowest priority per the plan — "fixed per-run cost, not
  per-sample"). `data_loader/data_loaders.py::NeSPReSODataLoader.__init__` does one `pickle.load()`
  of the entire train-ready cache including eval-only payloads (`profiles`, `true_profiles`,
  `clim_profiles`, PCA models) alongside `inputs`/`targets`/`JULD` needed by every consumer. Plan
  wants these split into an mmap-able core (`.npz` or small zarr) with eval-only payloads lazily
  loaded from a sidecar, keeping the pickle writer as a compatibility path. **Not investigated this
  session** — need to find the cache *writer* side first (likely
  `preproc/preproc_isas_sat.py::write_train_cache` or similar) before touching the loader, since both
  ends need to agree on format.

---

## Design notes worth preserving

- **The operator-version constraint applies:** `PLAN_datacube_speed.md` requires pure perf refactors
  to be numerically identical and to NOT bump `operator_versions` unless numerics genuinely changed.
  Everything landed in Phases 1–2 is bit-identical to the original (verified via golden-output
  diffing) — no operator version bumps were needed or made. This is *why* the Phase 2 regrid-weight
  attempt was reverted rather than shipped with a version bump: it wasn't a numerics change on
  purpose, it was an accidental one, caught before landing.
- `CubeProvider.weights_for()` cache is keyed by channel only (not by the specific `lats`/`lons`
  array passed in) — a **pre-existing** latent bug if `sample()` is ever called twice on the same
  provider instance with different profile sets for the same channel. Not introduced this work, not
  fixed either — flagged for whoever hits it.
- Phase 2's `_flush_chunked_writes` buffers all of a `build_product` call's planes in memory before
  flushing (not incrementally per-chunk-as-it-completes). This is simpler and was measured safe for
  this cube's grid size (~305×384, chunk_t 64-128, full time axis ~2600 days → ~1.2GB peak for an
  unresumed full sst build). If a much larger domain/resolution is ever adopted, revisit — the plan's
  incremental-flush-per-chunk alternative would bound memory tighter at the cost of more code.

## Useful commands to reorient in a fresh session

```bash
cd /unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project/NeSPReSO2_onTemplate
conda activate nespreso   # if python imports fail, use /conda/jmiranda/miniconda/envs/nespreso/bin/python3 directly
git log --oneline -5      # confirm Phase 1 (64053bd) + Phase 2 (d7519b7) are present
python3 -m pytest tests/test_sampler.py tests/test_cube_validate.py -q
python3 scripts/bench_datacube_speed.py --n-profiles 300 --check-golden
```

Next step for a fresh session: Phase 4 changes are implemented and verified but **uncommitted** —
review `git diff` first. Then ask the user whether to (a) commit Phase 4, (b) proceed into Phase 5
(cache format split — needs finding the writer side first, see above), (c) proceed into Phase 3
(synthetic-only validation, since no raw L3/L4 archives are present), or (d) stop here.
