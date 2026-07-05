# Session handoff — datacube speed plan execution

**Branch:** `residual_cube`
**Date:** 2026-07-05
**Source plan:** [`PLAN_datacube_speed.md`](PLAN_datacube_speed.md) (repo root) — read that first, this handoff assumes it.
**Code home:** `NeSPReSO2_onTemplate/` (all paths below are relative to that dir unless stated otherwise)
**Conda env:** `nespreso` for all python commands. On this machine `conda activate nespreso` may not
put the right python first on `PATH` — if `import astropy`/`import zarr` fails, call the interpreter
directly: `/conda/jmiranda/miniconda/envs/nespreso/bin/python3`.

User asked to execute the full plan (Phases 0–5). **Phases 0, 1, and 2 are done, verified, committed,
and pushed to `origin/residual_cube`.** Phases 3–5 are not started. Nothing is mid-flight — this is a
clean stopping point.

```
d7519b7 Phase 2 datacube speed: single NetCDF open per read, chunk-aligned buffered writes
64053bd Phase 1 datacube speed: fix geo_uv cache-key bug, batch sampling by day
2cc6c2c cube first draft and runs
```

Working tree is clean (`git status` has nothing pending) as of this handoff.

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

## Not started: Phases 3–5

Per the plan (`PLAN_datacube_speed.md`):

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

- **Phase 4 — auxiliary exporters.** `preproc/climatology.py::_ridge_solve` solves one ridge
  regression per depth level in a loop; plan suggests factoring `XtX + αI` once and solving
  multi-RHS when NaN patterns allow (needs a "group depths by identical valid-mask" step first, since
  masks can differ per depth). `preproc/basin_stats.py::compute_basin_daily_means` already dedupes to
  unique days but processes them sequentially (I/O bound — `ThreadPoolExecutor` candidate).
  `preproc/overlap.py::days_since_1950` loops per-element through
  `nespreso.utils.time.datenum_to_datetime` (external repo, `/unity/g2/jmiranda/v2-nespreso/src`) —
  vectorizable as a pure affine transform, **but** note the original's
  `timedelta(days=datenum % 1)` rounds to microsecond precision, so a fully vectorized version would
  be *more* precise than the original by sub-microsecond amounts — verify this is within tolerance
  for downstream matched-eval code before landing it. `preproc/ssh_obs.py::sample_ssh_obs` is
  I/O-bound against `retrieve_satellite_data` — lower priority, not easily vectorizable without
  changing that API.

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

Next step for a fresh session: ask the user whether to proceed into Phase 3 (and if so, whether to
scope it to synthetic-data validation only, given no raw L3/L4 archives are present), or to jump to
Phase 4/5, or stop here.
