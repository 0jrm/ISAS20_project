Use conda env nespreso when running python code.

## Findings — where the time actually goes

**1. `CubeProvider.sample()` — the `geo_uv` cache key is per-profile (worst offender).** In `sampler.py`, derived planes are cached by key, but for `geo_uv` the key includes `round(lat_i, 3)`:

```python
dkey = (name, ch, t_idx, scale_lbl, round(lat_i, 3))
```

So a *full-grid* smoothed gradient (two normalized Gaussian filter passes over the entire SSH plane) is recomputed for nearly every profile, because profile latitudes are essentially unique. This makes geo_uv O(N_profiles × full-plane-filter) instead of O(unique_days). The physics doesn't require it: `u = (-g/f)·gy`, `v = (g/f)·gx`, and `f` depends only on the profile latitude — a scalar. You can compute and cache `gx`, `gy` once per `(channel, t_idx, scale)` (same key as `grad`), bilinearly sample them, and divide by `f(lat_i)` *after* sampling. For ~10k profiles this alone is likely a 100–1000× reduction on the geo_uv features, and geo_uv is in the residual feature spec.

**2. Per-row sparse sampling in the operator loop.** The inner loop does, per profile per feature:

```python
wt = w.getrow(i)
val = float((wt @ flat).sum())
wsum = float((wt @ np.isfinite(flat)...).sum())
```

`getrow` on CSR is expensive, and this runs N×F times. Since the derived plane is shared by all profiles at the same `t_idx`, the right structure is: group profile indices by `t_idx`, and for each derived plane do one batched `weights[idx_group] @ flat` (a single sparse-dense matmul). That converts millions of tiny sparse ops into a few hundred matmuls. The existing `sample_plane()` already does exactly this batched form — it's just not used in the operator loop.

**3. Zarr read amplification.** `plane(ch, t_idx)` reads one day, but chunks are `(chunk_t=64, H, W)` — every new `t_idx` decompresses a 64-day chunk to extract one plane. `tendency` compounds this: it reads `window` consecutive days per unique t_idx. Profiles aren't sorted by time, and `plane_cache`/`stack_cache`/`derived_planes` are unbounded dicts (memory risk on full runs). Fixes: sort profiles by `t_idx` before the loop, read chunk-aligned slabs (`root[ch][t0:t1]`), and put a bounded LRU on the caches.

**4. `build_cube.py` writes one day at a time into 64-day chunks.** `zarr_arr[t_idx, :, :] = plane` triggers read-modify-write of the whole chunk for every daily write — ~64× write amplification, plus contention since workers return planes to a single writer loop. Buffering planes and flushing chunk-aligned slabs (or writing to per-chunk buffers keyed by `t_idx // chunk_t`) removes it.

**5. Regridding rebuilds interpolators per plane.** `_regrid_plane` constructs a `RegularGridInterpolator`, a meshgrid, and a second nearest-neighbor interpolator (for NaN fill) for every single day — but source and destination grids are fixed per product. Precompute bilinear index/weight arrays once per product (same idea as `build_bilinear_weights`) and apply them as array ops. Also `_worker_read_daily` opens each NetCDF file three times (`_indices_for_file`, `_read_hyperslab_plane`, `_slice_coords`) — one open suffices, and the lat/lon slices are identical across files of the same product so they can be computed once and shipped to workers.

**6. L3/L4 rasterization is pure-Python triple loops.** `rasterize_era5_wind_for_target` loops time × lat × lon appending scalars; `sample_l4_ssh_patch` calls `_nearest_index` (a full `argmin`) per grid cell per time bin, and re-opens/re-reads the same NetCDF for every time bin. All of this vectorizes with `searchsorted`/boolean masks, and files should be read once per target. `build_l3_processed_batch` builds samples strictly sequentially — an obvious `ProcessPoolExecutor` candidate since samples are independent.

**7. Smaller but cheap wins.** `time_index_of()` allocates the full time axis and does `np.where` per call — it's just `(d - TIME_START).astype(int)` with a bounds check. `basin_mean` rebuilds the meshgrid basin mask on every call (cache per channel). `_ridge_solve` in climatology solves per depth level; when NaN patterns allow, factor `XtX + αI` once and solve multi-RHS across all depths. `days_since_1950` and `sample_ssh_obs` do per-element Python loops with astropy/datetime conversions that vectorize. `basin_stats` processes unique days sequentially — parallelizable.

**8. Ingestion.** `NeSPReSODataLoader` unpickles the entire cache including eval-only payloads (`profiles`, `true_profiles`, `clim_profiles`, PCA models) on every train launch. Splitting the cache into a training core (inputs/targets/JULD, loadable via `np.load` mmap) and a lazily-loaded eval sidecar would cut startup time and memory. This is lower priority — it's a fixed per-run cost, not per-sample.

One important constraint: the cache hash includes `operator_versions`. Pure-performance refactors must produce numerically identical outputs and keep versions unchanged (so existing caches stay valid); anything that changes numerics (it shouldn't, but e.g. filter-order changes) must bump the operator version to force re-export.

## Proposed work plan

**Phase 0 — Measurement and safety net (do first, ~1 day).** Build a benchmark harness: fixed 500-profile subset with a pinned anchor date, timed end-to-end for cube build (one product, one month), residual cache export, and dataloader startup. Add per-stage timing (plane reads, operator application, sampling) inside `sample()`. Capture a golden `FeatureTable` (values + valid_mask) and golden cube slabs; every subsequent phase must reproduce them within `atol=1e-6` (byte-identical where the refactor shouldn't touch math). Run `py-spy`/cProfile on the current code to confirm the ranking above before committing to it.

**Phase 1 — Sampler hot path (~2–3 days, expected order-of-magnitude gain on residual export).** Fix the geo_uv cache key by sampling `gx`/`gy` planes and applying `f(lat)` post-sample; batch profile sampling by `t_idx` using one sparse matmul per derived plane; sort profiles by time index; vectorize `time_index_of`; cache basin masks; add bounded LRU to `plane_cache`/`derived_planes`/`stack_cache`; slab-read tendency stacks. Verify golden outputs, keep operator versions untouched.

**Phase 2 — Cube build (~2 days).** Single-open reads in workers with precomputed per-product slices; precomputed regrid weight arrays (bilinear + nearest-fallback indices) reused across all days; chunk-aligned buffered Zarr writes; batch manifest updates (currently the manifest dict grows and is rewritten wholesale at the end — fine, but per-entry bookkeeping in the loop can be simplified). Re-run validation A-V1…A-V5 as the regression gate.

**Phase 3 — L3/L4 path (~2 days, only if you're actively using it).** Vectorize `rasterize_era5_wind_for_target`, `sample_l4_ssh_patch` (searchsorted-based nearest indices, read each file once), and `bin_observations` (replace the per-obs dict with `np.add.at` histogramming). Parallelize `build_l3_processed_batch` across sample indices.

**Phase 4 — Auxiliary exporters (~1–2 days).** Vectorize datenum/JD conversions in `ssh_obs`, `overlap`, `climatology.design_matrix`; multi-RHS ridge solve; parallelize `compute_basin_daily_means` over unique days.

**Phase 5 — Ingestion format (~1–2 days, optional).** Split train-ready pickle into an mmap-able core (`.npz` or small Zarr) plus lazy eval sidecar; keep the pickle writer as a compatibility path so existing caches still load.

**Sequencing rationale:** Phase 1 dominates because cache export is the stage you re-run most (every feature-spec or split change), and the geo_uv defect is a genuine bug-shaped bottleneck, not just inefficiency. Phase 2 matters when `DATA_REVISION` bumps force cube rebuilds. Phases 3–5 are proportional to how much you use those paths.
