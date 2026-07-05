# Design Spec — L4 Residual Patch Model & Regional Data-Cube Ingestion

**Doc:** DESIGN-2026-07-04-l4-residual-cube (v1, standalone)
**Companion docs:** HANDOFF-2026-07-04 (current L4 status), HANDOFF-2026-07-03 (stale-sat root cause)
**Owners:** ML: TBD · Data/infra: TBD
**Repo root:** `/unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project`
**Working dir:** `NeSPReSO2_onTemplate/` · Conda env: `nespreso`

---

## 0. Summary

The L4 patch model (T RMSE 1.857) currently loses to climatology (1.657) and is far from the golden point baseline (0.416). Diagnosis from the 2026-07-04 handoff: (a) inputs are unstandardized, (b) the conv branch average-pools away the center-pixel signal, (c) the scalar encoder excludes local sss/sst/ssh, and (d) the NetCDF→HDF5 pipeline is query-major, slow to iterate, and structurally prone to the stale-file bug class.

This spec replaces the pipeline and model with three components:

1. **Regional space–time cube (Component A):** each NetCDF archive is read **once**, sliced to the GoM domain, and materialized as a chunked Zarr cube with explicit time coordinates. All downstream work is array math against the cube; NetCDF is never touched after the build.
2. **Feature operator layer (Component B):** a declarative, versioned registry of physically motivated operators (value, gradient at two scales, Laplacian, temporal tendency, geostrophic u/v) computed analytically on the cube and sampled at profile locations. Replaces the raw 525-D patch with a ~40–50-D named, z-scored feature vector.
3. **Point-anchored residual model (Component C):** output = frozen/warm-started point-model PCs + zero-initialized residual head over the new features. At initialization the model **is** the 0.416 baseline; training can only improve on it (subject to generalization, protected by val early stopping).

**Design principle:** the golden point model's inputs are a strict subset of ours, and interpolation/differentiation are computed in closed form rather than learned from ~2.9k samples. The baseline becomes the floor by construction; the experiment tests whether differential surface structure adds subsurface skill.

---

## 1. Goals, non-goals, success criteria

### 1.1 Goals

- **G1** — New models never underperform the golden point baseline on the chronological test split (n=623) by construction (residual anchoring).
- **G2** — Cache re-export (feature recipe change → new `train_ready` pickle) runs in **seconds to low minutes** on a login/compute node, from the cube, with no NetCDF access.
- **G3** — Feature experiments (add/remove operators, change scales) are **config diffs**, not code changes.
- **G4** — Stale/missing satellite data fails **loudly at cube build time** (contiguity assertion), eliminating the `select_candidate_file` nearest-file clamp bug class.
- **G5** — Per-feature attribution (permutation importance) is available for every trained model.

### 1.2 Non-goals

- Replacing the point baseline pipeline (v2 pickle scalars stay as-is).
- L3/swath ingestion (design leaves a seam for it; not implemented).
- New profile QC, domain changes, or split changes — frozen at 4145 profiles, GoM 18–31°N / −98 to −81°W, chronological 70/15/15.
- Architecture search beyond the residual head (conv redesign only if this fails; see §9).

### 1.3 Success criteria (in order)

| Tier | Criterion | Metric (test, native depth) |
|------|-----------|------------------------------|
| S0 | Sanity: residual model at init reproduces point baseline | T = 0.416 ± numerical noise, S = 0.072 |
| S1 | Beat climatology | T < 1.657, S < 0.216 |
| S2 | Beat ANOM-point | T < 0.647, S < 0.100 |
| S3 | Beat golden point, **paired-significant** | T < 0.416 with paired per-profile test (see §8.4) |

Realistic target for S3: 5–15% RMSE reduction (T ≈ 0.35–0.40), concentrated at eddy peripheries and rapid-evolution periods. S3 is the justification for the L4 program; S1–S2 falling out automatically from anchoring is expected, not a result.

---

## 2. System overview

```
                     ┌─────────────  ONE-TIME / APPEND-ONLY  ─────────────┐
NetCDF archives      │  build_cube.py (per product, parallel over files)  │
 OSTIA SST (daily)   │    - hyperslab GoM+margin, float32                 │
 CMEMS SSS (daily)   │    - unit normalization (K→°C, etc.)               │──► gom_cube.zarr
 CMEMS SSH (yearly)  │    - explicit time axis, NaN for missing days      │    /{sst,sss,ssh}(t,y,x)
 GEBCO bathy(static) │    - contiguity + coverage assertions              │    /bathy(y,x)
                     └────────────────────────────────────────────────────┘
                                                │
                     ┌──────────  FAST ITERATION (seconds)  ──────────────┐
                     │  feature layer: operators on cube (whole-plane)    │
 v2 pickle           │    value/grad/laplacian/tendency/geo_uv per chan   │
 (4145 profiles,  ──►│  sampler: precomputed bilinear weights per grid    │──► train_ready_<hash>.pkl
  targets, splits)   │  standardizer: train-split z-score, stored w/cache │    (named features + targets)
                     └────────────────────────────────────────────────────┘
                                                │
                     ┌────────────────────────────────────────────────────┐
 point ckpt ────────►│  Component C: point backbone (frozen/warm) +       │──► saved/models/.../model_best.pth
 (argo16_scales)     │  zero-init residual head on feature vector         │
                     └────────────────────────────────────────────────────┘
```

The intermediate per-profile HDF5 (`satellite_NeSPReSO_v2_ARGO_GoM.h5`) and the 42-batch regeneration flow are **retired** for the new models (kept read-only for legacy patch-l4 reproducibility).

---

## 3. Component A — Regional space–time cube

### 3.1 Storage format and layout

Single Zarr store: `data/cube/gom_cube.zarr` (Zarr v2, `zarr-python` + `numcodecs`; xarray-compatible).

| Group/array | Dims | Grid | Approx shape | Chunking (t,y,x) | Size (f32) |
|-------------|------|------|--------------|-------------------|------------|
| `sst` | (time, lat, lon) | 0.05° | (~2557, 268, 348) | (64, full, full) | ~0.95 GB |
| `sss` | (time, lat, lon) | 0.125° | (~2557, 108, 140) | (64, full, full) | ~0.15 GB |
| `ssh` | (time, lat, lon) | 0.25° | (~2557, 56, 72) | (128, full, full) | ~0.04 GB |
| `bathy` | (lat, lon) | 0.05°* | (268, 348) | (full, full) | ~0.4 MB |

\* GEBCO (~0.004°) is block-averaged to the SST grid at build time; keep a second full-res center-value array only if bathy-gradient features are later wanted.

**Domain:** 18–31°N, −98 to −81°W **plus margin** = `max(regional_stencil_halo, spatial_pad) + 2` cells per product grid, so gradient stencils and 5×5 patches near the boundary never hit the edge. Margin is recorded in attrs; the sampler masks features whose stencil would leave the padded domain (should be none for GoM profiles).

**Coordinates (attrs + coordinate arrays):**
- `time`: contiguous daily `datetime64[D]`, **2015-01-01 through 2022-03-01** (covers first profile − temporal_pad through last profile 2022-02-27 + slack). One plane per calendar day; missing source days are explicit all-NaN planes plus an entry in `attrs["missing_days"]`.
- `lat`, `lon`: native product grids, ascending. No regridding between products — each channel keeps its native resolution; harmonization happens at sampling time (per-product bilinear weights), not storage time.
- attrs per array: `product_id`, `product_version`, `source_var`, `native_units`, `stored_units`, `unit_transform` (e.g. `"K-273.15"` applied at build), `build_git_sha`, `build_date`, `cube_schema_version`.

**Time coordinate discipline:** cube time is plain calendar `datetime64`. The MATLAB-datenum (+366) pitfall lives only at the profile-matching boundary: the sampler converts profile dates from HDF5 `stations/julian_date` (astropy JD) — **never** from cache JULD — in exactly one function, `time_index_of(profile_date)`, with a unit test pinning known profiles.

### 3.2 Build process (`preproc/cube/build_cube.py`)

Per product, embarrassingly parallel over files, I/O-bound:

1. Enumerate archive files; parse date(s) from filename (OSTIA/SSS: one day/file; SSH: yearly files, iterate internal time axis).
2. Per file/worker: open with `netCDF4` (not `xarray.open_mfdataset` — per-file metadata parsing dominates at 2.5k files), read **only** the GoM+margin hyperslab via lazy slicing, apply unit transform, `scale_factor/add_offset` decode, `_FillValue`→NaN.
3. Workers return `(date, plane)`; a single writer process places planes at `time_index(date)` in the pre-allocated Zarr array (avoids concurrent-write coordination; chunked (t=64, full, full) so each write touches one chunk column).
4. `--resume`: skip dates already marked written in a sidecar manifest (`build_manifest.json`: per-date status + source filename + file mtime/size). Re-run with `--force-dates 2021-06-*` to rebuild specific ranges.

**Parallelism:** `multiprocessing` pool, 8–16 workers (`srun --ntasks=1 --cpus-per-task=16`). Expected wall time: OSTIA (largest, 2.5k files × ~360 KB hyperslab) minutes, not hours; SSS/SSH faster; GEBCO one-shot.

**CLI:**
```
python preproc/cube/build_cube.py --product ostia --workers 16 [--resume] [--force-dates ...]
python preproc/cube/build_cube.py --product all --workers 16
python preproc/cube/build_cube.py --validate            # assertions only, no writes
```

### 3.3 Validation (build-time, hard failures)

Replaces the after-the-fact stale-sat fingerprint diagnostics:

- **A-V1 Contiguity:** time axis strictly daily, no gaps, spans required range. Missing source days must be explicitly whitelisted in config (`allowed_missing_days`), else build fails.
- **A-V2 Coverage:** last required day (2022-03-01) present per product.
- **A-V3 Anti-stale:** for each product, assert no run of `k ≥ 3` consecutive identical planes outside the whitelist (catches upstream frozen-product bugs; OSTIA/CMEMS L4 are daily-varying by construction).
- **A-V4 Physical range:** per-channel plausibility bounds (SST −3…35 °C, SSS 20…40, SSH −1.5…1.5 m, bathy 0…4500 m); >0.1% out-of-range → fail with report.
- **A-V5 NaN budget:** ocean-mask NaN fraction per plane below threshold (land mask derived from bathy ≤ 0); spikes flagged.

`--validate` output goes to `data/cube/validation_report.json`; CI-style: nonzero exit on any failure.

### 3.4 Incremental extension

New year of data → run build for the new date range (append along time axis, Zarr supports resize), re-run `--validate`. New profiles need **no** cube work unless outside the time/space envelope. `cube_schema_version` bumps only on layout/unit changes; a data append bumps `attrs["data_through"]`, which participates in the downstream cache hash (§4.6).

---

## 4. Component B — Feature operator layer

### 4.1 Physical rationale (what the features are and why)

The surface constrains the subsurface through local differential properties of the fields at the profile point — a truncated space–time Taylor expansion ("jet") per channel — not through raw pixel neighborhoods:

| Feature family | Physics | Primary channel(s) |
|----------------|---------|--------------------|
| Value at point | Dynamic height ↔ thermocline displacement (LC eddies); surface T/S state | SSH ≫ SST, SSS |
| Gradient (local + regional scale) | Geostrophic velocity (SSH); fronts, river-plume edges (SST, SSS) | SSH, SST |
| Laplacian / curvature (regional) | Relative vorticity proxy — eddy core vs periphery discrimination | SSH |
| Temporal tendency (7 d) | Eddy propagation / intensification | SSH, SST |
| Geostrophic u, v | Rotated SSH gradient in dynamical units; f from profile lat | SSH-derived |

Two spatial scales: **local** = product grid scale (post-smoothing, §4.3), **regional** = ~1° ≈ GoM first-baroclinic Rossby radius (eddy scale). Expected signal ranking: SSH features ≫ SST gradients > SSS gradients (L4 SSS is heavily smoothed; likely prunable — that must be an ablation config, not a code decision).

### 4.2 Operator registry

Module: `preproc/features/operators.py`. Each operator is a named, versioned, unit-aware pure function on cube arrays:

```python
@register("grad", version=2, units_fn=lambda u: f"{u}/m")
def grad(field_t: DayPlaneStack, scale_deg: float, grid: Grid) -> tuple[Plane, Plane]:
    """Gaussian-derivative dF/dx, dF/dy at scale_deg; metric-corrected
    (dx = R·cos(lat)·dlon, dy = R·dlat). Returns physical units (per metre)."""
```

**Registry contract:** `name`, `version` (participates in cache hash), declared input channel(s), output feature names (e.g. `ssh.grad_x@1.0deg`), output units, `scale_deg` parameter, NaN policy (see §4.4). Adding an operator = one decorated function + config entry; no changes to sampler, cache export, or training code.

**Initial operator set:**

| Operator | Params | Outputs/channel | Notes |
|----------|--------|-----------------|-------|
| `value` | scale σ | 1 | Gaussian-smoothed at σ, sampled at point |
| `grad` | σ ∈ {local, 1.0°} | 2 (x, y) | Gaussian derivative — well-posed differentiation of noisy L4 data; **never** raw finite differences at grid scale (would partly measure the OI mapping kernel, not the ocean) |
| `laplacian` | σ = 1.0° | 1 | ∇²F, metric-corrected |
| `tendency` | window = 7 d | 1 | Robust (Theil–Sen or LSQ) slope per pixel over the 7-day stack ending on profile date, then sampled |
| `geo_uv` | σ ∈ {local, 1.0°} | 2 | u_g = −(g/f)∂η/∂y, v_g = (g/f)∂η/∂x; f at profile latitude; SSH only |

**Local σ per product** = the product's effective decorrelation scale, not grid spacing (config defaults: SST 0.15°, SSS 0.35°, SSH 0.4°; tunable).

### 4.3 Evaluation strategy on the cube

Whole-plane computation, then point sampling — never per-profile stencils:

1. Deduplicate: the set of distinct (product, date) planes needed across all 4145 profiles × 7-day windows is far smaller than 4145×7.
2. For each needed plane and operator/scale: compute the derived field for the whole GoM plane (`scipy.ndimage.gaussian_filter` and Gaussian-derivative kernels; vectorized, ~ms per plane at these sizes). Optional LRU/disk memo keyed by (product, date, operator, version, σ) under `data/cube/derived/` — cheap enough that memoization is an optimization, not a requirement.
3. Sample derived fields at profile lat/lon with precomputed bilinear weights: one sparse (n_profiles × 4) weight matrix per product grid, built once — sampling any field is a sparse matvec.
4. `tendency` operates on the 7-plane stack per profile date; dedup by date makes shared windows free.

### 4.4 NaN / coastal policy

- Gaussian filtering uses **normalized convolution** (filter value·mask and mask separately, divide) so land does not bleed into ocean values.
- A sampled feature is NaN if the effective valid-data weight at the sample point < 0.5.
- Cache stores a per-feature validity mask; models receive mean-imputed values (train-split mean, i.e. 0 after z-scoring) **plus** the mask is available for a value+mask ablation. Silent NaN→0 on raw scales (current production behavior) is removed.
- Basin means: masked spatial mean over the GoM polygon (existing basin exclusion at 23°N, −88°W) computed on the cube — replaces `basin_stats.py` per-day reads. **Missing basin mean is a hard error, not zero-fill** (handoff §7 guard).

### 4.5 Feature vector assembly and config

Feature spec lives in the training config:

```json
"features": {
  "spec_version": 1,
  "scalars": ["timecos","timesin","latcos","latsin","loncos","lonsin",
               "basin_sss","basin_sst","basin_ssh","bathy_depth"],
  "point_backbone_inputs": ["sss.value@local","sst.value@local","ssh.value@local"],
  "operators": [
    {"op":"value","channels":["sst","sss","ssh"],"scales":["local"]},
    {"op":"grad","channels":["sst","sss","ssh"],"scales":["local","1.0deg"]},
    {"op":"laplacian","channels":["ssh"],"scales":["1.0deg"]},
    {"op":"tendency","channels":["sst","ssh"],"window_days":7},
    {"op":"geo_uv","channels":["ssh"],"scales":["local","1.0deg"]}
  ]
}
```

Resulting vector (default spec): 10 scalars + 3 values + 12 gradients + 1 Laplacian + 2 tendencies + 4 geo-uv ≈ **32 named features** (+ point-backbone 9-D, which overlaps scalars/values). Every column has a name, units, and operator version stored in cache metadata. Ablations (e.g. drop SSS gradients) are config diffs (G3).

### 4.6 Standardization and cache

- Z-scoring is **structural**: `export_feature_cache.py` computes per-feature mean/std on the **train split only**, stores them in the cache alongside data, and applies them. (Generalizes the E2 `make_std_cache.py` result: train RMSE 0.648 → 0.150.) Circular scalars (harmonics) pass through unscaled.
- Cache hash = SHA-256 (first 12 hex) of `{feature_spec (incl. operator versions), cube_schema_version, cube data_through, io, outputs, split_def}` — extends the existing hashing scheme; any recipe or data change invalidates cleanly.
- Output: `data/cache/train_ready_<hash>.pkl` with `{X (n×d float32), feature_names, units, valid_mask, mu, sigma, targets, split_indices, provenance}`.
- CLI mirrors current flow: `python preproc/export_feature_cache.py -c <config> [--force]`; `train.py` auto-builds if missing (existing behavior preserved).

---

## 5. Component C — Point-anchored residual model

### 5.1 Architecture

```
x_point (9-D, from feature layer: harmonics + local sss/sst/ssh)
   │
   ▼
Point backbone  = PatchConvMLP loaded from
                  saved/models/NeSPReSO2_ARGO_GoM/argo16_scales/model_best.pth
                  (frozen in Phase 1; optionally unfrozen at low LR in Phase 2)
   │
   ▼  ŷ_point (32 PCs: 16 T + 16 S)
   │
x_feat (~32-D named features, z-scored)          ŷ = ŷ_point + g ⊙ Δ(x_feat)
   │                                                        ▲
   ▼                                                        │
Residual head Δ: MLP [32 → 128 → 128 → 32],                 │
GELU, dropout 0.1, LayerNorm on input                       │
Gate g: per-output scalar (32-D), init 0  ────────► (ReZero-style)
```

**Anchoring invariants:**
- **C-I1:** with `g = 0`, ŷ ≡ ŷ_point for all inputs → test RMSE = 0.416/0.072 exactly at init (S0 sanity check, automated in CI, §8.1).
- **C-I2:** point backbone consumes the **same standardization** as its original training (its scaler is loaded from the point run, not the new cache) — the feature layer emits `point_backbone_inputs` in raw physical units and the backbone applies its own scaling internally. This isolates the two standardization regimes.
- Gate is per-PC (not global) so the model can accept residual corrections on low-order PCs while ignoring noise-dominated high-order ones.

**PC-space consistency:** the residual head must emit corrections in the **same PCA basis** as the point checkpoint. The feature cache therefore reuses the point run's PCA components (T/S means + component matrices serialized with the point checkpoint) rather than refitting. `refit_pca` is disallowed in residual configs (config validation error). Anomaly-target variant (`config_argo_residual_anom.json`) is out of scope for Phase 1 — raw-PC residual first; revisit only if S3 fails.

### 5.2 Training protocol

| Item | Value |
|------|-------|
| Loss | `combined_pca_loss`, `pc_mse_only` (unchanged) |
| Optimizer | AdamW, lr 1e-3 (head + gate), weight decay 1e-4 |
| Backbone | **Phase 1: frozen.** Phase 2 (only if S3 within reach but not achieved): unfreeze at lr 1e-5, 10× warmup delay |
| Batch size | 512 (`--bs 512`) |
| Epochs / early stop | 8000 / patience 500 on min val_loss (unchanged trainer) |
| Monitoring extras | log ‖g‖₁ per epoch (gate opening = residual signal detected); log per-feature gradient norms |
| Config | `config/argo/config_argo_residual.json`, run id `residual_v1` |

```
cd NeSPReSO2_onTemplate
python preproc/export_feature_cache.py -c config/argo/config_argo_residual.json --force
python train.py -c config/argo/config_argo_residual.json --bs 512 -id residual_v1
python eval_run.py -c config/argo/config_argo_residual.json \
  -r saved/models/NeSPReSO2_ARGO_GoM_residual/residual_v1/model_best.pth \
  --split test --out saved/eval_residual_v1_test.json
```

Explicit `-r` paths always (discover_checkpoint pitfall, handoff §5.4, stands).

### 5.3 Interpretability deliverables (per trained model)

- Permutation importance per named feature on val split → `saved/eval_*_featimp.json` + bar plot.
- Final gate values `g` per PC.
- Error stratification: per-profile RMSE vs |∇SSH| and vs distance-to-eddy-edge proxy (|∇²SSH|) — tests the "gains at eddy peripheries" hypothesis directly.

---

## 6. Code layout and interfaces

```
NeSPReSO2_onTemplate/
  preproc/
    cube/
      build_cube.py          # Component A CLI
      cube_schema.py         # domains, grids, versions, unit transforms
      validate_cube.py       # A-V1..A-V5
    features/
      operators.py           # registry + operator impls
      sampler.py             # bilinear weight matrices, time_index_of()
      export_feature_cache.py# Component B CLI (replaces export_argo_l4_cache for new models)
  model/
    residual.py              # PointAnchoredResidual (loads point ckpt, gate, head)
  config/argo/
    config_argo_residual.json
  tests/
    test_cube_validate.py, test_operators.py, test_sampler.py, test_residual_init.py
```

**Key interface (the seam for future L3/swath inputs):**

```python
class FieldProvider(Protocol):
    def sample(self, feature_spec, lats, lons, dates) -> FeatureTable: ...
# CubeProvider implements it for L4 Zarr; a future SwathProvider can implement
# the same protocol over irregular sampling (simplex/triangle interpolation
# earns its keep there, not on regular L4 grids).
```

Legacy `retrieve_sat.py` / `generate_argo_satellite_data.py` / per-profile HDF5: frozen, kept for legacy patch-l4 reproduction only; new code must not import them.

---

## 7. Milestones

| # | Deliverable | Depends on | Est. effort | Exit criterion |
|---|-------------|------------|-------------|----------------|
| M0 | Z-score retrain on **existing** pipeline (`patch_l4_fixedsat_std`) — already planned in handoff §7; do not block on this spec | — | ~1 day | eval JSONs in `saved/`; informs urgency of M1–M5 |
| M1 | Cube build + validation (Component A) | — | 2–3 days | `--validate` clean; A-V1..V5 pass; build wall-time recorded |
| M2 | Sampler + bilinear weights + time mapping | M1 | 1–2 days | test: cube-sampled `value@local` matches v2-pickle point scalars within interp tolerance on ≥95% of profiles (discrepancies itemized) |
| M3 | Operator registry + initial operator set | M1 | 3–4 days | unit tests vs analytic fields (§8.2) |
| M4 | Feature cache export + z-scoring + hashing | M2, M3 | 1–2 days | cache builds < 2 min from cube (G2); named metadata present |
| M5 | Residual model + S0 sanity | M4 | 2 days | C-I1 automated test passes (0.416 at init) |
| M6 | Train `residual_v1`, eval, feature importance, error stratification | M5 | 1–2 days + GPU | S1/S2 verified; S3 paired test reported either way |
| M7 | Ablations: drop SSS grads; local-only vs +regional; gate-per-PC vs scalar | M6 | 2–3 days | ablation table in `saved/results/` |

---

## 8. Testing and acceptance

### 8.1 CI-style checks (run per change)

- `test_residual_init.py`: fresh residual model, gate=0 → test predictions bitwise-equal (fp32 tolerance) to point checkpoint predictions.
- `test_sampler.py`: known synthetic plane (e.g. F = a·lat + b·lon) → bilinear sample exact; `time_index_of` pinned against 5 known profiles (JD vs MATLAB-datenum regression guard).
- Config validation: residual configs reject `refit_pca: true`; feature specs reject unknown operator names/versions.

### 8.2 Operator correctness (analytic fields)

For F(lat, lon) = sin(k·x_m)·cos(l·y_m) on each product grid: `grad`, `laplacian`, `geo_uv` must match closed-form derivatives within Gaussian-scale-dependent tolerance, including the cos(lat) metric factor (test at 18°N and 31°N to catch missing metric correction). `tendency` tested on a linear-in-time synthetic stack.

### 8.3 Pipeline equivalence

M2 exit criterion above, plus: recompute 100 random legacy 5×5 patch center values from the cube and compare to the (post-fix) HDF5 — bounded discrepancy audit before retiring the old path for new work.

### 8.4 Statistical acceptance for S3

Headline RMSE difference at n=623 is small relative to noise. Required reporting: per-profile squared-error differences (residual vs point), paired test (Wilcoxon signed-rank primary; paired t secondary), 95% bootstrap CI on ΔRMSE, plus depth-resolved RMSE curves (existing `fixedsat_vs_golden_depth_rmse.png` style) and 1° bin maps. "Beats golden" claims require the paired test, not the point estimate.

---

## 9. Risks and mitigations

| Risk | Likelihood | Mitigation |
|------|------------|-----------|
| Residual gate never opens (no marginal signal in gradients/tendencies) | Medium | This is a *finding*, not a failure mode of the design — it cleanly falsifies the L4 spatial hypothesis at far lower cost than another conv redesign. Error stratification (§5.3) says where to look next. |
| Point backbone / new-cache standardization mismatch corrupts anchor | Low | C-I2 isolation + S0 automated test catch it immediately. |
| Gaussian σ mis-set (too small → OI-kernel noise; too large → smears eddy edges) | Medium | σ is config; M7 ablation sweeps {0.5×, 1×, 2×} defaults; SSH effective resolution literature values as priors. |
| Missing-day whitelist hides real archive gaps | Low | Whitelist requires explicit per-date entries in reviewed config; A-V2 still enforces endpoint coverage. |
| Cube grows stale vs upstream product revisions (CMEMS reprocessing) | Low | `product_version` in attrs + manifest mtimes; rebuild is cheap (minutes) by design. |
| Overfitting residual head despite small dim (~32-D, 2.9k samples) | Low–Med | dropout, weight decay, val early stop, per-PC gate; if seen, shrink head to [32→64→32]. |

---

## Appendix A — Default feature vector layout (residual_v1)

| Cols | Block | Names (pattern) |
|------|-------|-----------------|
| 0–9 | Scalars | timecos, timesin, latcos, latsin, loncos, lonsin, basin_sss, basin_sst, basin_ssh, bathy_depth |
| 10–12 | Values | {sst,sss,ssh}.value@local |
| 13–24 | Gradients | {sst,sss,ssh}.grad_{x,y}@{local,1.0deg} |
| 25 | Curvature | ssh.laplacian@1.0deg |
| 26–27 | Tendencies | {sst,ssh}.tendency@7d |
| 28–31 | Geostrophic | ssh.geo_{u,v}@{local,1.0deg} |

(9-D point-backbone inputs are drawn from cols 0–5 + 10–12 in raw units per C-I2.)

## Appendix B — Size/perf budget

| Item | Budget |
|------|--------|
| Cube total on disk | ≤ 1.5 GB (fits node RAM; SSH/SSS trivially cacheable in-process) |
| Cube build wall time (16 workers) | ≤ 30 min all products (expect ≪) |
| Feature cache export from cube | ≤ 2 min (G2) |
| Residual train run (frozen backbone, bs 512) | ≤ existing patch-l4 runtime (smaller model) |