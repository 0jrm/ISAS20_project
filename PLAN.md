# NeSPReSO v2 Dissertation Branch: AI-Agent Roadmap and Implementation Handoff

You are working inside the NeSPReSO v2 thesis codebase. Your job is not to redesign the science, chase new architectures, or optimize for flashy benchmarks. Your job is to make the current codebase scientifically defensible, reproducible, and ready for a Gulf of Mexico dissertation experiment centered on ARGO/profile targets, mask-native surface observations, readiness diagnostics, and carefully justified temporal evaluation.

The guiding principle is:

A result is only useful if the data split, target definition, surface-input realism, missingness handling, and evaluation protocol are scientifically defensible.

Do not begin transformer, GAN, diffusion, LSTM/GRU, or large architecture experiments. Do not rewrite the project. Work in small, reviewable commits.

---

# Primary Objective

Implement the foundation for a GoM-focused NeSPReSO dissertation branch with:

1. Data census and temporal split design based on actual available data.
2. Date-based chronological train/validation/test splitting.
3. ARGO-first training and evaluation path.
4. L3/masked-input surface observation pipeline.
5. Explicit support for missing inputs through masks, age, uncertainty, and observation-count channels.
6. L4 auxiliary/augmentation pathway that does not silently turn L4 into the truth source.
7. Readiness diagnostics for predicted temperature/salinity profiles.
8. Baseline evaluation reports suitable for thesis tables/figures.
9. Minimal hooks for later physics-aware loss and ensemble uncertainty.
10. Reproducible scripts, configs, metadata, and reports.

The primary target should be ARGO/CORA/profile observations when available. ISAS/L4 products may be used for comparison, pretraining, auxiliary context, or augmentation, but must not become the implicit truth target unless explicitly marked.

---

# Repository Context

Important areas likely include:

* `NeSPReSO2_onTemplate/data_loader/data_loaders.py`
* `NeSPReSO2_onTemplate/config_argo.json`
* `NeSPReSO2_onTemplate/config_argo_smoke.json`
* `NeSPReSO2_onTemplate/config_argo_pred_profile_cached.json`
* `NeSPReSO2_onTemplate/config_isas*.json`
* `NeSPReSO2_onTemplate/scripts/`
* `NeSPReSO2_onTemplate/notebooks/`
* `NeSPReSO2_onTemplate/notebooks/compare_outputs/`
* `NeSPReSO2_onTemplate/preproc/`
* `NeSPReSO2_onTemplate/model/`
* `NeSPReSO2_onTemplate/base/`
* `utils/`
* `PLAN.md`
* `PLAN-phase5.md`
* `AGENTS.md`
* `CLAUDE.md`
* `eng-principles-pack/`

Before editing, inspect the relevant data loader, preprocessing, config, training, and evaluation files.

---

# Non-Negotiable Guardrails

1. Do not silently use random train/validation/test splitting for dissertation results.
2. Do not silently compare ARGO and ISAS on mismatched samples/grids/times.
3. Do not silently interpolate sparse L3 observations into complete L4-like maps.
4. Do not treat missing values as zeros without corresponding mask channels.
5. Do not use L4 data as the hidden truth target unless the experiment explicitly says so.
6. Do not change scientific behavior without recording it in config and output metadata.
7. Do not hard-code paths if config support exists.
8. Do not celebrate surprisingly good metrics before checking for leakage.
9. Do not change architecture before the data and evaluation foundation is stable.
10. If required metadata is missing, stop and report exactly what is missing.

---

# Phase 0 — Data Census and Temporal Split Design

This phase comes before implementing the final split. The goal is to inspect the actual data and decide which temporal splits are scientifically valid.

Do not simply hard-code 2002–2015 / 2016–2017 / 2018–2020 without checking the data. Treat that split as an initial candidate, not a conclusion.

## Tasks

Inspect available datasets and metadata for:

* ARGO/profile target coverage.
* ISAS coverage, if used for comparison.
* Existing L4 satellite products.
* Available or planned L3 along-track SSH.
* Available or planned L3/L2P/L3U SST.
* ERA5 10 m wind history.
* Optional SSS products.
* Timestamp representation in every relevant cache/sample.
* Spatial coverage in the GoM domain.
* Depth coverage of target profiles.
* Availability of quality-control flags.
* Sample density by year, month, season, region, and depth.
* Surface observation density and missingness by year/month/season.
* High-observation and low-observation regimes.

## Required Analysis

Produce a data-census report with:

1. Number of target profiles by year.
2. Number of target profiles by month and season.
3. Number of target profiles by region/subregion if location metadata supports it.
4. Number of profiles by depth coverage.
5. Missingness statistics for each surface input.
6. L3 SSH coverage statistics:

   * number of observations per target patch,
   * nearest-track distance,
   * number of altimeter tracks in each time window,
   * time since nearest SSH observation.
7. SST coverage statistics:

   * cloud/missing fraction per target patch,
   * number of observed SST pixels,
   * time since nearest SST observation.
8. Wind-history availability:

   * completeness of u10/v10 history windows,
   * missing or corrupt periods.
9. Optional SSS availability:

   * start date,
   * coverage,
   * coastal quality issues if metadata supports this.
10. Candidate temporal splits with justification.

## Required Split Candidates

Evaluate at least the following candidate split families:

### Candidate A — Simple chronological split

Example prior:

* Train: 2002–2015
* Validation: 2016–2017
* Test: 2018–2020

Use this only if data coverage supports it.

### Candidate B — Data-balanced chronological split

Choose train/validation/test windows that preserve:

* all seasons,
* sufficient ARGO/profile density,
* adequate satellite coverage,
* comparable regional coverage,
* enough high- and low-observation conditions in validation and test.

### Candidate C — High-observation test split

Design a test period or subset where L3 coverage is unusually good.

Purpose:

* evaluate best-case observation geometry,
* check upper-bound performance,
* compare L3-native model against L4 baseline under favorable conditions.

### Candidate D — Low-observation / missingness stress-test split

Design a test period or subset where L3 coverage is sparse or SST missingness is high.

Purpose:

* evaluate robustness,
* test whether mask-native channels matter,
* expose failure modes relevant to operations.

### Candidate E — Recent-period split

If the L3/SST/SSS input stack only exists for later years, propose a split restricted to the common-overlap era.

Example:

* Train: earlier common-overlap years
* Validation: middle common-overlap years
* Test: most recent common-overlap years

This is especially important if VIIRS, SMAP, SWOT, or newer L3 products are used.

### Candidate F — Sensor-era split

If major observing-system changes occur, propose splits that avoid or deliberately test sensor-era shifts.

Examples:

* pre/post specific satellite missions,
* pre/post SMAP era,
* pre/post SWOT era,
* high-altimeter-constellation versus low-altimeter-constellation years.

## Split Decision Criteria

For each candidate split, report:

* train/validation/test sample counts,
* year/month/season counts,
* region counts,
* depth-coverage counts,
* target-profile density,
* SSH L3 coverage density,
* SST missingness/cloud fraction,
* wind-history completeness,
* nearest-track-distance distribution,
* high/low observation-regime representation,
* whether validation/test are independent in time,
* whether augmentation could leak information,
* whether L4 auxiliary products are temporally aligned without future leakage.

## Output

Create:

* `reports/data_census.md`
* `reports/data_census.json`
* `reports/split_design.md`
* `reports/split_design.json`

The split-design report must recommend:

1. Default dissertation split.
2. High-observation test subset.
3. Low-observation stress-test subset.
4. Optional common-overlap split for L3/SST/SSS experiments.
5. Any periods that should be excluded because of missing/corrupt data.

Acceptance criteria:

* The final split is justified by actual data coverage, not assumption.
* The chosen split is deterministic and reproducible.
* The report includes enough information to defend the split in a thesis methods section.
* If the data do not support the proposed split, the report recommends a better one.

---

# Phase 1 — Configurable Temporal Splitting

Implement configurable split logic after Phase 0.

## Requirements

Support a config field such as:

```json
{
  "split_mode": "chronological",
  "split_config": {
    "train": {"start": "2002-01-01", "end": "2015-12-31"},
    "val": {"start": "2016-01-01", "end": "2017-12-31"},
    "test": {"start": "2018-01-01", "end": "2020-12-31"}
  }
}
```

Also support named subsets if Phase 0 recommends them:

```json
{
  "evaluation_subsets": {
    "high_observation": "...",
    "low_observation": "...",
    "common_overlap": "..."
  }
}
```

Preserve old random split mode only for backwards compatibility or ablation, but chronological splitting must be the dissertation default.

## Acceptance Criteria

* No sample from validation or test periods appears in training.
* Split assignment is deterministic.
* Split metadata is saved with every processed dataset and result.
* A smoke test verifies correct year/date assignment.
* A report is generated with sample counts by split, year, month, season, and region if available.
* Configs clearly record the split mode and date ranges.

---

# Phase 2 — ARGO-First Target Path

Make ARGO/profile observations the primary dissertation target.

## Tasks

1. Confirm ARGO cached/profile dataset path works end-to-end.
2. Confirm ARGO configs use the selected chronological split.
3. Add or update a fast smoke config.
4. Add or update baseline training/evaluation scripts.
5. Confirm target profiles contain usable temperature, salinity, depth/pressure, location, and timestamp metadata.
6. Add explicit checks for profile quality, depth coverage, and missing target values.

## Matched ARGO-vs-ISAS Evaluation

If comparing ARGO and ISAS:

* compare only matched spatial/temporal samples where possible,
* report mismatch tolerance,
* record interpolation/resampling method,
* document unavoidable mismatch,
* never report raw ARGO and ISAS metrics as if they were directly comparable when the sample support differs.

## Acceptance Criteria

* Working ARGO smoke run.
* Working ARGO baseline evaluation command.
* Machine-readable ARGO evaluation output.
* Human-readable ARGO summary report.
* ARGO-vs-ISAS comparison does not silently compare mismatched samples.
* Any unavoidable mismatch is explicitly documented.

---

# Phase 3 — L3 / Masked-Input Data Pipeline

Implement an L3/masked-input data pipeline for NeSPReSO.

Do not redesign the model yet. First build the data layer, sample format, masks, uncertainty channels, local cache, and reports.

## Required Downloaders

Implement or document programmatic downloaders for:

1. Copernicus Marine L3 along-track SSH historical product.
2. Optional Copernicus Marine L3 NRT SSH.
3. GHRSST/VIIRS L3U SST via PO.DAAC or NOAA CoastWatch.
4. ERA5 hourly u10/v10 winds.
5. Optional SMAP L3 SSS.
6. Optional L4 SSH/SST/SSS auxiliary products for augmentation.

Downloader requirements:

* Store raw files unchanged.
* Record product name, dataset ID, version, download date, variables, date range, spatial bounds, and command/API call.
* Use idempotent downloads where possible.
* Skip existing valid files unless explicitly forced.
* Detect corrupt/incomplete files.
* Produce a download manifest.

Suggested manifest path:

* `data/manifests/download_manifest.jsonl`

## Local Cache Format

Create a clear separation between raw and processed data:

* `data/raw/...`
* `data/processed/...`
* `data/cache/...`
* `data/manifests/...`
* `reports/...`

Every processed sample must record:

* target dataset,
* source products,
* product versions,
* variables used,
* target timestamp,
* input time windows,
* spatial bounds,
* preprocessing code version or git commit,
* split assignment,
* quality-control flags,
* surface-observation coverage metrics.

## Rasterization

Convert sparse L3 observations into local patch tensors around each ARGO/profile target.

For each target profile at latitude, longitude, and timestamp:

1. Define a spatial patch.
2. Define historical time windows.
3. Load relevant L3 observations.
4. Rasterize sparse observations into the local patch grid.
5. Aggregate observations per cell/time bin.
6. Preserve missingness explicitly.

For every variable, produce channels such as:

* `value`
* `mask`
* `age`
* `uncertainty`
* `count`
* optional `source_id`
* optional `quality_flag`
* optional `synthetic_source_flag`

Never silently interpolate missing L3 cells into complete maps.

The raster grid is only a sparse-observation container. It is not a gridded objective analysis unless explicitly labeled as L4 auxiliary context.

## Aggregation Rule

When uncertainty exists, use inverse-variance weighting or another clearly documented method:

* weighted value,
* effective uncertainty,
* count,
* age,
* mask.

If uncertainty is unavailable, use count-weighted or nearest-observation aggregation, but record this clearly.

## Time Windows

Support configurable historical windows, for example:

* same day,
* previous 24 hours,
* previous 72 hours,
* previous 7 days,
* previous 14 days.

Do not assume the optimal window. Make it configurable.

## Acceptance Criteria

* L3 samples can be generated around ARGO/profile targets.
* Every value channel has a corresponding mask channel.
* Missingness is represented explicitly.
* Coverage metrics are saved per sample.
* A smoke test loads a batch with missing inputs.
* The batch has the expected shape and metadata.
* No missing value is confused with a physical zero.

---

# Phase 4 — L4 Auxiliary and Augmentation Pathway

L4 products may be used, but only with strict source labeling.

Acceptable uses:

* pretraining,
* teacher signal,
* auxiliary context,
* uncertainty-aware augmentation,
* mask simulation,
* comparison baseline.

Unacceptable use:

* silently treating L4 as the main truth source,
* silently filling L3 missingness with L4 and calling it L3,
* using future L4 fields to construct past inputs,
* reporting L4-trained performance as operational L3 performance.

## Required L4 Augmentation Features

Implement or prepare:

1. Real L3 mask libraries.
2. Real SST cloud-mask libraries.
3. Application of real L3/L3U masks to L4 fields.
4. Uncertainty-scaled perturbations.
5. Spatially correlated perturbations.
6. Sensor/mission dropout.
7. Time-window dropout.
8. Source-flag channels.

## Source-Flag Channels

Each surface value should be identifiable as one of:

* real L3 observation,
* synthetic L4 masked augmentation,
* L4 auxiliary context,
* missing,
* optionally climatology or fallback if such a fallback is ever used.

## Noise and Uncertainty

If L4 uncertainty variables are available, use them to scale perturbations. However, do not assume L4 formal mapping error is equivalent to independent pixelwise observation error.

Implement both:

* local independent perturbation,
* spatially correlated perturbation.

Record the perturbation settings in config and output metadata.

## Acceptance Criteria

* L4 augmentation can be enabled/disabled by config.
* Real L3 masks can be applied to L4 fields.
* Synthetic samples are clearly labeled.
* L4 auxiliary channels are distinguishable from real observations.
* Augmentation does not leak validation/test information into training.
* Augmentation settings are recorded in result metadata.

---

# Phase 5 — Model Input Compatibility

Do not redesign the model. Make the existing PatchConvMLP/PCA path compatible with expanded masked-input channels.

## Tasks

1. Extend input channel configuration.
2. Support variable-feature bundles:

   * value,
   * mask,
   * age,
   * uncertainty,
   * count.
3. Add config fields for enabled variables.
4. Add config fields for enabled feature channels.
5. Add smoke tests for batch loading and forward pass.
6. Ensure normalization does not corrupt masks, counts, ages, or uncertainty channels.

## Normalization Rules

* Value channels may be standardized.
* Mask channels must remain binary.
* Count channels may be transformed or normalized only if explicitly configured.
* Age channels should be scaled in physically interpretable units, such as hours or days.
* Uncertainty channels should preserve units or use documented normalization.
* Missing values should be filled only after masks are created.

## Acceptance Criteria

* Existing L4 baseline still runs.
* New L3/masked-input batch loads successfully.
* Forward pass works with expanded channels.
* Channel names/order are saved in metadata.
* Config determines which variables/features are used.
* Smoke tests catch channel-order mismatch.

---

# Phase 6 — Baseline Evaluation Reports

Evaluate the existing baseline and the new data pathways before changing training logic.

## Required Metrics

For temperature and salinity:

* overall RMSE,
* overall bias,
* RMSE by depth,
* bias by depth,
* optional MAE,
* optional correlation by depth.

## Required Stratified Metrics

Where metadata supports it, report metrics by:

* year,
* season,
* month,
* region/subregion,
* depth,
* nearest SSH track distance,
* L3 SSH coverage density,
* SST cloud fraction,
* wind-event intensity,
* high-observation subset,
* low-observation subset,
* common-overlap subset.

## Required Experiments

Implement evaluation support for:

1. Current L4 baseline.
2. L4 with synthetic L3 masks and uncertainty augmentation.
3. Real L3 SSH + wind history.
4. Real L3 SSH + L3/L3U SST + wind history, if SST data are available.
5. Real L3 + L4 auxiliary context.
6. No-wind ablation.
7. No-mask/age/uncertainty ablation.
8. L3 SSH-only versus SST-only, if both are available.

The first minimum viable comparison is:

* L4 baseline,
* real L3 SSH + wind,
* L4 masked augmentation,
* real L3 SSH + L4 auxiliary context.

## Output Requirements

Every result must include:

* dataset name,
* target source,
* input source products,
* split mode,
* split dates,
* subset name if applicable,
* model/config name,
* seed,
* git commit if available,
* timestamp,
* preprocessing version,
* metrics,
* coverage statistics.

Save:

* machine-readable JSON,
* human-readable Markdown or CSV,
* plots where existing conventions support them.

## Acceptance Criteria

* Evaluation can be rerun from a single command.
* Results are saved in a clear results directory.
* Reports are thesis-usable.
* Stratified metrics expose high/low data-regime behavior.

---

# Phase 7 — Readiness Diagnostics Module

Create a readiness diagnostics module, for example:

* `NeSPReSO2_onTemplate/diagnostics/readiness.py`

The module should run on saved predictions.

## RC-1: Static Stability Violations

Input:

* predicted temperature profiles,
* predicted salinity profiles,
* depth or pressure grid,
* latitude/longitude if needed,
* optional time metadata.

Output:

* profile-level stability flag,
* violation rate,
* violation count by depth,
* violation magnitude,
* depth distribution of failures.

Use available GSW-Torch functionality if it exists and is stable. If not, implement a clearly marked fallback or placeholder and document what remains.

## RC-2: Steric SSH Consistency

Input:

* predicted temperature/salinity profiles,
* SSH input or target where available,
* depth/pressure grid,
* location metadata.

Output:

* steric-height-derived consistency metric,
* correlation,
* bias,
* RMSE,
* residual summary.

Keep this modular because exact steric-height methodology may need scientific review.

## RC-4: Uncertainty Calibration Hook

Do not fully implement uncertainty unless ensemble outputs exist.

Define the expected interface:

* ensemble mean,
* ensemble spread/std,
* target,
* depth,
* optional region/season/subset metadata.

Add placeholder or partial implementations for:

* spread-error ratio,
* ENCE,
* reliability by depth,
* CRPS if feasible.

## Acceptance Criteria

* Diagnostics run on saved predictions.
* Diagnostics produce JSON output.
* Diagnostics produce at least one Markdown/CSV report.
* Required missing metadata triggers a helpful error.
* Optional missing metadata does not crash the full diagnostic suite.
* Diagnostics are callable from a script.

---

# Phase 8 — Physics-Loss Hook, Minimal Only

Add a clean interface for static-stability loss without forcing all training to use it.

## Requirements

* Configurable weight, e.g. `physics_loss_weight`.
* Configurable annealing schedule.
* Disabled by default unless explicitly enabled.
* Log data loss, physics loss, total loss, and current physics weight separately.
* Must not break existing non-physics training.

Suggested ablation configs:

* lambda = 0.0
* lambda = 0.01
* lambda = 0.1
* lambda = 1.0

## Acceptance Criteria

* Existing non-physics training still works.
* Physics loss can be enabled from config.
* Loss components are logged separately.
* A smoke test verifies forward/backward pass with physics loss enabled.
* Physics-loss behavior is documented.

---

# Phase 9 — Ensemble Support, Minimal Only

Prepare support for five independent seeds using the existing PatchConvMLP/PCA path.

## Tasks

1. Add launch script or config template for ensemble members.
2. Save predictions in a consistent format.
3. Add aggregation script for ensemble mean and spread.
4. Ensure calibration diagnostics can consume ensemble output.

## Acceptance Criteria

* Five model outputs can be discovered by one aggregation script.
* Ensemble mean and spread are saved.
* Aggregated output records member configs/seeds/checkpoints.
* Calibration diagnostics consume the ensemble format.

---

# Phase 10 — Documentation and Thesis-Ready Reporting

Update project documentation.

## Required Documentation

Add or update a project plan file with:

1. What changed.
2. Why random split is no longer dissertation-default.
3. How the final temporal split was selected.
4. What high/low data-regime subsets mean.
5. How to run the data census.
6. How to run ARGO smoke training.
7. How to run L3 sample generation.
8. How to run baseline evaluation.
9. How to run readiness diagnostics.
10. How to enable/disable L4 augmentation.
11. Known limitations.
12. Next steps.

## Suggested Files

* `PLAN-dissertation-data-foundation.md`
* `reports/data_census.md`
* `reports/split_design.md`
* `reports/baseline_eval.md`
* `reports/readiness_diagnostics.md`

---

# Engineering Quality Requirements

Make small, reviewable changes.

Prefer tests over assumptions.

Every new scientific behavior must be controlled by config.

Every generated result must record:

* dataset,
* target source,
* input sources,
* split mode,
* split dates,
* subset name,
* config path,
* seed,
* timestamp,
* git commit if available,
* preprocessing version.

Every processed sample must be traceable back to raw products.

If a result looks surprisingly good, first check:

* temporal leakage,
* target leakage,
* L4 future leakage,
* train/test overlap,
* duplicate profiles,
* normalization leakage,
* accidental use of test statistics,
* ARGO/ISAS mismatch.

If a result looks surprisingly bad, first check:

* target scale,
* salinity units,
* temperature units,
* depth alignment,
* mask handling,
* missing-value fill behavior,
* channel order,
* normalization,
* sample/target alignment,
* corrupt files.

---

# Suggested Commit Order

Use this order unless the data inspection reveals a blocker:

1. Data census scripts.
2. Split-design report.
3. Configurable chronological split.
4. Split smoke tests and split reports.
5. ARGO-first smoke path.
6. Baseline chronological evaluation.
7. L3 downloader scaffolding.
8. L3 rasterization/cache format.
9. Masked-input batch loading.
10. L4 augmentation scaffolding.
11. Model input channel compatibility.
12. Stratified evaluation reports.
13. Readiness diagnostics.
14. Physics-loss hook.
15. Ensemble aggregation support.

Do not skip directly to physics loss or ensembles before split and data-pipeline validity are established.

---

# Minimum Viable Dissertation Foundation

If time is limited, prioritize this subset:

1. Data census and split design.
2. Chronological split implementation.
3. ARGO-first target path.
4. L3 SSH + ERA5 wind masked-input pipeline.
5. L4 baseline comparison.
6. L4 masked augmentation.
7. Evaluation by coverage density and nearest-track distance.
8. Static-stability readiness diagnostic.
9. Baseline result report.

Only after that is stable should SST L3U, SMAP SSS, SWOT, physics loss, and ensembles be expanded.

---

# Final Deliverable From Agent

Return a concise implementation report containing:

1. Files changed.
2. New commands added.
3. Data inspected.
4. Recommended temporal split and why.
5. Alternative high/low observation-regime splits.
6. Tests or smoke checks run.
7. Results produced.
8. Remaining blockers.
9. Known scientific limitations.
10. Recommended next coding task.

The report must clearly distinguish:

* what was implemented,
* what was only scaffolded,
* what was analyzed but deferred,
* what remains scientifically uncertain.
