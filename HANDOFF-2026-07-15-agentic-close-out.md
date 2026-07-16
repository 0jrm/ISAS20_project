# Session handoff — close-out plan (Steps 1–3)

**Date:** 2026-07-15
**Plan:** [`PLAN-agentic-close-out.md`](PLAN-agentic-close-out.md)
**Status:** Steps 1, 2, 3 **done**. Steps 4, 5, 6 open.
**Prior:** [`HANDOFF-2026-07-15-agentic-track0.md`](HANDOFF-2026-07-15-agentic-track0.md),
[`HANDOFF-2026-07-15-agentic-track-a.md`](HANDOFF-2026-07-15-agentic-track-a.md)

Use conda env `nespreso`. **Environment gotcha:** this shell's `PATH` lacks `/usr/bin` and `/bin`
(so `git`, `head`, `tail` are missing). Fix with `export PATH="$PATH:/usr/bin:/bin"` — **append, do
not prepend**, or `/usr/bin/python` shadows the `nespreso` interpreter and every import fails.

---

## Step 1 — Track 0.2 loss-scale retrain: **CLOSED-NEGATIVE**

Full writeup in `HANDOFF-2026-07-15-agentic-track0.md` § "Track 0.2 — CLOSED-NEGATIVE". Summary:

| model | T_rmse_native | S_rmse_native | avg_common |
|---|---:|---:|---:|
| `golden_point` (target) | 0.5367 | 0.0897 | 0.3159 |
| `anom_point` (stale scales) | 0.6803 | 0.1043 | 0.3940 |
| **retune (corrected scales)** | **0.6545** | **0.1013** | **0.3797** |

−3.79% on T; **18% of the parity gap**. The pre-registered prediction ("will not reach parity")
held. Keep the corrected scales (analytically right, free); **stop blaming `loss_scales`**.
`val_loss` is not comparable across the two runs — only native RMSE is. n=1 per arm, so ~4% is not
separable from trajectory noise.

Free second read: σ₀ 7.38% → 5.14% (slightly *more* over-smoothed); RC-2 r 0.8299 → 0.8313 (still
saturated). Neither changes a decision.

---

## Step 2 — Track B retired in the docs: **DONE**

The retirement argument was already written (commit `385ea75`). This session cleared the **stale
status markers** that survived it — the header still said "Track A is next; Track B stays gated
behind it" and Track 0.4 still said "IN FLIGHT" — and folded Step 1's number into the `loss_scales`
non-goal: the *entire* T:S knob is worth −3.79% on T RMSE, so even a perfect search over it wins
~4%. That retires it as a search target on **measured** grounds, not just analytic ones.

---

## Step 3 — RC-4 MC-dropout ensemble: **DONE — prediction confirmed, plus a new lead**

`uncertainty_calibration_hook` returns real metrics instead of `not_implemented`.

### Pre-registered prediction: **CONFIRMED**

Spread-error ratio is **≪ 1** — the models are **over-confident**, exactly as predicted.

| | `golden_point` | `anom_point` |
|---|---:|---:|
| RMSE (ensemble mean, T) | 0.5360 | 0.6118 |
| RMS spread | 0.1992 | 0.1937 |
| **Spread-error ratio** | **0.3716** | **0.3165** |
| ratio / finite-N ideal (0.9901) | 0.3753 | 0.3197 |
| ENCE | 4.9456 | 3.5575 |
| rank corr, pooled | 0.7478 | 0.7797 |
| **rank corr, within depth (median)** | **0.1192** | **0.1573** |
| CRPS (MAE) | 0.2209 (0.2668) | 0.2614 (0.3142) |

The plan's expected caveat also holds: MC dropout only sees the epistemic spread **dropout**
induces, not the PC-shrinkage systematic bias, so spread ≪ error was the likely outcome. This is a
finding about the metric's reach as much as about the models.

### Three things worth more than the headline

1. **The pooled rank correlation is a trap — it is depth-confounded.** 0.75/0.78 pooled looks like
   the spread predicts error well. It does not: spread and |error| are *both* large near the surface
   and small at depth, so pooling scores high on depth structure alone. **Controlled within depth,
   the median collapses to 0.119 / 0.157.** The dropout spread carries almost no per-profile
   information — it re-encodes the climatological depth profile of variability. Both numbers are
   reported; the markdown labels the pooled one ⚠️ and names the within-depth one as the DA-relevant
   one.
2. **Over-confidence is worst exactly where the model is most confident.** The RMV/RMSE column rises
   monotonically across spread bins (`golden_point`: 0.077 → 0.573). In the lowest-spread decile the
   spread understates the error **13×**. For DA that is the dangerous direction: the field is most
   wrong about its own confidence precisely where it claims certainty.
3. **The spread barely knows which model is worse.** RMS spread is nearly identical for the two
   models (0.199 vs 0.194) while their RMSE differs by 14% (0.536 vs 0.612). The dropout spread
   tracks architecture and dropout rate, not data-driven uncertainty.

**Verdict for DA:** MC dropout as-is is **not** a usable uncertainty source — not because the scale
is wrong (a single ~2.7× recalibration would fix the ratio) but because the within-depth rank
correlation is ~0.12, so after rescaling it still would not know *which* profiles are bad. Fixing the
scale would produce a confidently-wrong covariance. **Do not feed this to DA as an error covariance.**

### ⚠️ NEW LEAD for Step 6 — `anom_point`'s deterministic inference is 10% worse than its own ensemble

Found while sanity-checking "ensemble_mean ≈ deterministic prediction". It is **not** ≈ for
`anom_point`:

| model | deterministic (eval mode) | ens_mean (N=50) | improvement | mean **single** member |
|---|---:|---:|---:|---:|
| `golden_point` | 0.5367 | 0.5360 | +0.14% | 0.5711 (worse than det — **normal**) |
| **`anom_point`** | **0.6803** | **0.6118** | **+10.06%** | **0.6411 (better than det — abnormal)** |

For `anom_point`, turning dropout **on** *improves* a single forward pass over eval-mode inference.
That is backwards, and `golden_point` shows nothing like it. **MC averaging alone closes 48% of the
anomaly parity gap** (0.1436 → 0.0751) — versus the loss-scale retrain's 18%. Even ensemble-to-
ensemble the gap shrinks (0.6118 vs 0.5360).

**Hypothesis to test in Step 6:** dropout's Jensen effect through the head's ReLUs inflates the
predicted PC amplitude, partially undoing the known high-order-PC shrinkage (0.196× true std). The
anomaly PCA spectrum is flatter, so proportionally more signal sits in the high-order PCs and
`anom_point` benefits far more. **Test:** pred/true PC-std ratio for deterministic vs ens_mean,
anom vs golden (reuse the Track 0 / Step 4 diagnostic).

### Implementation

| Path | Change |
|---|---|
| `notebooks/nb_metrics.py` | `enable_mc_dropout()`, `_mc_dropout_pcs()`, `run_inference(..., mc_samples=0)` → adds `mc_pcs` (N, n_samples, n_pc) |
| `diagnostics/readiness.py` | `ensemble_crps()`, real `uncertainty_calibration_hook()`, `readiness_report(..., uncertainty=)`, `readiness_from_checkpoint(..., mc_samples=, mc_variable=)`, RC-4 markdown, `--mc-samples` / `--mc-variable` CLI |
| `selfcheck.py` | 5 new tests (below), registered in the runner |

```bash
python diagnostics/readiness.py -c config/argo/config_argo.json \
  -r saved/models/NeSPReSO2_ARGO_GoM/scratch_0705_204716_golden_point/model_best.pth \
  --split test --mc-samples 50 \
  --out saved/readiness/readiness_golden_point_mc50.json \
  --md-out saved/readiness/readiness_golden_point_mc50.md
```

Artifacts: `saved/readiness/readiness_{golden_point,anom_point}_mc50.{json,md}`.

**Design notes worth keeping:**

- **Fails loudly instead of reporting zero spread.** `enable_mc_dropout` raises if the model has no
  `nn.Dropout` or if every `p == 0`, and `_mc_dropout_pcs` raises if all members come out identical.
  A silent zero spread would read as "perfectly confident" — the exact silent-wrong-number class
  Track 0 found in this file.
- **`model.eval()` is restored in a `finally`**, so a raising pass cannot leave dropout live and
  quietly randomize later evaluations.
- **A reliable N-member ensemble scores `sqrt(N/(N+1))`, not 1** (the ensemble mean carries σ²/N of
  sampling error on top of the σ² it represents). Reported as `finite_n_ideal_ratio` alongside
  `spread_error_ratio_corrected`, so 0.99 at N=50 is not misread as under-dispersion.
- **The target is `cache["profiles"]`** (physical, depth-major) via `profiles_depth_major` — *not*
  `cache["true_profiles"]` (anomalies, sample-major). Verified on the anom cache: `profiles` mean
  8.60 °C, `true_profiles` mean 0.019. Every member is reconstructed with
  `reconstruct_physical_profiles(clim_profiles=, indices=)`; the climatology cancels in the spread
  but **not** in the error. See memory `readiness-anomaly-cache-gotchas`.

### Verification

- **Default path byte-identical** (the plan's gate): `golden_point` reproduces
  `T=0.5366791090646792`, `S=0.08974318457647712` — **exact float equality**, not just close.
- **RC-1/RC-2 unchanged** under `--mc-samples 50`: `golden_point` violation_rate 0.0 → 0.0;
  `anom_point` 0.0738362760834671 → identical, RC-2 r 0.8299386336840576 → identical.
- **`ensemble_mean ≈ deterministic`** holds for `golden_point` (mean |Δ| 0.015 vs RMSE 0.54) — and
  its *failure* for `anom_point` is the lead above, not a bug (the two paths share all machinery and
  differ only in the PCs).
- `pytest tests/test_sampler.py tests/test_cube_validate.py tests/test_operators.py -q` → **20
  passed**.
- 5 new selfchecks pass: `test_ensemble_crps_matches_pairwise_definition`,
  `test_uncertainty_calibration_detects_calibrated_ensemble`,
  `test_uncertainty_calibration_detects_underdispersion`,
  `test_uncertainty_hook_not_implemented_without_ensemble`,
  `test_mc_dropout_enables_only_dropout`.

**The calibrated-ensemble test is a deliberate positive control.** The plan said: *"If it reports
≈1, suspect the calibration code before believing the models are calibrated."* That warning is only
actionable if the metric is known to be able to report ≈1. The test builds an ensemble that is
calibrated **by construction** and asserts ratio ≈ `sqrt(N/(N+1))` and ENCE < 0.05 — so the real
0.37 is a statement about the models, not about the code.

**ENCE has a sampling-noise floor — do not read it against 0.** ENCE bins by predicted spread, so it
only has signal when the spread genuinely varies. On *homoscedastic* data the bins sort on sampling
noise in the sample std (relative error ~`1/sqrt(2(N-1))` ≈ 9% at N=60) and ENCE reports ≈0.08 no
matter how well calibrated the ensemble is — measured, not theorized: the first draft of the
positive control was homoscedastic and scored 0.077. The test is now heteroscedastic on purpose.
The observed 4.95 / 3.56 are far above any such floor, so the conclusion is safe.

---

## Step 4 — `point_cube`'s 38.52% σ₀: **cause stated. Phase 8 should NOT proceed.**

### 🚨 First, a correction that outranks the step itself: **"nature's 24.70%" is not nature**

The Track 0 scoreboard row `TRUTH (raw cache) — 4145 — 24.70% / 0.0586%` is the **PCA-16
reconstruction** of truth (the regression *target*), not the raw ARGO profiles. Reproduced to the
digit: PCA-16 truth, tol=0.01, n=4145 → **24.73% / 0.0586%**.

| row (tol=0.01, `saved/readiness/rc1_reference_rows_corrected.json`) | n=4145 | test n=623 |
|---|---:|---:|
| **RAW TRUTH (nature)** | **3.88%** | **1.12%** |
| **PCA-16 TRUTH (the regression target)** | 24.73% | 21.83% |

**The PCA-16 truncation is the dominant source of σ₀ inversions** — it turns a 1.12%-unstable ocean
into a 21.83%-unstable target, a **6.4× inflation**. What changes:

- **"Models are over-smoothed" survives; its magnitude does not.** `golden_point`'s 0.00% is smoother
  than nature's **1.12%**, not 24.70% — real, but ~20× less dramatic than recorded.
- **A model that perfectly hit its training target would violate at ~21.83%.** Every model except
  `point_cube` sits below its own target's rate: they smooth away a basis artifact.
- **`point_cube` is still the outlier, by more than recorded** — 38.52% is **34× nature's 1.12%** and
  the only model exceeding even the 21.83% target.
- **Track 0's "not a PCA artifact" control was vacuous.** It concluded raw truth (24.7%) ≈ PCA-16
  truth (24.7%) — but *both sides were the PCA-16 reconstruction*. It compared a number to itself.
  The real comparison (3.88% vs 24.73%) says the opposite: **this instability largely IS a basis
  artifact.**

This is the same silent-wrong-number class Track 0 existed to catch, found by re-deriving the
reference rather than citing it.

### The plan's hypothesis for `point_cube` is **FALSIFIED**

The plan predicted `point_cube` has a *higher* high-PC ratio ("injects amplitude into the fine modes
without the accuracy to justify it"). It does not (`saved/readiness/pc_shrinkage_by_model.json`):

| model | σ₀ viol | PC1 | first-4 | **last-8** |
|---|---:|---:|---:|---:|
| `golden_point` | 0.00% | 0.973 | 0.690 | **0.196** |
| `residual_cube` | 2.57% | 0.930 | 0.782 | **0.249** |
| `anom_point` | 7.38% | 0.776 | 0.631 | **0.341** |
| `point_cube` | **38.52%** | 0.934 | 0.726 | **0.220** |

`point_cube`'s high-PC amplitude (0.220) is barely above `golden_point`'s (0.196), while
`residual_cube` carries **more** (0.249) with 15× fewer violations and `anom_point` carries the
**most** (0.341) with 7.4%. The ordering is uncorrelated with σ₀. **High-PC noise injection is ruled
out**, and with it the "broadband input noise" reading in the plan.

### Actual cause: **the near-surface halocline is missing**

**100% of `point_cube`'s 750 violations sit at 2.5–11.5 m** — only 10 of 1800 interfaces, all at the
surface. Not broadband at all. And **every violation, in every model, is marginal**: all fall in the
0.010–0.02 kg/m³ band, none above 0.02. `point_cube` → 2.09% at tol=0.02 and **0.00% at tol=0.03**.

The mechanism, in the top 15 m (test split):

| | dS(0→15 m) | mean d(σ₀)/dz @3.5 m | % profiles violating @3.5 m |
|---|---:|---:|---:|
| RAW TRUTH | +0.1425 | +0.0128 | 0.00% |
| PCA-16 target | +0.1426 | +0.0147 | 7.70% |
| `golden_point` | +0.1476 | +0.0076 | 0.00% |
| `residual_cube` | +0.1162 | +0.0081 | 2.41% |
| **`point_cube`** | **+0.0156** | **−0.0042** | **37.88%** |

`point_cube` reproduces the near-surface salinity increase **10× too weakly** (+0.016 vs nature's
+0.14). In the GoM that halocline — fresher surface over saltier water, from river plumes and rain —
is what stabilizes the surface layer. Without it `point_cube`'s mean density *decreases* with depth
at 3.5–5.5 m, and the violations peak exactly there (37.9% @3.5 m, 25.0% @4.5 m).

**Contributing input-side cause: the cube's SSS is a degraded proxy.**

| cube vs point-cache input | corr | note |
|---|---:|---|
| SST | 0.9885 | transfers fine |
| SSH | 0.9651 | but a **+0.38 m** mean offset (ADT-vs-SLA convention?) — absorbable by bias |
| **SSS** | **0.7438** | RMS diff **0.48 PSU** vs a natural std of 0.66; **−13.2%** variance |

Halocline information carried by each SSS: `corr(dS, sss)` = **−0.616** (original) vs **−0.549**
(cube). So the cube's SSS is measurably worse at locating the halocline — and it is `point_cube`'s
*only* surface-salinity input. `residual_cube` escapes (dS +0.1162, 2.57%) precisely because it
anchors on the **point block**, which carries the original SSS — which is exactly the contrast the
plan asked for, and it lands.

**Honest limit:** the SSS degradation (−0.616 → −0.549) is real but *modest*, while the dS/dz
collapse is 10×. So the degraded SSS is a **contributing** cause, not a proven sole root cause.
A second suspect is on the table and not yet chased: **`point_cube`'s best epoch is 27** (vs
`golden_point` 313, `anom_point` 3067) — it stopped improving almost immediately, so it may simply
be undertrained/badly conditioned. Its inputs are also z-scored (`train_zscore_point_sat_only_v1`,
3 columns) while `golden_point` feeds **raw, unnormalized** SSS/SST/SSH — an uncontrolled difference
between the two.

### Exit: **`PLAN.md` Phase 8 should NOT proceed**

1. The target is **marginal**: every violation is within 0.010–0.02 kg/m³ of the threshold and all
   vanish by tol=0.03. This is not a physics failure; it is a near-threshold artifact.
2. The dominant source of σ₀ instability is the **PCA-16 basis** (1.12% → 21.83%), which no loss
   term on the model output can fix — the target itself is unstable.
3. `point_cube`'s excess is a **feature-pipeline defect** (surface salinity information), not a
   physics-knowledge defect. A σ₀ penalty would force it to fabricate a halocline it has no input
   to place — buying stability by degrading T/S, i.e. treating the symptom, exactly as the plan
   feared.

**Recommended instead:** fix the cube's SSS feature (why does it correlate only 0.744 with the same
quantity in the point cache?), and investigate `point_cube`'s best_epoch=27. Both are cheaper than a
physics loss and address causes rather than symptoms.

---

## Step 5 — `export_field_product.py`: **runs end-to-end. Output is NOT yet shippable — and that is the finding.**

The plan predicted "expect more bugs downstream". Correct — and they were not only in this file:
**the field *training* path had never run either.** Five bugs, in the order they surfaced:

| # | file:line | bug | effect |
|---|---|---|---|
| 1 | `export_field_product.py:33` | `pickle.load(cache)` inside `with open(...) as f` | `NameError` |
| 2 | `export_field_product.py:83` | `float(ord(ds))` on `"2020-01-01"` | `TypeError` |
| 3 | `export_field_product.py:94` | `temp_arr[..., land]` — a `(lat,lon)` mask applied to the **depth** axis | `IndexError` |
| 4 | `data_loader/data_loaders.py` `FieldDataLoader.__init__` | never sets `l3_enabled` / `l3_channel_metadata` / `sat_patch_shape`, which `train.py:294-296` reads off any loader | `AttributeError` — **field training impossible** |
| 5 | `data_loader/data_loaders.py` `FieldDataLoader.split_validation/split_test` | return a bare `DataLoader` without `pca_models`/`outputs`/`cache`, which `metric.profile_rmse` reads | `AttributeError` — **field validation impossible** |

4 and 5 are fixed by mirroring `NeSPReSODataLoader`'s attribute surface — `FieldDataLoader` already
mirrors it deliberately (`pca_models`, `outputs`, `input_params`, `dataset_tag`, …); it just missed
these. `train.py` treats the attributes as an implicit loader interface.

**Bug 2's fix — a real date→JULD, not an invented one.** Added `base/split_utils.py::dates_to_juld`,
the exact inverse of the existing `sample_dates` (ISAS: days since 1950-01-01; ARGO: MATLAB datenum
`toordinal() + 366`), covered by `selfcheck.py::test_dates_to_juld_round_trip`. **The JULD must use
`clim.meta["dataset_tag"]`** (the *source* tag, `argo_v2`) — not the field cache's own `argo_field` —
because `eval_climatology → design_matrix` decodes it with the climatology's tag. Getting that wrong
does not crash; it silently shifts the seasonal cycle. (The old `ord(ds)` returned **50.0** for every
date in the 2000s — a constant. It would have "worked" forever.)

### Verified end-to-end (smoke: `config_argo_field_smoke.json`, 8 dates, 2 epochs)

```bash
python train.py -c config/argo/config_argo_field_smoke.json -id smoke_field_export3
python scripts/export_field_product.py -c config/argo/config_argo_field_smoke.json \
  -r saved/smoke_argo_field/models/NeSPReSO2_ARGO_GoM_field_smoke/smoke_field_export3/checkpoint.pth \
  -o <out>.nc
```

**Structure: passes.** Opens in xarray with dims **`time/lat/lon/depth`** = (8, 52, 68, 1801);
coords lat 18–31, lon −98..−81, depth 0–1800 m; land NaN-masked (26%); mean T decreases with depth
**23.48 °C → 4.01 °C**. `time` is now `datetime64[D]` rather than strings.

**Values: FAIL, and I am not calling this done.** T is fine (2.71–28.06 °C). **Salinity spans
22.13–45.02 PSU — 23% of finite points outside 30–38.** Traced to two causes:

1. **The climatology itself is unphysical outside the ARGO hull — the dominant cause, and it is
   *not* the model.** Evaluated alone, with no model involved, the climatology reaches **43.80 PSU**
   at **ocean** points (not land, so the mask does not save it). The worst point is
   **lat 31.00, lon −81.00** — the grid's NE corner, which is off the **Atlantic** coast of Georgia,
   *not in the Gulf at all*. The climatology is a ridge fit on a lat/lon/doy tensor basis over the
   ARGO sampling hull (**lat 19.44–28.70, lon −95.33..−84.74**), but the export grid is
   **lat 18–31, lon −98..−81** — substantially larger. Outside the hull the basis extrapolates.
   **449 of 2616 ocean points (17%) exceed 37 PSU somewhere in the column.**
2. **The 2-epoch smoke model** contributes a ~−2 PSU mean bias. Expected; it is a smoke model.

**Added a plausibility tripwire** (`report_plausibility`), in the Track A spirit — the script now
prints T/S ranges and refuses to call the output shippable. *A netCDF that opens cleanly is not a
netCDF that contains an ocean:* this script wrote `ord("2020-01-01")` as its date for its entire
life and nothing noticed, because nothing ever looked at the numbers.

**Remaining gap (do not skip):** the plan's criterion "values physically plausible" is **not met**,
and cannot be met by fixing this script. Two follow-ups: **(a)** clip the export grid to the ARGO
sampling hull (or refit/regularize the climatology) — the product currently reports confident
salinity in the Atlantic from a Gulf climatology; **(b)** re-verify with a *properly trained* field
model (`config_argo_field.json`), not the 2-epoch smoke, to separate remaining model bias from the
climatology defect.

---

## Files changed this session

| Path | Change |
|---|---|
| `notebooks/nb_metrics.py` | `enable_mc_dropout`, `_mc_dropout_pcs`, `run_inference(..., mc_samples=0)` |
| `diagnostics/readiness.py` | `ensemble_crps`, real `uncertainty_calibration_hook`, RC-4 wiring + markdown, `--mc-samples`/`--mc-variable` |
| `base/split_utils.py` | **new** `dates_to_juld` (exact inverse of `sample_dates`) |
| `scripts/export_field_product.py` | 3 bug fixes + `report_plausibility` tripwire + `datetime64` time coord |
| `data_loader/data_loaders.py` | `FieldDataLoader`: `l3_enabled`/`l3_channel_metadata`/`sat_patch_shape`; `split_validation`/`split_test` now carry `pca_models`/`outputs`/`cache` |
| `selfcheck.py` | 6 new tests, registered in the runner |
| `PLAN.md`, `HANDOFF.md`, `PLAN-agentic-*.md` | corrections (24.70%; Phase 8 closed; Track B stale markers) |

New artifacts under `saved/readiness/`: `retune_0715_anom_point_profile_metrics.json`,
`readiness_retune_0715_anom_point.{json,md}`, `readiness_{golden_point,anom_point}_mc50.{json,md}`,
`pc_shrinkage_by_model.json`, `rc1_reference_rows_corrected.json`,
`step6_parity_dropout_inflation.json`.

**Nothing committed** — repo convention is commit only when asked.

### Verification summary

- `pytest tests/test_sampler.py tests/test_cube_validate.py tests/test_operators.py -q` → **20 passed**
- `python scripts/bench_datacube_speed.py --cascade --json` → **verdict: pass** (goldens pass, all configs)
- 6 new selfchecks pass (`dates_to_juld`, CRPS, calibrated/under-dispersed controls, N=0 contract, MC-dropout)
- **Default eval path byte-identical**, re-verified *after* the `data_loaders.py` edit:
  `golden_point` T = `0.5366791090646792`, S = `0.08974318457647712` — **exact float equality**
- RC-1/RC-2 bit-identical under `--mc-samples 50`
- `export_field_product.py` verified end-to-end on the smoke field model (structure ✅, values ❌ — see Step 5)

---

## Pre-existing failures found (NOT caused by this session's changes)

The full `selfcheck.py` (2h03m) reports **3 failed, 57 passed**. All three fail **identically on
clean `HEAD`** with this session's changes stashed — verified, not assumed:

| test | failure |
|---|---|
| `test_combined_pca_loss_v2` | `assert False` — golden `combined_loss` mismatch (`selfcheck.py:326`) |
| `test_sync_arch_l3_config` | `UnboundLocalError: count_encoding_dims` (`preproc/l3_input.py:144`) |
| `test_field_loss_grad_with_steric` | `RuntimeError: size of tensor a (3) must match b (4)` (`model/loss.py:275`) |

The regression gate the plan specifies (`tests/test_sampler.py test_cube_validate.py
test_operators.py`) is **20/20 green**, and `bench_datacube_speed.py --cascade` → **verdict: pass**
(goldens pass on all configs).

---

## Step 6 — the anomaly parity gap: **LOCALIZED. Much of it is an inference artifact.**

Two probes localized it, so this did not need the timebox.

### Probe 1 — PC shrinkage: **this is the gap**

`anom_point` shrinks the **dominant** mode: **PC1 at 0.776× true std vs `golden_point`'s 0.973×**.
The anomaly PCA spectrum is flatter (its PC1 is the *deviation*, a genuinely lower-SNR target than
the raw PC1, which is dominated by the highly-predictable seasonal/geographic signal), so an
MSE-optimal conditional mean regresses further toward the mean. High-order PCs are **not** the
story — `anom_point` actually carries *more* high-PC amplitude than `golden_point` (0.341 vs 0.196).

### Probe 2 — the discovery: **eval-mode inference under-predicts anomaly PC amplitude by ~16–18%**

MC-dropout averaging *recovers* PC1 amplitude, but **only for anomaly models**
(`saved/readiness/step6_parity_dropout_inflation.json`, N=50, chrono test):

| model | best epoch | anomaly | PC1 det | PC1 ens | **PC1 inflation** | T det | T ens | **RMSE gain** |
|---|---:|:--:|---:|---:|---:|---:|---:|---:|
| `point_cube` | 27 | no | 0.934 | 0.934 | +0.1% | 0.5770 | 0.5770 | −0.01% |
| `residual_cube` | 65 | no | 0.930 | 0.929 | −0.0% | 0.5975 | 0.5977 | −0.04% |
| `golden_point` | 313 | no | 0.973 | 0.975 | +0.2% | 0.5367 | 0.5360 | +0.14% |
| `anom_patch_l4` | 1048 | **yes** | 0.663 | 0.776 | **+16.9%** | 0.8821 | 0.8126 | **+7.87%** |
| `anom_point` | 3067 | **yes** | 0.776 | 0.918 | **+18.3%** | 0.6803 | 0.6118 | **+10.06%** |
| `retune_anom` | 3382 | **yes** | 0.791 | 0.915 | **+15.7%** | 0.6545 | 0.5900 | **+9.85%** |

**The confound is resolved: it tracks anomaly-ness, not training length.** `anom_point` was both an
anomaly model *and* trained ~10× longer than `golden_point`, so training length was the obvious rival
explanation. It fails: `anom_patch_l4` at **1048** epochs already shows **+16.9%**, and within the
anomaly group the effect is **flat across a 3.2× epoch range** (1048 → 3382 actually *decreases*,
16.9 → 15.7). If epochs drove it, 3382 would far exceed 1048. Meanwhile all three raw models sit at
~0% across 27–313 epochs. The split lands exactly on the anomaly/raw boundary.
*Honest limit:* the two groups do not **overlap** in epochs (raw ≤313, anom ≥1048), so the confound
is broken by the flat within-group trend rather than by an interleaved design. The clean test —
train `golden_point` to ~3000 epochs — was not run (~100 min).

**Mechanism (hypothesis, not proven):** dropout's Jensen gap through the head's ReLUs. `E[relu(z)] ≥
relu(E[z])`, so the MC mean carries systematically larger hidden activations than the eval-mode
weight-scaled network. On the low-SNR anomaly task the units sit nearer the ReLU kink (where the gap
is largest) and the network is more heavily regularized, so eval-mode shrinks the output; averaging
undoes part of that. On the high-SNR raw task the approximation is nearly exact. **What is measured,
independent of the mechanism: eval-mode weight scaling is a materially wrong approximation for these
anomaly models.**

### Why this matters: the parity gap is ~62% smaller than reported

Comparing like with like (both MC-averaged): **`retune_anom` 0.5900 vs `golden_point` 0.5360.**

| | T RMSE | gap to `golden_point` |
|---|---:|---:|
| `anom_point` as reported (det, stale scales) | 0.6803 | **0.1436** |
| + corrected loss scales (Step 1) | 0.6545 | 0.1178 (−18%) |
| **+ MC-averaged inference** | **0.5900** | **0.0540 (−62%)** |

**The reported anomaly parity gap is substantially an artifact of how inference is done**, not of the
anomaly reframing. Steps 1 and 6 together account for 62% of it. The anomaly models have been
penalized by an evaluation choice that costs the raw models nothing.

**Recommendations (not run — decisions for the user):**
1. **Evaluate anomaly models with MC-dropout averaging**, or state the eval-mode number as a known
   ~10% pessimistic bound. The whole scoreboard's anomaly rows are affected.
2. **Cheaper than 50× inference: lower `dropout_prob` for the anomaly configs and retrain.** p=0.2 in
   the head appears too strong for a low-SNR target — that is what makes the weight-scaling
   approximation fail. This is a config-level knob, and unlike `loss_scales` it demonstrably moves
   the metric by ~10%.
3. `anom_patch_l4`'s 0.8821 here vs the scoreboard's 0.9325 is a **masking** difference (the L4 cache
   carries `bottom_depth`; this probe used all finite points). The det-vs-ens contrast is internally
   consistent, so the +7.87% stands, but do not cross-quote the absolute value.

---

## Next / open threads

- **Follow-ups opened by Step 4** (not in the plan): the **cube SSS degradation** (corr 0.744 vs the
  point cache — the Phase 8 root cause); `point_cube` **best_epoch=27**; the cube-vs-point SSH
  **+0.38 m** offset; and a cube-vs-point **lat-encoding difference** (`golden` latcos ∈ 0.54–0.78 vs
  `cube` latcos ∈ 0.88–0.94 — the cube uses cos/sin of latitude in radians, `golden` does not; both
  are invertible over the GoM so the network can learn either, but it is an uncontrolled difference
  between two models that are supposed to differ only in feature *source*).
- **Follow-ups opened by Step 5:** clip the field export grid to the ARGO hull (or refit the
  climatology); re-verify the product with a trained field model.
- **Follow-up opened by Step 6:** train `golden_point` long to fully break the anomaly/epochs
  confound; decide on dropout_prob for anomaly configs.
- **Pre-existing test failures** (3) documented above — untouched, not this session's.

Nothing committed — repo convention is commit only when asked.
