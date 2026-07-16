# PLAN — close Track 0.2, retire Track B, execute Track C + the two scoped defects

**Created:** 2026-07-15
**Status:** **ALL 6 STEPS DONE (2026-07-15).** Session record:
[`HANDOFF-2026-07-15-agentic-close-out.md`](HANDOFF-2026-07-15-agentic-close-out.md).
Two findings outrank the plan's own questions: **(a)** "nature's 24.70%" was the **PCA-16 target**,
not nature (Step 4) — the basis, not the model, injects σ₀ inversions, and `PLAN.md` Phase 8 is
**closed**; **(b)** eval-mode inference under-predicts **anomaly** PC1 by ~16–18% (Step 6), so the
anomaly parity gap is **~62% smaller** than reported. One criterion is **not met** and is flagged,
not glossed: Step 5's field product is structurally correct but **not physically plausible** (the
climatology extrapolates outside the ARGO hull).
- **Step 1 (closed-negative):** retune T RMSE **0.6803 → 0.6545** (−3.79%), only 18% of the parity
  gap → loss scales were **not** the cause; corrected scales kept; gap handed to Step 6.
- **Step 2:** done (stale markers cleared; the retirement argument was already committed).
- **Step 3:** RC-4 live. Prediction **confirmed** — spread-error ratio **0.37 / 0.32** (over-confident).
  Pooled spread↔error rank corr is **depth-confounded** (0.75 → **0.12** within depth) ⇒ **not
  DA-usable**. Lead: `anom_point`'s eval-mode inference is **10% worse than its own dropout ensemble**.
- **Step 4:** 🚨 **"nature's 24.70%" was mislabelled** — it is the **PCA-16 target**; nature is
  **1.12%** (test) / 3.88% (all). The **basis**, not the model, injects σ₀ inversions (6.4×).
  `point_cube`'s 38.52% = a **cube-SSS / missing-halocline** defect, all violations marginal and at
  2.5–11.5 m. **`PLAN.md` Phase 8 is CLOSED — do not run it.**
**Supersedes the open half of:** [`PLAN-agentic-ai-experiment.md`](PLAN-agentic-ai-experiment.md)
(Tracks 0 and A there are **done**; Track B is **retired** — see Step 2)
**Results feeding this plan:** [`HANDOFF-2026-07-15-agentic-track0.md`](HANDOFF-2026-07-15-agentic-track0.md),
[`HANDOFF-2026-07-15-agentic-track-a.md`](HANDOFF-2026-07-15-agentic-track-a.md)

Use conda env `nespreso` to run any python code.

---

## Context

`PLAN-agentic-ai-experiment.md` had four tracks. **Track 0** (measure) and **Track A** (harden the
evaluator) are done — see `HANDOFF-2026-07-15-agentic-track0.md` and
`HANDOFF-2026-07-15-agentic-track-a.md`. They produced the plan's intended deliverables *and*
falsified several of its premises:

- σ₀ static stability is **slack** (models over-smoothed: 0.00% violations vs ~~nature's 24.70%~~) —
  except `point_cube` at **38.52%**, the only model less stable than the ocean.
  **⚠️ CORRECTED 2026-07-15 (Step 4): "nature's 24.70%" is the PCA-16 *target*, not nature. Nature is
  1.12% (test) / 3.88% (all). The conclusion survives; the magnitude was ~20× overstated.**
- steric-vs-SLA is **saturated** (model r=0.8299 vs true-profile ceiling r=0.8297) — unusable as a
  search objective.
- **Both** evaluators the plan counted on as existing assets were broken: `readiness.py` had never
  run and shipped two silent-wrong-number bugs; the golden gate asserted a **3 °C Gulf of Mexico**
  and failed against HEAD's own cube for 10 days.
- The one crisp measurable defect is **under-dispersion** (high-order PCs at 0.196× true std).

**Decisions taken (2026-07-15):** Track B (the OpenEvolve vs hand-optimization experiment) is
**skipped** — Track A measured its economics and they don't hold (below). This plan covers closing
the last open Track 0 question, retiring Track B honestly in the docs, and executing Track C plus
the two defects Track 0/A surfaced.

**Outcome:** every open question from Tracks 0/A either answered or explicitly escalated; DA
prerequisites unblocked; then stop (OSSE design remains a separate, dissertation-scale plan).

---

## Step 1 — Close Track 0.2: evaluate the finished retrain — **DONE (closed-negative)**

> **Result (2026-07-15):** `T_rmse_native` **0.6545** / `S_rmse_native` **0.1013** / avg_common
> **0.3797** (chrono test, n=623; `eval_run.py` cross-check passed). That is −3.79% on T vs the
> 0.6803 baseline — the **first branch of the decision rule below** ("≈0.68 → loss scales were not
> the cause"). Corrected scales kept; question closed-negative; parity gap handed to Step 6.
> Free second read: σ₀ 7.38% → 5.14%, RC-2 r 0.8299 → 0.8313 (both decision-neutral).
> Artifacts: `saved/readiness/retune_0715_anom_point_profile_metrics.json`,
> `saved/readiness/readiness_retune_0715_anom_point.{json,md}`. Written up in
> `HANDOFF-2026-07-15-agentic-track0.md` and memory `anom-phase-a-results`.
> *Original step text preserved below.*

The retrain **completed**: early stop at epoch 3382 (`retune_0715_anom_point`, tmux `anom_retune`,
EXIT=0). Seed 42 + chronological split were pinned, so `loss_scales` is the only difference from
`scratch_0705_204716_anom_point`.

**⚠️ The obvious comparison is invalid.** `val_loss` is **not comparable across the two runs** —
correcting `profile_scales` changed the loss normalization itself (a 0.733× global factor). Baseline
`mnt_best` 0.16285 vs retune `mnt_best` 0.22589 is apples-to-oranges. Use only scale-independent
metrics: **`raw_profile_rmse_native`** on the test split.

**Do:** reuse `notebooks/nb_metrics.py::profile_metrics_from_inference(config, ckpt, split="test")`
— the exact call that produced `notebooks/scratch_outputs/scratch_all_models_results.json`. Compare
against that file's baselines (chronological test, n=623):

| model | T_rmse_native | S_rmse_native | avg_common |
|---|---:|---:|---:|
| `golden_point` (target) | 0.5367 | 0.0897 | 0.3159 |
| **`anom_point` (beat this)** | **0.6803** | **0.1043** | **0.3940** |

Also re-run `diagnostics/readiness.py` on the retune checkpoint — RC-1/RC-2 are cheap now and give a
free second read on whether the rebalance changed the smoothing at all.

**Pre-registered prediction (from the Track 0 handoff): this will not reach parity.** Early evidence
agrees — retune best `val_profile_rmse` 0.2918, and early stop came *sooner* (3382 vs 3568).

**Decision rule:**
- If `T_rmse_native` ≈ 0.68 → **loss scales were not the cause.** Keep the corrected scales anyway
  (they are analytically right — they normalize each branch to 1.0 at zero-pred), record the
  question as closed-negative, and hand the parity gap to Step 5.
- If it materially improves toward 0.5367 → surprising; re-examine the 9.7% T:S argument.
- If it is *worse*: that is informative too — report it, do not bury it.

Write results to `saved/readiness/` and update `HANDOFF-2026-07-15-agentic-track0.md` + the memory
`anom-phase-a-results`.

---

## Step 2 — Retire Track B in the docs — **DONE**

> **Done (2026-07-15).** `PLAN-agentic-ai-experiment.md` now carries "Track B — **RETIRED, not run**"
> with all four arguments below and their measured numbers, the original design preserved under
> "Original design, preserved below for revival", and the hardened evaluator + goldens named as the
> durable deliverable. This session additionally cleared the stale status markers that survived the
> retirement commit (header said "Track A is next", Track 0.4 said "IN FLIGHT") and folded Step 1's
> result into the `loss_scales` non-goal: the *entire* T:S knob is worth −3.79% on T RMSE, so even a
> perfect search over it wins ~4% — retiring it as a search target on measured grounds.
> *Original step text preserved below.*

Skipping is a **result**, not an omission, and the plan explicitly wanted the failure modes. Record
in `PLAN-agentic-ai-experiment.md` why the experiment was not run, with the measured numbers:

- **The cascade is upside-down.** Its cheap stage-1 filter (ppd50, ~4 s) is its *noisiest*
  (σ 1.8–17.3%; 7.4% over 10 repeats), while the expensive `v1` (~173 s) is quietest (σ 1.3%). Short
  benchmarks are dominated by fixed overhead and jitter. So stage 1 cannot discriminate speed — it
  is a *correctness* filter that happens to be cheap.
- **Real cost:** speed discrimination only exists at `v1` → ~15 min/candidate at min-of-5, against
  the plan's "~500 candidates ≈ $25 … the loop is evaluator-bound, not token-bound." The loop is
  evaluator-bound *far harder than assumed*; the plan's own reasoning, run on real numbers, argues
  against running it.
- **The mutable set's premise is unverified.** Stubbing `normalized_gaussian_filter` to identity —
  deleting the work the plan calls "the largest untapped win" in `operators.py` — **did not speed up
  ppd50** (4.51 s vs 4.15 s).
- **The honest headline:** the plan asked whether an agentic loop beats a competent assistant on a
  461-line file. The answer found instead is more useful: *on this codebase, both "existing"
  evaluators were broken, and the loop would have optimized against a gate asserting a 3 °C Gulf of
  Mexico.* The brief's thesis ("the evaluator, not the agent") is **strengthened** — the assumption
  that a trustworthy evaluator already existed is what failed. That is the publishable-adjacent
  result, and it cost ~$0.

Keep the hardened evaluator + goldens as the durable deliverable; note Track B is revivable if the
`v1`-only fitness ever becomes affordable.

---

## Step 3 — RC-4: MC-dropout ensemble (the highest-value diagnostic left)

The only RC not saturated or slack, and it targets the measured under-dispersion. **Zero retraining:**
`dropout_prob: 0.2` is live in `config/argo/config_argo.json` and `config_argo_anom.json`, and
`nn.Dropout` sits in the head (`model/model.py:14`, `:48`).

**Implement:**
1. An MC-dropout inference variant. `nb_metrics.py::run_inference` (`:273`) calls `model.eval()`,
   which disables dropout — add an opt-in (e.g. `mc_samples: int = 0`) that, after `eval()`,
   re-enables **only** `nn.Dropout` modules (`for m in model.modules(): if isinstance(m, nn.Dropout): m.train()`),
   then runs N forward passes. Keep the default path byte-identical so existing evals don't move.
2. Reconstruct each member with `model/loss.py::reconstruct_physical_profiles(..., clim_profiles=, indices=)`
   — **required**, or anomaly configs produce garbage (see `readiness-anomaly-cache-gotchas`).
3. Fill `diagnostics/readiness.py::uncertainty_calibration_hook` (`:228`), whose interface is already
   specified (`ensemble_mean`, `ensemble_spread`, `target`, `depth`, `subset_metadata`) and whose
   metric list is already written (`PLAN.md:686-689`): spread-error ratio, ENCE, reliability by
   depth, CRPS if feasible.
4. Wire it through `readiness_from_checkpoint` behind a flag; keep `not_implemented` when N=0.

**Pre-registered prediction:** spread-error ratio **< 1** (over-confident). If it reports ≈1,
**suspect the calibration code before believing the models are calibrated** — nothing else measured
so far is consistent with that.

**Expected honest caveat to state in the writeup:** MC dropout measures only the epistemic spread
dropout induces. It will *not* capture the PC-shrinkage under-dispersion, which is a systematic bias
of an MSE-optimal conditional mean, not a variance the model knows about. So spread ≪ error is the
likely result, and that is a finding about the metric's reach, not just the model.

**Verify:** run on `golden_point` (raw cache, simplest path) and `anom_point`; N≈50; confirm
ensemble_mean ≈ the deterministic prediction (sanity), and that RC-1/RC-2 are unchanged.

---

## Step 4 — Diagnose `point_cube`'s 38.52% σ₀ — **DONE. Verdict: Phase 8 must NOT proceed.**

> **Result (2026-07-15)** — full writeup in `HANDOFF-2026-07-15-agentic-close-out.md` Step 4.
> - 🚨 **The reference was wrong:** "nature's 24.70%" is the **PCA-16 regression target**. Nature =
>   **1.12%** (test) / 3.88% (n=4145); the target = 21.83% / 24.73%. **The basis truncation, not the
>   model, is the dominant source of σ₀ inversions.** Track 0's "not a PCA artifact" control was
>   vacuous — it compared the PCA-16 number to itself.
> - **Probe 1 (PC-shrinkage) FALSIFIED the hypothesis below:** `point_cube` last-8 = **0.220** vs
>   `golden_point` 0.196, while `residual_cube` carries **more** (0.249) with 15× fewer violations
>   and `anom_point` the **most** (0.341) with 7.4%. Ordering uncorrelated with σ₀.
> - **Actual cause:** 100% of the 750 violations are at **2.5–11.5 m** (not "spread thinly"), all
>   **marginal** (0.010–0.02 kg/m³; 0.00% at tol=0.03). `point_cube` reproduces the near-surface
>   halocline **10× too weakly** (dS(0→15 m) **+0.016** vs nature's **+0.14**) → density decreases
>   with depth at 3.5–5.5 m, where 37.9% of profiles violate.
> - **Probe 2 (standardization contrast) landed:** the cube's **SSS correlates only 0.744** with the
>   point cache's (RMS diff 0.48 PSU, −13% variance) while **SST transfers at 0.9885**.
>   `residual_cube` escapes precisely because it anchors on the point block's original SSS.
> - **Honest limit:** the SSS degradation is *contributing*, not proven-sole (halocline info
>   −0.616 → −0.549 is modest against a 10× collapse). Second suspect, not chased:
>   **`point_cube` best_epoch = 27** (vs golden 313), plus z-scored-vs-raw input mismatch.
> - **Exit:** a σ₀ penalty would force a fabricated halocline. **Fix the cube SSS feature instead.**
>
> *Original step text preserved below.*

The only model less stable than nature (24.70%), a 15× outlier against the other four
(0.00–8.99%). **`PLAN.md` Phase 8 is already scoped to `point_cube` only and gated behind this
diagnosis — do not reach for a physics-loss term first.**

Two cheap probes, both reusing tooling already written this session:

1. **PC-shrinkage contrast (most likely mechanism).** The Track 0 diagnostic showed `golden_point`
   shrinks high-order PCs to ~0.196× true std → 0.00% inversions. Run the same pred/true PC-std
   ratio for `point_cube`. **Hypothesis:** `point_cube` has a *higher* high-PC ratio — it injects
   amplitude into the fine modes without the accuracy to justify it (its T RMSE 0.5770 is *worse*
   than `golden_point`'s 0.5367 despite more structure). That would make its instability **noise in
   the high PCs**, not physics — and a σ₀ penalty would be treating the symptom.
2. **Standardization contrast.** `residual_cube` (2.57%) shares the cube feature path but anchors on
   the point block. Compare `input_standardization` between the two caches
   (`data/cache/train_ready_269d311a4b02.pkl` vs `train_ready_76aa50b84810.pkl`). Note the
   uncommitted `export_feature_cache.py` change added `point_block_norm` /
   `train_zscore_v2_point_sat_zscored` for the S0 anchoring contract — check whether `point_cube`'s
   cube features got a comparable z-score. See memory `residual-anchor-standardization`.

Supporting evidence for the "thin spread" reading: `point_cube`'s *interface* violation rate
(0.0669%) is close to nature's (0.0586%) — violations are spread across many profiles rather than
concentrated in a few, which looks like broadband input noise.

**Exit:** a stated cause (or a ruled-out list), and a decision on whether `PLAN.md` Phase 8 should
proceed at all.

---

## Step 5 — Fix `scripts/export_field_product.py` — **runs end-to-end; output NOT yet shippable**

> **Result (2026-07-15)** — full writeup in `HANDOFF-2026-07-15-agentic-close-out.md` Step 5.
> **Five** bugs, not two — and "expect more bugs downstream" was right in an unexpected place: the
> field **training** path had never run either.
> 1. `:33` `pickle.load(cache)` → `pickle.load(f)`
> 2. `:83` `float(ord(ds))` → new `base/split_utils.py::dates_to_juld` (exact inverse of
>    `sample_dates`, round-trip tested). **Must use `clim.meta["dataset_tag"]`**, not the field
>    cache's `argo_field` — `eval_climatology` decodes with the climatology's tag, and a mismatch
>    silently shifts the seasonal cycle. (`ord(ds)` returned a *constant* 50.0 for every 2000s date.)
> 3. `:94` `temp_arr[..., land]` applied a `(lat,lon)` mask to the **depth** axis → `IndexError`.
> 4. `FieldDataLoader` never set `l3_enabled`/`sat_patch_shape` → `train.py:294` `AttributeError`.
> 5. `FieldDataLoader.split_validation/split_test` returned a bare `DataLoader` lacking
>    `pca_models` → `metric.profile_rmse` `AttributeError`.
>
> **Structure verified:** xarray dims **`time/lat/lon/depth`** (8, 52, 68, 1801), land NaN-masked,
> mean T 23.48 → 4.01 °C with depth, `time` now `datetime64[D]`.
>
> **🚨 "Values physically plausible" is NOT met — and cannot be fixed in this script.** Salinity
> spans **22.13–45.02 PSU** (23% of finite points outside 30–38). **The climatology alone — no model
> involved — reaches 43.80 PSU at *ocean* points**, worst at **lat 31.0, lon −81.0**: the grid's NE
> corner, off the **Atlantic** coast of Georgia, *outside the Gulf*. It is a ridge fit on the ARGO
> hull (**lat 19.44–28.70, lon −95.33..−84.74**) but the export grid is **lat 18–31, lon −98..−81**;
> outside the hull it extrapolates. **17% of ocean points exceed 37 PSU.** Added
> `report_plausibility` as a tripwire (Track A instinct: ranges, not existence).
> **Follow-ups:** (a) clip the grid to the ARGO hull or refit/regularize the climatology;
> (b) re-verify with a *trained* field model, not the 2-epoch smoke.
>
> *Original step text preserved below.*

Two confirmed bugs prove it has **never run**:

- `:33` — `cache = pickle.load(cache)` inside `with open(cache_path, "rb") as f:` → **`NameError`**;
  should be `pickle.load(f)`.
- `:83` — `juld = np.full(..., float(ord(ds)), ...)` where `ds` is a date string like `"2020-01-01"`
  → **`TypeError`** (`ord()` wants a single char). Needs a real date→JULD conversion in the cache's
  convention; mirror how `eval_field.py` / `eval_climatology` obtain `juld` rather than inventing one.

**Expect more bugs downstream** — nothing past line 83 has ever executed. Budget for that; fix
forward rather than assuming two edits finish it.

**Verify (per the plan's own criterion):** writes a netCDF that opens in xarray with dims
`time/lat/lon/depth`, values physically plausible (reuse the Track A instinct: check T/S ranges, not
just that the file exists).

---

## Step 6 — Chase the anomaly parity gap — **DONE, LOCALIZED (no timebox needed)**

> **Result (2026-07-15)** — full writeup in `HANDOFF-2026-07-15-agentic-close-out.md` Step 6.
> Two probes localized it, exactly as the plan hoped.
> - **Probe 1 (PC shrinkage) — this is the gap.** `anom_point` shrinks the **dominant** mode:
>   **PC1 = 0.776×** true std vs `golden_point`'s **0.973×**. The plan's guess was right in spirit
>   ("the anomaly PCA spectrum is flatter → a model that regresses to the mean loses proportionally
>   more"), but it is **PC1**, not the high-order PCs — `anom_point` carries *more* high-PC amplitude
>   than `golden_point` (0.341 vs 0.196).
> - **Probe 2 — the discovery: eval-mode inference under-predicts anomaly PC1 by ~16–18%.**
>   MC-dropout averaging recovers it, **only for anomaly models**: `anom_patch_l4` +16.9%,
>   `anom_point` +18.3%, `retune_anom` +15.7% — vs `point_cube` +0.1%, `residual_cube` −0.0%,
>   `golden_point` +0.2%. **Confound resolved:** it tracks anomaly-ness, **not** training length
>   (`anom_patch_l4` at 1048 epochs already shows +16.9%, and the effect is flat across 1048→3382).
> - **The parity gap is ~62% smaller than reported.** Like-for-like (both MC-averaged):
>   `retune_anom` **0.5900** vs `golden_point` **0.5360** → gap **0.1436 → 0.0540**. Step 1
>   (corrected scales) gave 18%; Step 6 (inference) gives the rest. **The gap is substantially an
>   artifact of how inference is done, not of the anomaly reframing.**
> - **Probe 3 (T:S ratio) not needed** — Step 1 showed the whole knob is worth ~4%.
> - **Recommendation:** evaluate anomaly models with MC averaging, or (cheaper) **lower
>   `dropout_prob` for anomaly configs and retrain** — p=0.2 in the head is what makes the
>   weight-scaling approximation fail on a low-SNR target.
>
> *Original step text preserved below.*

Only if Step 1 confirms loss scales were not the cause. `anom_point` (0.6803) trails `golden_point`
(0.5367) by ~27% on T RMSE despite the anomaly reframing being theoretically favourable.

First probes, cheapest first:
- **PC shrinkage, anom vs golden** (reuse the Step 4 / Track 0 diagnostic): is `anom_point` shrinking
  its PCs *more*? The anomaly PCA is fit on anomalies, so its spectrum is flatter — a model that
  regresses to the mean loses proportionally more of the signal.
- **Climatology quality:** the target is `obs − clim`; a biased `clim_profiles` transfers its bias
  straight into the anomaly target. Known-good: anomaly caches' PCA EV is 99.7%/99.2% @16 PCs, so
  the basis is not the problem.
- **The T:S ratio (now 58.32:1)** — the plan calls this "a scientific choice wearing a normalization
  constant's clothes". It is the one genuinely tunable knob left, and Step 1 will have shown how
  little a ~10% move buys.

**Timebox this.** It is the only open-ended item here; if two probes don't localize it, write it up
as an open question rather than drifting.

---

## Verification

- **Step 1:** retune `T/S_rmse_native` recorded next to the 0.6803/0.1043 baseline; `readiness_*.json`
  for the retune checkpoint exists.
- **Step 3:** `readiness.py` uncertainty hook returns real metrics instead of `not_implemented`;
  default (non-MC) eval path unchanged — re-run `profile_metrics_from_inference` on `golden_point`
  and confirm the existing 0.5367 is reproduced bit-for-bit.
- **Step 4:** a written cause or ruled-out list; `PLAN.md` Phase 8 updated with the verdict.
- **Step 5:** `python -c "import xarray; xarray.open_dataset(out).dims"` shows `time/lat/lon/depth`;
  T/S ranges physically plausible.
- **Regression gate for any code change** (note the existing gate in
  `HANDOFF-2026-07-05-datacube-speed.md` omits `test_operators.py` — include it):
  ```
  pytest tests/test_sampler.py tests/test_cube_validate.py tests/test_operators.py -q   # 20 passing now
  python3 scripts/bench_datacube_speed.py --cascade --json                              # verdict: pass
  ```
  Readiness selfchecks: `test_static_stability_readiness_synthetic`,
  `test_readiness_report_requires_lat_lon`.
- **Do not** run `--save-golden` with local `sampler.py`/`operators.py` edits — see memory
  `golden-files-cube-revision-provenance`.

## Non-goals

- Track B / OpenEvolve (retired above — revivable, not deleted).
- A physics-loss term before Step 4 states a cause.
- OSSE / nature-run / synthetic-obs design — separate, dissertation-chapter-scale.
- Committing: only when asked.
