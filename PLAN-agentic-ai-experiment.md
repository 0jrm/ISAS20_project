# PLAN — Agentic AI on NeSPReSO: measure, then a controlled closed-loop experiment

**Created:** 2026-07-15
**Status:** **Tracks 0 and A complete; Track B RETIRED (not run); Track C moved to
[`PLAN-agentic-close-out.md`](PLAN-agentic-close-out.md), which supersedes the open half of this
plan.** Results: [`HANDOFF-2026-07-15-agentic-track0.md`](HANDOFF-2026-07-15-agentic-track0.md),
[`HANDOFF-2026-07-15-agentic-track-a.md`](HANDOFF-2026-07-15-agentic-track-a.md). Track 0.2 closed
**negative** (retrain: T 0.6803 → 0.6545, only 18% of the parity gap — loss scales were not the
cause). Tracks 0 and A invalidated several of this plan's own assumptions; the affected sections
below are revised and marked **[revised 2026-07-15]**.
**Source brief:** [`agentic-science.html`](agentic-science.html) (deep research, July 2026)
**Related:** [`PLAN_datacube_speed.md`](PLAN_datacube_speed.md) (Phase 5 is this plan's target),
[`PLAN.md`](PLAN.md) Phases 8–9 (uncertainty/ensemble scaffolding)

Use conda env `nespreso` to run any python code.

---

## Context

Prompted by `agentic-science.html` and the answers to its open questions: 4× A100 fair-share,
API-indifferent (public data / private code), user owns the verification harness, shadow agents
present, existing caches expendable. The goal is **not** "build an AI scientist" — it is:

1. automate and speed up NeSPReSO specifically, and
2. **investigate the benefits and limitations of agentic AI for this application**.

The science of interest is model skill: matching observations and improving data assimilation via
synthetic obs.

The brief's thesis is that the infrastructure layer is commoditized and the differentiated artifact
is the **verification harness** — "the evaluator, not the agent." Reconnaissance (2026-07-15) found
this repo already has, largely by accident, the setup the brief describes as ideal:

- **An AlphaEvolve-shaped evaluator already exists.** `scripts/bench_datacube_speed.py` times
  `CubeProvider.sample()` and gates correctness via `--check-golden` against
  `tests/golden/sampler_golden_*.npz` at `atol=1e-6` with an *exact* `valid_mask` match. An agent
  cannot win by being fast and wrong — precisely the silent-fabrication failure mode the brief
  identifies as the field's most dangerous.
  **[FALSIFIED 2026-07-15 — see [`HANDOFF-2026-07-15-agentic-track-a.md`](HANDOFF-2026-07-15-agentic-track-a.md)]**
  The evaluator existed as *code*; it did not exist as a *trustworthy artifact*. **All three
  goldens failed against the repo's own cube and had since 2026-07-05**, because the cube was
  rebuilt rev 2 → rev 3 (double-decode fix) ~5 h after the goldens were saved and nothing
  re-derived them. They asserted a **2.87–3.06 °C Gulf of Mexico** (true: 22.7–29.6 °C). The gate
  would have **rejected every correct candidate** — and the natural response ("golden must be
  stale, regenerate it") would have regenerated it against a *mutated* sampler, laundering an
  unverified mutation into the baseline. The anti-silent-fabrication gate would have become the
  fabrication mechanism. Fixed: goldens regenerated from the **committed** sampler + rev-3 cube;
  plausibility + end-to-end tests added so the gate cannot rot silently again.
  **Read this together with Track 0's finding that `readiness.py` had never run and shipped two
  silent-wrong-number bugs: *both* evaluators this plan counted on as existing assets were broken.*
  The brief's thesis ("the evaluator, not the agent") is *strengthened* — the plan's assumption that
  we already had one is what was wrong.
- **A control condition already exists.** `PLAN_datacube_speed.md` Phases 1–4 were done *by hand*
  and committed, with measured before/after (271→185s, 60→27s, 22→4.2s). Phase 5 is scoped and
  unclaimed.
- **Zero GPU contention.** The benchmark is numpy/Zarr-bound. The box has 256 cores at load ~4.3
  and 1 TB RAM; GPUs 0–1 were busy at survey time (one running another user's vLLM — the brief's
  "shadow agents" question, answered empirically). The loop runs on idle CPU with mutations from
  the API and touches no A100s.
- **The science evaluators exist and have never been run.** `diagnostics/readiness.py` computes σ₀
  static-stability violation rates *and* steric-height-vs-observed-SLA RMSE/correlation. No
  `readiness_*.json` exists anywhere on disk.
  **[revised 2026-07-15]** Run. `saved/readiness/readiness_*.{json,md}` now exist for all five
  scratch models. "Never been run" was load-bearing in a way this plan did not anticipate: the
  script carried **two latent bugs that returned plausible-looking wrong numbers rather than
  errors** — it fed anomalies (~0 °C / ~0 PSU) into GSW on any `*_anom` config, and RC-2 was never
  wired at all, always reporting `status:"ok"` with a null RMSE. A never-run evaluator is not a
  free asset; it is an *unvalidated* one. This is the brief's own silent-fabrication warning
  landing on our own diagnostic code — the same class of defect Track A exists to prevent in the
  speed evaluator, found here by accident. Both fixed; selfchecks pass.

Intended outcome, in priority order:

1. Numbers where there are currently unknowns (cheap, hours).
2. A hardened evaluator — the actual deliverable in the brief's terms.
3. A controlled hand-vs-loop comparison yielding both a faster sampler *and* falsifiable data on
   where agentic AI helps and where it fails on this codebase.
4. DA prerequisites unblocked, then stop. OSSE design is a separate plan.

---

## Track 0 — Measure what already exists — **DONE 2026-07-15**

Cheapest, highest-information work in the plan, and it paid off: it produced the two unknown
numbers *and* invalidated two assumptions this plan was built on. Full detail and controls in
[`HANDOFF-2026-07-15-agentic-track0.md`](HANDOFF-2026-07-15-agentic-track0.md).

**Prediction vs. outcome, recorded honestly** — the plan guessed wrong twice, which is the point of
measuring first:

| Plan said | Reality |
|---|---|
| RC-1 "is it 0.1% or 40%?" | **0.00%** (0 / 1,121,400 interfaces) for `golden_point` — *below* the plan's optimistic bound. Nature is 24.7%. |
| RC-2 = the "matching observations" score to optimize | **Saturated.** Model r=0.8299 vs true-profile ceiling r=0.8297. |
| Loss-scale item "likely closed by this" | Diagnosis **confirmed exactly**, but effect is only a 9.7% T:S rebalance. **Escalated, not closed.** |

1. **RC-1 σ₀ violation rate — `readiness.py` run on all five scratch models.**
   `golden_point` **0.00%**, `residual_cube` 2.57%, `anom_point` 7.38%, `anom_patch_l4` 8.99%,
   **`point_cube` 38.52%**. True ARGO profiles: **24.70%**.
   The models are **over-smoothed, not unstable** — verified by three controls (detector is
   sensitive; not a PCA artifact, since true PCs through the same PCA-16 basis still violate at
   24.7%; predictions are sane at T RMSE 0.53). Mechanism: PC1 is reproduced at 0.97× true std but
   **high-order PCs are shrunk to ~0.196×** — MSE-optimal regression to the mean. The high PCs
   carry the fine structure that *creates* real inversions.
   **Answer to "how much weight does physical realism deserve": ~none, for 4 of 5 models** — the
   constraint is already slack, and penalizing instability would push the model *further* from
   nature's real roughness. **`point_cube` is the sole exception and the only genuine candidate for
   a stability penalty.**
2. **RC-2 steric-vs-SLA — wired up (it never ran) and measured.** `anom_point` **r=0.8299 /
   RMSE 0.130 m**; true ARGO profiles through the identical pipeline **r=0.8297 / RMSE 0.141 m**;
   climatology floor RMSE 0.234 m ≈ obs SLA std 0.223 m. The model *equals and marginally beats
   truth* — smoothing removes profile noise that does not project onto SLA (same mechanism as
   RC-1; **RC-1 and RC-2 are two views of one fact**). The residual ~13 cm is irreducible: deep
   steric below 1800 m, barotropic signal, DUACS retrieval + collocation error.
   Computable on **anomaly caches only** — they alone carry `clim_steric` / `ssh_obs_sla` /
   `steric_calibration` (alpha 0.885, r_train 0.808).
3. **Loss scales — hypothesis confirmed, magnitude small.** `config_argo_anom.json` held
   T 2.0029 / S 0.0313, **byte-identical** to the raw cache's derived scales: copied from
   `config_argo.json` and never re-derived. The only stale config of five. Corrected to
   **T 1.3998 / S 0.0240** (mse 0.1561). The plan's instinct that this is analytic and not
   searchable was right. **But** profile_scales are *divisors*, so the error is a 0.733× global
   rescale (Adam is gradient-scale invariant and absorbs it) **+ only a 9.7% relative
   over-weighting of salinity** (T:S 63.99 → 58.32). **Unlikely to close the parity gap alone.**
   *Note:* `--update-config` reflows the whole JSON into expanded form; edit the numbers in place
   to keep the repo's compact style, or a 3-number change drowns in reformatting churn.

**Exit: met.** σ₀ and steric-vs-SLA are written down; the loss-scale question was escalated with
evidence to a deciding retrain (Track 0.4), which has since run and **closed it negative**.

### Track 0.4 — `anom_point` retrain — **DONE, closed-negative (2026-07-15 16:04)**

The retrain finished (early stop **epoch 3382**, EXIT=0). Seed 42 + chronological split pinned, so
`loss_scales` (T 2.0029/S 0.0313 → **1.3998/0.0240**) was the *only* difference from
`scratch_0705_204716_anom_point`.

**The pre-registered prediction held — no parity** (chronological test, n=623, native RMSE):

| model | T | S |
|---|---:|---:|
| `golden_point` (target) | 0.5367 | 0.0897 |
| `anom_point` (stale scales) | 0.6803 | 0.1043 |
| **retune (corrected scales)** | **0.6545** | **0.1013** |

T improved **−3.79%**, closing only **18%** of the gap to `golden_point` — a small improvement that
leaves the anomaly reframing unexplained, exactly the "most likely outcome" written below before the
run. A 9.7% T:S rebalance bought ~4% on T. **Verdict: stop blaming `loss_scales`.** Keep the
corrected values (analytically right, free); hand the parity gap to
[`PLAN-agentic-close-out.md`](PLAN-agentic-close-out.md) Step 6.

**⚠️ `val_loss` is not comparable across the two runs** — correcting `profile_scales` changed the
loss normalization itself (0.733× global factor), so baseline `mnt_best` 0.16285 vs retune 0.22589
is apples-to-oranges. Only `raw_profile_rmse_native` (above) is scale-independent. *Caveat:* n=1 per
arm — ~4% is not separable from trajectory noise without several seeds, but no decision turns on 4%.

**Free second read (RC-1/RC-2 on the retune checkpoint):** σ₀ profile violations 7.38% → **5.14%**
(slightly *more* over-smoothed, still far under nature's 24.70%); RC-2 r 0.8299 → 0.8313 (noise
against an already-saturated 0.8297 ceiling). Neither changes a decision.

- **Log:** `saved/readiness/retune_retune_0715_anom_point.log`
- **Checkpoint:** `saved/models/NeSPReSO2_ARGO_GoM_anom/retune_0715_anom_point/`
  (the original `scratch_0705_204716_anom_point` is untouched)
- **Artifacts:** `saved/readiness/retune_0715_anom_point_profile_metrics.json`,
  `saved/readiness/readiness_retune_0715_anom_point.{json,md}`

### Track 0.5 — `point_cube` σ₀ violation rate, **SCOPED** (was: unscoped finding)

`point_cube` violates static stability at **38.52%** against nature's 24.70% — the **only** model
less stable than the ocean, and a 15× outlier against the other four (0.00–8.99%). This is a real
defect Track 0 surfaced and this plan did not anticipate. Scoped, not deferred.

**The shape of the defect is already informative:** its *interface* violation rate (0.0669%) is
close to nature's (0.0586%), so violations are **spread thinly across many profiles** rather than
concentrated in a few bad ones — the signature of broadband noise in the cube features, not a
handful of outlier profiles.

Cheapest first move, before any physics-loss term: **contrast against `residual_cube` (2.57%)**,
which shares the cube feature path but anchors on the point block. A 15× stability gap across that
one architectural difference localizes the cause. Note [[residual-anchor-standardization]] — the
residual cache's point block is z-scored; check whether the cube features feeding `point_cube` are
standardized comparably, since unstandardized broadband features would produce exactly this
thin-spread instability.

**Do not reach for `PLAN.md` Phase 8's physics loss first.** If the instability is an input-noise
symptom, a σ₀ penalty treats the symptom and buys a smoother model that is wrong for a second
reason. `PLAN.md` Phase 8 is now scoped to `point_cube` **only**, and gated behind this diagnosis.

---

## Track A — Harden the evaluator — **DONE 2026-07-15**

Full detail: [`HANDOFF-2026-07-15-agentic-track-a.md`](HANDOFF-2026-07-15-agentic-track-a.md).

The suspicion recorded here — *"assume `bench_datacube_speed.py` has the same class of defect as
`readiness.py` until the negative control proves otherwise"* — **was correct, and understated.** The
defect was not in the code but in the **golden data**: it asserted a 3 °C Gulf of Mexico and failed
against HEAD's own cube (see the falsified bullet in Context). Every defect in the table below is
now fixed, plus that one, which the table did not anticipate.

**Delivered:**

| | |
|---|---|
| **Noise floor** | **σ = 0.305 s = 7.36% of median** (ppd50, n=500, 10 cold repeats; min 3.809 / median 4.145; spread min→max ≈ 26%) |
| Memory | peak RSS 332 MB against an asserted 8192 MB ceiling |
| JSON | `{verdict, peak_rss_mb, results:[{elapsed_min, elapsed_median, sigma, sigma_pct, golden, cache_state, …}]}` |
| Exit codes | `0` pass, `1` golden drift / RSS breach, `2` traceback (hard reject) |
| Cascade | stage 1 ppd50 (~4 s) filter → stage 2 (v1 + ppd5 + ppd50) gate |
| Tests | `tests/test_sampler.py` **20 passed**, incl. end-to-end analytic + golden-plausibility + weights-key regression |

**Negative controls — both gate paths verified (the plan asked for one):**

| control | result |
|---|---|
| values drift (stale rev-2 golden vs rev-3 cube) | **caught** — `max abs diff = 2.78e+01` on `sst.value@local` |
| `valid_mask` drift (Gaussian filter stubbed to identity — "fast + wrong") | **caught** — `valid_mask differs in 73 cells` |

**Three findings that change Track B's design:**

1. **σ = 7.4% means single-shot fitness cannot resolve anything under ~20%** — precisely the range
   of the Phase-5 targets. **Use `min` of N ≥ 5** (minimum is the least-contaminated timing
   estimator; interference is one-sided). Treat sub-10% wins as unproven.
2. **The cascade is upside-down: its cheap filter is its noisiest stage. [revises the Cascade
   section above]** Measured σ per config (`--cascade --repeat 3`, all goldens pass):

   | config | wall | sigma |
   |---|---:|---:|
   | ppd50 (stage-1 filter) | ~4 s | **1.8–17.3%** (7.4% over 10 repeats) |
   | ppd5 | ~23 s | 5.2% |
   | `v1` (stage-2) | ~173 s | **1.3%** |

   Short benchmarks are dominated by fixed overhead and jitter. So **stage 1 cannot do fine speed
   discrimination** — tune it as a *catastrophe* filter only (multiples-slower, or golden failure,
   which is exact and noise-free). **Stage 1 is a correctness filter that happens to be cheap; it is
   not a speed filter.** Real speed discrimination exists only at `v1` (σ 1.3%) at ~173 s → ~15 min
   per candidate at N=5. That is the true budget, well above this plan's estimate, and it undercuts
   the "~500 candidates ≈ $25" framing below: the loop is **evaluator-bound far harder than
   assumed** — which, note, *reinforces* the plan's own "volume beats brilliance" caveat pointing
   the wrong way. Re-cost Track B before launching.
3. **The `operators.py` "largest untapped win" is unverified.** Stubbing `normalized_gaussian_filter`
   to identity entirely — deleting the work the plan wants optimized — **did not speed up ppd50**
   (4.51 s vs 4.15 s baseline). At high profiles-per-day the double-float64 filter is not the
   bottleneck. **Profile before committing Track B's mutable set to that hypothesis.**

**Carry into Track B:** the failure mode found here was **provenance, not code**. Any future cube
rebuild silently invalidates the goldens again. Record `data_revision` in the golden `.meta.json`
and assert it at check time — that turns a 10-day silent rot into a one-second failure.

`scripts/bench_datacube_speed.py` is 80% of an evaluator but is **not loop-ready**. Fix in place:

| Defect | Location | Fix |
|---|---|---|
| Single shot, no warmup/repeats — fitness noise unmeasured | `:108-110` | `--repeat N`, report min and median; measure run-to-run σ first |
| Timing prints as prose, not JSON | `:139` | Emit JSON; `run()` at `:103` already returns `(payload, elapsed)` |
| `_day_is_usable()` pre-warms page cache before the timer | `:53-62` | Explicit `--cold` / `--warm` mode; report which |
| Cache-size knobs are constructor defaults *inside the mutable file* — reward-hack surface | `sampler.py:220` | Pin `plane_cache_size`/`stack_cache_size`/`derived_cache_size` from the evaluator; assert memory ceiling |
| Off-golden `--n-profiles` → uncaught `ValueError` traceback, not a clean FAIL | `:134-136` | Guard shapes; traceback ⇒ hard reject |
| `GOLDEN_PATH` mutated via `global` — state leaks across in-process runs | `:134` | Pass explicitly |
| `weights_for()` caches on channel alone, ignoring `lats`/`lons` — latent wrong-results bug if a provider is reused across candidates | `sampler.py:253-258` | Fix the cache key (pre-existing bug; fix before it gets blamed on the loop) |
| Golden `.npz` is the **sole** correctness net for `sample()` — no test exercises it end-to-end against the real cube | `tests/test_sampler.py` | Add an end-to-end numerical test |

**Cascade evaluator** (OpenEvolve supports this natively):

- Stage 1 filter: `ppd50` config, ~4.2s — cheap reject.
- Stage 2 gate: all three configs (`v1`, `ppd5`, `ppd50`), ~215s — survivors only.
- Parallelism: ~32-way across 256 mostly-idle cores.

**Exit:** the evaluator emits JSON `{elapsed_min, elapsed_median, sigma, golden: pass|fail}`, exits
nonzero on any drift or traceback, and its own noise floor is a known number. **Do not start Track B
before this is true** — an evolutionary loop on a noisy fitness function optimizes noise.

---

## Track B — The two-arm experiment — **RETIRED, not run (decided 2026-07-15)**

Retired on Track A's measurements, by this plan's own gate ("an evolutionary loop on a noisy fitness
function optimizes noise"). Revivable, not deleted — the design below stands if the economics change.

**Why it was not run:**

1. **The cascade is upside-down.** Its cheap stage-1 filter (ppd50, ~4 s) is its *noisiest*
   (σ 1.8–17.3%; 7.4% over 10 repeats), while the expensive `v1` (~173 s) is the quietest (σ 1.3%).
   Short benchmarks are dominated by fixed overhead and jitter. Stage 1 therefore **cannot
   discriminate speed** — it is a *correctness* filter that happens to be cheap.
2. **Real cost is ~15 min/candidate**, since fine discrimination exists only at `v1` at min-of-5 —
   against the "~500 candidates ≈ $25 … the loop is evaluator-bound, not token-bound" estimate
   below. It is evaluator-bound *far harder than assumed*. This plan's own reasoning, run on
   measured numbers, argues against running it.
3. **The mutable set's premise is unverified.** Stubbing `normalized_gaussian_filter` to identity —
   deleting the very work called "the largest untapped win" — **did not speed up ppd50** (4.51 s vs
   4.15 s baseline).

**What the experiment answered anyway — and it is the more useful answer.** The question was whether
an agentic loop beats a competent assistant on a 461-line file with a known plan. Tracks 0 and A
found something better-grounded: **on this codebase, both evaluators assumed to exist were broken.**
`readiness.py` had never run and shipped two silent-wrong-number bugs; the golden gate asserted a
**3 °C Gulf of Mexico** and failed against HEAD's own cube for 10 days. A loop launched on the
stated premise would have failed 100% of candidates, and the natural fix ("regenerate the golden")
would have laundered an unverified mutation into the baseline permanently.

The brief's thesis — *"the evaluator, not the agent"* — comes out **strengthened**. What failed was
this plan's assumption that a trustworthy evaluator already existed. That is a falsifiable,
publishable-adjacent result about agentic-science infrastructure, and it cost **$0 in API spend**.

The durable deliverable is the hardened evaluator + correct goldens (Track A), which stand on their
own regardless of whether a loop ever runs against them.

**Original design, preserved below for revival.**

Mutable set: **`preproc/features/sampler.py` + `preproc/features/operators.py`**. Including
`operators.py` unlocks the largest untapped win: `normalized_gaussian_filter` /
`normalized_gaussian_derivative` (`operators.py:53-81`) each run `ndimage.gaussian_filter` **twice
in float64**; separable/float32/single-pass lives here.

Cost, accepted deliberately: `@register(version=N)` feeds `list_operators()` → `_hash_payload()`
(`export_feature_cache.py:62`) → `train_ready_<hash>.pkl`, so any numerics change invalidates all
caches and forces re-export. This overrides the "keep operator versions unchanged" constraint in
`PLAN_datacube_speed.md`, which assumed caches were worth preserving. They are not.

**Arm E (treatment):** OpenEvolve 0.3.1 (pip-installable into `nespreso`) seeded from the current
Phase-4 `HEAD`, mutations from a frontier API, fixed budget. Cost estimate: ~500 candidates ×
(~10k in / 3k out) ≈ **$25 at Sonnet 5 introductory pricing**; even 2000 candidates ≈ $100. The
loop is evaluator-bound, not token-bound — volume beats brilliance here (CodeEvolve's finding).

**Arm H (control):** Claude Code, hand-optimizing Phase 5 + the `operators.py` wins from the *same*
Phase-4 baseline.

**Ordering protocol — matters, and is cheap:** run **Arm E first**, and do not read its diff until
Arm H is complete. Otherwise the loop's findings contaminate the hand arm and the comparison is
worthless. Pre-register the budget and the metrics before starting.

**Measure, per arm:** speedup vs. Phase-4 baseline on all three configs; wall-clock (human and
agent); dollars; and — the interesting half — **failure modes**: did it reward-hack the cache
sizes, break the golden, drift the `FeatureTable` contract consumed by
`export_feature_cache.py:25`, or produce a plausible-looking regression?

**Exit:** a faster sampler (either arm), plus a written comparison. Both a null result ("the loop
did not beat a competent assistant on a 461-line file with a known plan") and a positive one are
publishable-adjacent and answer the actual question.

---

## Track C — Unblock DA prerequisites, then stop (~2 days)

1. **MC dropout ensemble.** **[priority raised 2026-07-15]** Track 0 found the models are
   **under-dispersed** (high-order PCs at 0.196× true std; 0.00% σ₀ violations as the symptom), so
   this is no longer just a DA prerequisite — it is the track that targets the one crisp defect
   Track 0 actually measured. Expect the spread-error ratio to come out **< 1** (over-confident);
   that is the hypothesis to test.
   `dropout_prob: 0.2` is live in `config/argo/config_argo.json` and
   `nn.Dropout` is in the head (`model/model.py:157`). Keep dropout on at inference → ensemble
   spread with **zero retraining**. Fills `uncertainty_calibration_hook` (`readiness.py:218`),
   whose interface is already specified (`ensemble_mean`, `ensemble_spread`, `target`, `depth`,
   `subset_metadata`) and whose metric list is already written (`PLAN.md:686-689`: spread-error
   ratio, ENCE, reliability by depth, CRPS if feasible).
2. **Fix the gridded field product** — two one-line bugs prove it has never run:
   - `scripts/export_field_product.py:33` — `pickle.load(cache)` should be `pickle.load(f)`
   - `scripts/export_field_product.py:83` — `float(ord(ds))` on a date string `"2020-01-01"`
   This is the DA-relevant output (dims `time/lat/lon/depth` → netCDF), ~2 fixes plus a run away.
3. **Stop.** OSSE / nature run / synthetic-obs / covariance vocabulary has **zero hits** in this
   repo — that design is genuinely cold and is a separate, dissertation-chapter-scale plan.

---

## Deliberate non-goals

- **No LiteLLM / Langfuse / vLLM.** The brief's Phase 0 is an institutional answer to a problem a
  single researcher with public data does not have. API-indifference removes the sovereignty
  argument; the 256 idle cores remove the self-hosting argument. YAGNI (ponytail).
- **No evolutionary search over `loss_scales`.** Analytic, not tunable. See Track 0.2 — **confirmed
  2026-07-15**: the scales were simply stale (raw-cache values on the anomaly cache) and one
  `derive_loss_scales.py` run fixed them. A search would have burned budget rediscovering a
  closed-form answer. The only real knob is the T:S *ratio* (**measured: 63.99 stale → 58.32
  correct**) — a scientific choice wearing a normalization constant's clothes. **The retrain
  (Track 0.4) now quantifies the whole knob's reach: the full 63.99 → 58.32 move bought −3.79% on T
  RMSE.** So even a *perfect* search over this parameter wins ~4% — which retires it as a search
  target on measured grounds, not just analytic ones.
- **No search against val-split RMSE.** The multiple-comparisons trap. One split-confusion
  incident is already on record (`0.416` random-split headlined against chronological `0.514`).
- **No science loop against steric-vs-observed-SLA either. [revised 2026-07-15]**
  This plan originally proposed that if a science loop ever happens, its objective should be
  **steric-vs-observed-SLA** — a held-out *observation type*, not a held-out sample, and far harder
  to overfit. **The reasoning still stands; this specific metric does not.** Track 0 measured it and
  it is **saturated**: `anom_point` scores r=0.8299 against a true-profile ceiling of r=0.8297. The
  model is already at the observational limit, so the only thing left to optimize is the ~13 cm
  irreducible residual — deep steric below 1800 m, barotropic signal, DUACS retrieval and
  collocation error. **A loop pointed at this metric would optimize noise**, and worse, would be
  *rewarded* for over-smoothing (smoothing is why the model beats truth here). Any future science
  loop needs a different objective. Note this does **not** touch Tracks A/B, which optimize
  **speed** against a golden correctness gate.
- **The real open problem is under-dispersion, not realism or SLA skill. [added 2026-07-15]**
  The one crisp, measurable defect Track 0 found is the **0.196 variance ratio in the high-order
  PCs** (and 0.00% σ₀ violations as its symptom). It is not addressable by either objective above.
  It strengthens Track C's MC-dropout ensemble, which targets spread directly.
- **No manuscript generation.**

## Governance

One email to the committee chair documenting AI-assisted code and analysis, filed. No FSU-specific
ETD policy surfaced in search; Florida peers (USF, FIU) require committee permission plus
disclosure. The `HANDOFF-*.md` files are already ~80% of a disclosure log. The risk isn't the
policy — it's not having documented it contemporaneously.

## Verification

- **Track 0: DONE.** `saved/readiness/readiness_*.{json,md}` exist with real numbers for all five
  scratch models; `config_argo_anom.json` scales re-derived and `validate_config` passes; readiness
  selfchecks (`test_static_stability_readiness_synthetic`, `test_readiness_report_requires_lat_lon`)
  pass. **Still outstanding:** `scripts/results_table.py` does not yet reflect the loss-scale change
  — that needs the `anom_point` retrain, which is the escalation, not a verification gap.
  *Method note worth carrying into Track A:* the RC-1 result (0 violations in 1.12M interfaces) was
  only trustworthy **because** it was checked against a working detector, a
  not-a-PCA-artifact control, and sane-prediction evidence. A suspiciously clean number is a
  hypothesis, not a result — the same standard the negative control below applies to the speed
  evaluator.
- **Track A:** run the hardened evaluator 10× on unmodified `HEAD` — σ must be small relative to the
  speedups being chased. Deliberately inject a wrong-but-fast `sample()` and confirm the golden gate
  rejects it (a negative control for the evaluator itself — the brief's central warning).
- **Track B:** `pytest tests/test_sampler.py tests/test_cube_validate.py tests/test_operators.py -q`
  plus `--check-golden` on all three configs for any accepted candidate. Note the existing gate in
  `HANDOFF-2026-07-05-datacube-speed.md` omits `test_operators.py` — include it, since
  `operators.py` is now mutable. Re-export one cache and confirm training still loads it.
- **Track C:** `readiness.py` uncertainty hook returns real metrics instead of `not_implemented`;
  `export_field_product.py` writes a netCDF that opens in xarray with the expected dims.
