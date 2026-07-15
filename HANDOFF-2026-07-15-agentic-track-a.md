# Session handoff — Agentic AI experiment, Track A (harden the evaluator)

**Date:** 2026-07-15
**Branch:** `residual_cube`
**Plan:** [`PLAN-agentic-ai-experiment.md`](PLAN-agentic-ai-experiment.md) Track A
**Prior:** [`HANDOFF-2026-07-15-agentic-track0.md`](HANDOFF-2026-07-15-agentic-track0.md) (Track 0)
**Status:** Track A **substantially complete**. Track B still gated (needs API key + `pip install openevolve`).

---

## Headline: the golden gate was broken at HEAD, and had been for 10 days

`PLAN-agentic-ai-experiment.md` opens by claiming the repo's key asset:

> **An AlphaEvolve-shaped evaluator already exists.** … gates correctness via `--check-golden`
> against `tests/golden/sampler_golden_*.npz` at `atol=1e-6` … **An agent cannot win by being fast
> and wrong** — precisely the silent-fabrication failure mode the brief identifies as the field's
> most dangerous.

**All three goldens failed against the repo's own cube**, and had since 2026-07-05. Worse, they were
anchored to **physically impossible data**:

| | `sst.value@local` |
|---|---|
| Golden (saved 2026-07-05) | **2.87 – 3.06 °C** |
| Actual cube (rev 3) | **22.67 – 29.63 °C** |
| Physical reality (Gulf of Mexico SST) | ~15 – 33 °C |

A 3 °C Gulf of Mexico. The gate was not merely stale — it was **asserting garbage**, and would have
**rejected every correct candidate** while presenting as a working safety net.

**Root cause (not a code bug — a provenance gap):**

| event | time |
|---|---|
| goldens saved | 2026-07-05 14:24 / 15:27 / 15:28 |
| cube rebuilt rev 2 → rev 3 | **2026-07-06T00:47:20Z** (= 07-05 20:47 EDT), ~5 h later |

The rebuild is the uncommitted `DATA_REVISION = 2 → 3` in `preproc/cube/cube_schema.py`:
*"rev 3: rebuild with single scale/offset decode (pre-fix rev-2 store was double-decoded)."* The
cube values were legitimately corrected; nothing re-derived the goldens, and no test compared them
against anything. The ~9× decode error is exactly the 27 °C → 3 °C ratio observed.

**Verified pre-existing, not introduced here:** with `sampler.py` reverted to the committed version
(my fix stashed), all three goldens still fail identically — `max |diff| = 2.78e+01` on
`sst.value@local`. The sampler code was never the problem; only the recorded values were.

### Why this matters more than the fix

Track 0 found `diagnostics/readiness.py` had never run and shipped two silent-wrong-number bugs.
Track A finds the speed evaluator's golden anchored to impossible data. **Both evaluators the plan
counted on as existing assets were broken.** The plan's thesis — *"the infrastructure layer is
commoditized and the differentiated artifact is the verification harness"* — survives, but its
premise that this repo *already had* that harness does not. The harness existed as *code*; it did
not exist as a *trustworthy artifact*. Those are different things, and the difference is the whole
plan.

Had Track B started first (per the plan's own gate: don't), the loop would have failed 100% of
candidates on the golden. The likely human response — "the golden must be stale, regenerate it" —
would have regenerated it **against a mutated sampler**, laundering an unverified mutation into the
correctness baseline forever. The gate that was supposed to prevent silent fabrication would have
been its delivery mechanism.

---

## Goldens regenerated — and the provenance chain used

Regenerated against the rev-3 cube using the **committed** `sampler.py` (my `weights_for` fix
stashed during generation), so the baseline is defined by reviewed, committed code plus the
corrected cube — **not** by any edit from this session.

| golden | n_profiles | new `sst.value@local` |
|---|---:|---|
| `sampler_golden_v1.npz` | 300 | 15.72 – 30.82 °C |
| `sampler_golden_ppd5.npz` | 300 | 12.36 – 30.82 °C |
| `sampler_golden_ppd50.npz` | **500** | 14.08 – 32.46 °C |

Then my fix was restored and `--check-golden` **passed** — proving the `weights_for` change is
**value-neutral** (it is: both call sites pass the full `lats`/`lons`, so the old key never
collided *within* a single `sample()` call).

Old goldens are recoverable — `tests/golden/` is git-tracked and nothing was committed.

> **`--n-profiles` trap:** the ppd50 golden was saved at **n=500**, not the CLI default of 300.
> Mismatching it used to raise a bare `ValueError` traceback; it is now a clean FAIL naming the
> cause. `GOLDEN_N_PROFILES` in the bench pins the right n per config — cross-check
> `tests/golden/*.meta.json` before trusting any run.

---

## Noise floor — the number Track B was gated on

`--repeat 10`, ppd50 (n=500), cold, pinned caches, shared box at load ~48:

```
min 3.809 s | median 4.145 s | sigma 0.305 s = 7.36% of median
all: 4.40 4.79 4.43 3.81 4.01 4.07 3.91 4.04 4.22 4.50
peak RSS 332 MB (ceiling 8192 MB)
```

### The cascade's cheap filter is its noisiest stage — this inverts the plan's design

Full cascade, `--repeat 3` (**verdict: pass**, peak RSS 1249 MB, all goldens pass):

| stage | config | n | min | median | **sigma** |
|---|---|---:|---:|---:|---:|
| 1 | ppd50 | 500 | 3.888 s | 3.990 s | **1.82%** |
| 2 | ppd1 (`v1`) | 300 | 172.593 s | 173.763 s | **1.31%** |
| 2 | ppd5 | 300 | 22.383 s | 22.974 s | **5.20%** |
| 2 | ppd50 | 500 | 3.600 s | 4.081 s | **17.33%** |

**The short configs are the unreliable ones.** ppd50 (~4 s) ranged from 1.82% to 17.33% σ across
runs — and 7.36% over 10 repeats — while `v1` (~173 s) sat at **1.31%**. Short benchmarks are
dominated by fixed overhead and system jitter; the long one averages it out.

This matters because the plan's cascade uses ppd50 as the **cheap stage-1 filter** and `v1` as the
expensive gate. That is exactly backwards from a signal-to-noise standpoint:

* **Stage 1 cannot do fine discrimination.** With σ up to 17%, a stage-1 threshold tuned to catch a
  20% regression will reject good candidates and admit bad ones at a high rate. Use it **only as a
  catastrophe filter** (reject candidates that are *multiples* slower, or that break the golden —
  the golden check is exact and noise-free, so stage 1 remains excellent at *correctness* rejection).
* **Fine speed discrimination only exists at `v1`** (σ 1.31%) — and `v1` costs ~173 s, so ~5 repeats
  ≈ 15 min per candidate. That is the real budget, and it is far above what the plan assumed.
* **Reframing:** stage 1 is a **correctness** filter that happens to be cheap; it is not a speed
  filter. Speed is measured at stage 2 or not at all.

### Rules Track B must honor or the loop optimizes noise

* **A single-shot measurement cannot resolve anything below ~20%** on the short configs. The plan's
  Phase-5 targets are in exactly that range.
* **Use `min` of N ≥ 5, not median or a single shot.** Minimum is the least contaminated estimator
  for timing — noise is one-sided: interference only ever adds time. (Note `min` was markedly more
  stable than `median` across these runs: ppd50 min varied 3.60–3.89 s while median varied
  3.99–4.15 s.)
* **Treat < 10% "improvements" as unproven** regardless of how confident a candidate's diff looks.
* The box is shared and was at **load ~48** during these measurements (a retrain on GPU 2 plus other
  users; the bench is CPU/Zarr so contention is indirect but real). **Re-measure the floor under the
  machine state Track B will actually run in** — these numbers are a realistic-conditions floor, not
  a best case.

---

## Negative controls — both gate paths verified

The plan asks for one; two were run, because a gate has two independent failure paths.

| control | result |
|---|---|
| **values drift** — stale rev-2 golden vs rev-3 cube | **caught**: `values differ, max abs diff=2.78e+01 (near 'sst.value@local')` |
| **valid_mask drift** — `normalized_gaussian_filter` stubbed to identity (a "fast + wrong" candidate) | **caught**: `valid_mask differs in 73 cells` |

The stubbed-filter candidate was rejected even though it was *not* actually faster (4.51 s vs
4.15 s) — at ppd50 the filter is not the bottleneck. Useful signal for Track B: **`operators.py`'s
double-float64 `gaussian_filter` may be a smaller win than the plan assumes at high
profiles-per-day.** The plan calls it "the largest untapped win"; that is untested at ppd50.

---

## Changes

| Path | Change |
|---|---|
| `scripts/bench_datacube_speed.py` | rewritten as a loop-ready evaluator (below) |
| `preproc/features/sampler.py` | `weights_for()` cache-key bug fixed; `_weights` bounded |
| `tests/test_sampler.py` | +3 tests (end-to-end analytic, weights-key regression, golden plausibility) |
| `tests/golden/sampler_golden_*.npz` | regenerated against rev-3 cube from committed sampler |

### Evaluator (`bench_datacube_speed.py`)

Every defect in the plan's Track A table is addressed:

* `--repeat N` → `elapsed_min`, `elapsed_median`, `sigma`, `sigma_pct`, plus every raw timing.
* `--json` → `{verdict, peak_rss_mb, results:[{elapsed_min, elapsed_median, sigma, golden, …}]}`.
* `--cold` (default) / `--warm` — cold builds a fresh provider per repeat so `make_profiles`'
  plane scan can't warm in-process caches into the timer. Reported in every result. The OS page
  cache is reported honestly as warm (dropping it needs root) rather than claimed cold.
* **Cache sizes pinned by the evaluator** (`PINNED_CACHES`), not read from `sampler.py`'s
  constructor defaults — that file is the mutable one, so a candidate could otherwise buy
  wall-clock with RAM. `--no-pin-caches` opts out explicitly.
* `--max-rss-mb` (default 8192) — memory ceiling asserted; breach ⇒ `fail`.
* Shape/name guards ⇒ clean FAIL naming the cause. **Any traceback ⇒ hard reject (exit 2)**, never
  a stack dump mistaken for a slow run.
* `GOLDEN_PATH` global removed → `golden_path_for(ppd)`, passed explicitly.
* `--cascade`: stage 1 (ppd50, ~4 s) filters, stage 2 (v1 + ppd5 + ppd50) gates; a stage-1 reject
  never pays for stage 2.
* Exit codes: `0` pass, `1` golden drift / RSS breach, `2` error.

### Tests (`tests/test_sampler.py`) — 20 passed

* `test_sample_end_to_end_on_linear_field` — the plan's missing end-to-end test. Asserts against
  **mathematics, not a recorded artifact**: Gaussian smoothing preserves a linear field exactly, so
  `value@local` must return the field; a linear field's gradient is constant everywhere. This is the
  test whose absence let a 3 °C Gulf of Mexico survive for 10 days.
* `test_golden_files_are_physically_plausible` — sanity-checks **the gate itself**. Loose envelopes
  (SST 5–40 °C, SSS 15–40 PSU, SSH ±2 m) — catches decode/unit regressions, not physics.
  **Verified against the old golden: it fails it** (`sst spans 2.86..3.04, outside [5.0, 40.0]`).
  SSS bound is 15, not 25: the Mississippi plume genuinely runs to ~24 PSU (my first bound was too
  tight and this test caught it). A ~9× decode error is still caught comfortably.
* `test_weights_for_keys_on_points_not_just_channel` — regression for the cache-key fix.

---

## Track B readiness

**Gate satisfied:** the evaluator emits JSON, exits nonzero on drift/traceback, and its noise floor
is a known number (**7.36%**). Still required before starting:

1. **API key** + `pip install openevolve` into `nespreso` — user decisions, not made here.
2. **Re-measure the noise floor** under Track B's actual machine state.
3. **Honor the ordering protocol**: Arm E first, and its diff must not be read until Arm H is done.
4. **Budget the estimator**: min-of-N ≥ 5 means ~5× the plan's per-candidate cost. At stage-1 ~4 s
   that is fine; at stage-2 v1 (~165 s) it is not. Expect the loop to live in stage 1.
5. **Re-check the goldens before the run.** They are now correct, but the failure mode found here
   is *provenance*, not code: any cube rebuild silently invalidates them again. Consider recording
   `data_revision` inside the golden `.meta.json` and asserting it at check time — that would have
   caught this in one second instead of ten days.

**Revised expectation for the `operators.py` win:** see the negative control above — stubbing the
Gaussian filter entirely did not speed up ppd50. The plan's "largest untapped win" is unverified.
Profile before believing it.
