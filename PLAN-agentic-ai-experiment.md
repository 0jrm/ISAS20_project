# PLAN — Agentic AI on NeSPReSO: measure, then a controlled closed-loop experiment

**Created:** 2026-07-15
**Status:** approved, not started
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

Intended outcome, in priority order:

1. Numbers where there are currently unknowns (cheap, hours).
2. A hardened evaluator — the actual deliverable in the brief's terms.
3. A controlled hand-vs-loop comparison yielding both a faster sampler *and* falsifiable data on
   where agentic AI helps and where it fails on this codebase.
4. DA prerequisites unblocked, then stop. OSSE design is a separate plan.

---

## Track 0 — Measure what already exists (~half a day, no agents)

Cheapest, highest-information work in the plan. Every item is an existing script producing a
currently-unknown number.

1. **Run `diagnostics/readiness.py` on the from-scratch checkpoints** (`scratch_0705_204716_*`,
   manifest at `notebooks/scratch_outputs/scratch_manifest.json`). CLI takes `-c/--config`,
   `-r/--resume`, `--split`, `--out`, `--md-out`; ~1 GPU-minute. Yields two unknowns:
   - RC-1 σ₀ violation rate (`readiness.py:75`) — is it 0.1% or 40%? This single number should
     decide how much weight physical realism deserves in any future objective.
   - RC-2 steric-vs-SLA RMSE + correlation (`readiness.py:159`) — the "matching observations"
     score, already coded and calibrated (alpha 0.88, r_train 0.81).
2. **Re-derive loss scales on the anomaly cache** — `scripts/derive_loss_scales.py --update-config`.
   Seconds on CPU. The open scoreboard item ("anom point below parity, suspect `loss_scales` tuned
   for raw-PC magnitudes") is likely closed by this, *not* by a search: the scales are analytic
   (zero the predicted PCs, measure reconstruction MSE), so searching them is a category error.
   Record before/after.
3. **Record what Track 0 found** in `HANDOFF.md`.

**Exit:** σ₀ violation rate and steric-vs-SLA correlation are written down. Anomaly loss-scale
question is closed or escalated with evidence.

---

## Track A — Harden the evaluator (~1 day, the real deliverable)

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

## Track B — The two-arm experiment (~1 week)

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

1. **MC dropout ensemble.** `dropout_prob: 0.2` is live in `config/argo/config_argo.json` and
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
- **No evolutionary search over `loss_scales`.** Analytic, not tunable. See Track 0.2. The only
  real knob there is the T:S *ratio* (currently 64:1) — a scientific choice wearing a
  normalization constant's clothes.
- **No search against val-split RMSE.** The multiple-comparisons trap. One split-confusion
  incident is already on record (`0.416` random-split headlined against chronological `0.514`). If
  a science loop ever happens, the objective should be **steric-vs-observed-SLA** — a held-out
  *observation type*, not a held-out sample, and far harder to overfit.
- **No manuscript generation.**

## Governance

One email to the committee chair documenting AI-assisted code and analysis, filed. No FSU-specific
ETD policy surfaced in search; Florida peers (USF, FIU) require committee permission plus
disclosure. The `HANDOFF-*.md` files are already ~80% of a disclosure log. The risk isn't the
policy — it's not having documented it contemporaneously.

## Verification

- **Track 0:** `readiness_*.json` and `.md` exist with real numbers; `scripts/results_table.py`
  reflects any loss-scale change.
- **Track A:** run the hardened evaluator 10× on unmodified `HEAD` — σ must be small relative to the
  speedups being chased. Deliberately inject a wrong-but-fast `sample()` and confirm the golden gate
  rejects it (a negative control for the evaluator itself — the brief's central warning).
- **Track B:** `pytest tests/test_sampler.py tests/test_cube_validate.py tests/test_operators.py -q`
  plus `--check-golden` on all three configs for any accepted candidate. Note the existing gate in
  `HANDOFF-2026-07-05-datacube-speed.md` omits `test_operators.py` — include it, since
  `operators.py` is now mutable. Re-export one cache and confirm training still loads it.
- **Track C:** `readiness.py` uncertainty hook returns real metrics instead of `not_implemented`;
  `export_field_product.py` writes a netCDF that opens in xarray with the expected dims.
