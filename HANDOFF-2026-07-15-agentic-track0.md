# Session handoff — Agentic AI experiment, Track 0 (measure what exists)

**Date:** 2026-07-15
**Branch:** `residual_cube`
**Plan:** [`PLAN-agentic-ai-experiment.md`](PLAN-agentic-ai-experiment.md) Track 0
**Status:** Track 0 **complete**. **`anom_point` retrain IN FLIGHT** (tmux `anom_retune`, GPU 2,
launched 15:10) — closes Track 0.2, runs in parallel with Track A. Track A (harden evaluator) next.

Track 0 asked for two currently-unknown numbers (RC-1 σ₀ violation rate, RC-2 steric-vs-SLA)
and a loss-scale re-derivation. All three are now written down. Two of the three answers
**invalidate assumptions the plan itself was built on** — read "Consequences for the plan".

---

## Headline

1. **RC-1 is not a constraint — the models are *smoother than nature*, not unstable.**
   `golden_point` produces **0 σ₀ inversions in 1,121,400 interfaces (0.00%)** while the true
   ARGO profiles violate at **24.7%**. Same PCA-16 basis on both, so this is not a basis artifact.
   Exception: `point_cube` at **38.5%** — *worse* than nature. RC-1 is model-specific, not global.
2. **RC-2 is saturated — the model is already AT the observational ceiling.**
   `anom_point` scores **r=0.830 / RMSE 0.130 m** against observed SLA. Feeding the *true* ARGO
   profiles through the identical pipeline scores **r=0.830 / RMSE 0.141 m**. The model equals
   (and marginally beats) the truth. There is **no headroom in this metric.**
3. **The loss-scale hypothesis is confirmed but small.** `config_argo_anom.json` really did carry
   raw-PC scales — byte-identical to `config_argo.json`. But the error is mostly a global rescale
   Adam absorbs; the real effect is a **9.7% T:S rebalance**. Unlikely to explain the parity gap
   on its own. **Escalated, not closed.**

---

## RC-1 / RC-2 scoreboard

`diagnostics/readiness.py`, test split, `saved/readiness/readiness_*.{json,md}` (first run ever).

| model | cache | n | σ₀ profile viol | σ₀ iface viol | RC-2 r | RC-2 RMSE | RC-2 status |
|---|---|---:|---:|---:|---:|---:|---|
| `golden_point` | raw | 623 | **0.00%** | 0.0000% | — | — | unavailable |
| `anom_point` | anom | 623 | 7.38% | 0.0061% | **0.8299** | **0.1295 m** | ok |
| `anom_patch_l4` | anom | 623 | 8.99% | 0.0757% | 0.7967 | 0.1415 m | ok |
| `point_cube` | cube | 623 | **38.52%** | 0.0669% | — | — | unavailable |
| `residual_cube` | cube | 623 | 2.57% | 0.0021% | — | — | unavailable |
| *TRUTH (raw cache)* | — | 4145 | *24.70%* | *0.0586%* | — | — | reference |
| *TRUTH (anom cache)* | — | 623 | *30.82%* | *0.0656%* | *0.8297* | *0.1405 m* | **ceiling** |
| *CLIMATOLOGY only* | — | 623 | — | — | *n/a (const)* | *0.2341 m* | **floor** |

RC-2 is only computable on the **anomaly caches** — they alone carry `clim_steric`,
`ssh_obs_sla`, `steric_calibration`. Raw/cube caches report an honest `unavailable`.
Observed SLA std = 0.2228 m (so the climatology floor ≈ predicting the mean).

### Why RC-1 is 0.00% — verified, not assumed

Three controls, because "0 out of 1.12M" is exactly the kind of number that is usually a bug:

- **Detector works:** true profiles → 24.7%; depth-reversing 50 profiles raises violations as
  expected. The diagnostic is sensitive.
- **Not a PCA artifact:** true PCs pushed through the *same* PCA-16 basis still violate at 24.7%.
- **Predictions are sane:** T RMSE 0.53 vs PCA-reconstructed truth, matching the scoreboard.

Mechanism — **the model reproduces PC1 at 0.97× true std but shrinks high-order PCs to ~0.13–0.20×**:

| PC group | pred/true std |
|---|---:|
| PCs 1–4 (mean) | 0.685 |
| last 8 PCs of each var (mean) | **0.196** |

That is textbook MSE-optimal conditional-mean regression to the mean. The high PCs carry the fine
vertical structure that *creates* real inversions; shrink them 5× and the inversions vanish. The
model is not "physically excellent" — it is **over-smoothed**, and 0% stability violations is a
*symptom of under-dispersion*, not a sign of health.

The same mechanism explains RC-2 beating truth: smoothing removes profile-scale noise that does
not project onto SLA, so the smoothed profile is a *better* SLA predictor than the real one.
**RC-1 and RC-2 are two views of one fact.**

---

## Consequences for the plan (read before Track B)

- **A σ₀/physics term in the objective is unmotivated for 4 of 5 models.** RC-1 was meant to decide
  "how much weight physical realism deserves." Answer: for `golden_point`/`residual_cube` the
  constraint is already slack — penalizing instability optimizes a satisfied constraint and would
  push the model *further* from nature's real roughness. **`point_cube` (38.5%) is the sole
  genuine candidate** for a stability penalty.
- **`PLAN-agentic-ai-experiment.md` "Deliberate non-goals" needs revising.** It states: *"If a
  science loop ever happens, the objective should be **steric-vs-observed-SLA** — a held-out
  observation type."* That objective is **saturated**: the model already matches the true profiles
  (0.8299 vs 0.8297). Searching against it would optimize the 13 cm irreducible residual —
  deep steric below 1800 m, barotropic signal, DUACS retrieval + collocation error — i.e. **noise**.
  The reasoning (held-out *observation type* beats held-out *sample*) is still right; this
  particular metric just has no room left. A science loop needs a different objective.
- **The interesting open problem is under-dispersion, not realism or SLA skill.** The measurable
  defect is the 0.196 variance ratio in the high PCs. That also makes Track C's MC-dropout ensemble
  more attractive: it directly targets spread.

---

## Bugs found and fixed in `diagnostics/readiness.py`

The script had never been run; both defects were latent and would have produced
**plausible-looking wrong numbers** rather than errors.

1. **Anomaly caches were silently scored on anomalies.** `readiness_from_checkpoint` called
   `pcs_to_profiles_depth_major`, which does **not** add climatology back. On any `*_anom` config it
   fed ~0 °C / ~0 PSU into GSW and reported a confident σ₀ number. Now uses the canonical
   `model.loss.reconstruct_physical_profiles(..., clim_profiles=, indices=)` (same helper as
   `eval_run` / `eval_matched` / `selfcheck`); a no-op on raw caches.
   *Verified:* steric height now 1.888 m mean — physically sane for GoM dynamic height re 1800 dbar.
2. **RC-2 was never wired.** `readiness_from_checkpoint` never passed `ssh` / `clim_steric` /
   `calibration`, so `steric_ssh_diagnostic` always returned `status:"ok"` with null RMSE — a
   green-looking result for a metric that never ran. Now pulls `ssh_obs_sla`, `clim_steric`,
   `steric_calibration` from the cache. When absent it returns **`unavailable`** instead of `ok`,
   because with `clim_steric=None` the old code compared an *absolute* steric height against an
   *anomaly* — a meaningless number that looked fine.

`to_markdown` now prints the RC-2 calibration, RMSE and correlation.
Selfcheck: `test_static_stability_readiness_synthetic`, `test_readiness_report_requires_lat_lon` **pass**.

> **Cache gotcha (cost me a wrong ceiling):** in an **anomaly** cache, `true_profiles` are stored as
> **anomalies** (T −22..+10, S −5.9..+1.4) while `clim_profiles` are **absolute** and
> **depth-major** — opposite orientation to `true_profiles` (sample-major). Truth must be
> reconstructed as `true_profiles[idx].T + clim_profiles[:, idx]`. In a **raw** cache
> `true_profiles` are already absolute. Feeding anomaly `true_profiles` to GSW yields a
> confident-looking 42 m steric height.

---

## Track 0.2 — loss scales

`config/argo/config_argo_anom.json` carried **raw-cache** scales. The raw golden cache
(`train_ready_3adcff404b0b.pkl`) derives to T 2.0029 / S 0.0313 / mse 0.2174 — **byte-identical** to
what the anom config held. It was copied from `config_argo.json` and never re-derived. Hypothesis
in `PLAN_datacube_speed.md` scoreboard: **confirmed**.

| | temperature | salinity | combined_mse_scale | T:S ratio |
|---|---:|---:|---:|---:|
| before (stale, raw-derived) | 2.0029 | 0.0313 | 0.2174 | 63.99 |
| after (anomaly-derived) | **1.3998** | **0.0240** | **0.1561** | **58.32** |

**Applied** to `config/argo/config_argo_anom.json` (2-line diff; `--update-config` reflows the whole
file into expanded JSON, so the numbers were edited in place to preserve the repo's compact style —
prefer that, or the real change drowns in reformatting). `validate_config` passes.

**Staleness survey — only this one config was affected:**

| config | status |
|---|---|
| `config_argo.json` | ok |
| `config_argo_anom.json` | **STALE → fixed** |
| `config_argo_point_cube.json` | ok (cube targets are raw PCs) |
| `config_argo_patch_l4_anom.json` | no `loss_scales` key |
| `config_argo_residual_cube.json` | no `loss_scales` key |

**But the effect is modest — this is escalated, not closed.** Scales are *divisors* normalizing each
branch to 1.0 at zero-pred. Under the stale scales:

- temperature branch contributed 0.699 (intended 1.0), salinity 0.767 → total 1.466 (intended 2.0)
- = a **0.733× global loss rescale** (largely absorbed by Adam, which is gradient-scale invariant)
  **+ a 9.7% relative over-weighting of salinity**

A ~10% T:S rebalance is a real but small lever. **It is unlikely to explain the anom-point parity
gap by itself.** The decisive test is a retrain (`anom_point` was ~2367 s / ~40 min, best epoch
3067) — **not run this session**; no GPU job was launched without asking.

---

## Files changed

| Path | Change |
|---|---|
| `diagnostics/readiness.py` | anomaly reconstruction; RC-2 wiring; honest `unavailable`; RC-2 in markdown |
| `config/argo/config_argo_anom.json` | loss scales re-derived on the anomaly cache (2 lines) |
| `saved/readiness/readiness_*.{json,md}` | **new** — 5 models, first readiness artifacts on disk |

Nothing committed — repo convention is commit only when asked.

---

## In flight — `anom_point` retrain (Track 0.2 closer)

Launched **2026-07-15 15:10**, running **in parallel with Track A** (GPU-bound on an idle A100 vs
Track A's numpy/Zarr CPU work — no contention).

```bash
tmux attach -t anom_retune                                   # watch
tail -f NeSPReSO2_onTemplate/saved/readiness/retune_retune_0715_anom_point.log
```

| | |
|---|---|
| Launcher | `scripts/run_anom_loss_scale_retune.sh` (GPU 2, `RUN_ID=retune_0715_anom_point`) |
| Checkpoint out | `saved/models/NeSPReSO2_ARGO_GoM_anom/retune_0715_anom_point/` |
| Baseline (untouched) | `saved/models/NeSPReSO2_ARGO_GoM_anom/scratch_0705_204716_anom_point/` |
| Controlled by | seed 42 + chronological split pinned → `loss_scales` is the **only** difference |
| Beat this | ANOM-point **0.680 / 0.104** (T/S RMSE, chrono test n=623) |
| Reference | point raw 0.537/0.090; golden 0.514/0.083 |
| Prior run | best epoch 3067, ~40 min, early_stop 500, 8000 max |

**On completion:** run eval for RMSE **and** re-run `diagnostics/readiness.py` on the new checkpoint
(RC-1/RC-2 are cheap now, and σ₀ is a free second read on whether the rebalance changed the
smoothing at all).

**Pre-registered prediction:** *this will not reach parity.* The stale scales were a 0.733× global
rescale (Adam absorbs it) plus a 9.7% salinity over-weighting. If T RMSE jumps 0.680 → ≈0.537 off a
10% rebalance, that is **surprising** and means the gap was never about scales. Expected: a small
improvement that leaves the anomaly reframing unexplained → **stop blaming `loss_scales`.**

---

## Next

1. **Track A — harden `scripts/bench_datacube_speed.py`** into a loop-ready evaluator. Unblocked,
   and unaffected by the findings above (it gates *speed* against a golden *correctness* check).
   Still the plan's real deliverable. **Do not start Track B first** — Track B is now explicitly
   gated behind Track A's noise floor, and needs an API key + `pip install openevolve`.
   *Track 0's warning for this track:* `readiness.py` shipped two bugs that returned confident wrong
   numbers instead of failing. Assume `bench_datacube_speed.py` has the same class of defect until
   the wrong-but-fast negative control proves otherwise.
2. **`point_cube`'s 38.5% σ₀ violation rate — now SCOPED**, not a loose finding. See
   `PLAN-agentic-ai-experiment.md` Track 0.5 and `PLAN.md` Phase 8 (now scoped to `point_cube`
   **only**). Cheapest first move: contrast against `residual_cube` (2.57%), which shares the cube
   feature path but anchors on the point block; check cube-feature standardization. **Diagnose
   before reaching for a physics loss** — violations are spread thinly across many profiles
   (interface rate 0.0669% ≈ nature's 0.0586%), which looks like broadband input noise, and a σ₀
   penalty would treat the symptom.
3. **Plan revised already** — steric-vs-SLA is retired as a science-loop objective (saturated), and
   `PLAN.md` Phase 9 (ensemble) now carries the under-dispersion prediction: spread-error ratio
   should come out **< 1**; if it reports ≈1, suspect the calibration code first.
