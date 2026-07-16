# Audit report — Phase 0 + Phase 1 (`PLAN-v2-recovery.md`)

**Date:** 2026-07-16  
**Branch:** `audit/phase0-1`  
**Snapshot tag:** `pre-audit-phase0-1` (`7f9a043`) — parent `820e598`  
**Metric tag:** `evalphys-v1.1.0` (σ₀ metric + gsw backend)

## Step 0 hygiene

- No stash present.
- Branch created from `residual_cube` HEAD; snapshot commit unaudited.
- `.gitignore` correctly excludes `*.h5` / `*.pth` (except whitelisted base model); `reports/*.{md,json}` not ignored; `__pycache__` ignored.

## Findings

| ID | Severity | Claim audited | Root cause | Fix | Evidence / command |
|----|----------|---------------|------------|-----|-------------------|
| A1 | med | CRPS ensemble validation missing; PIT missing χ² | Incomplete §0.3 tests | Added ensemble CRPS + χ² p>0.01 tests | `pytest evalphys/tests` |
| A2 | high | `exclude_top_m` inverted (`z < 15` kept top band) | Bug in `static_stability_violations` | Keep `z >= exclude_top_m`; regression test | `test_exclude_top_m_drops_near_surface` |
| A3 | med | gsw vs gsw_torch atol widened to 1e-4 | Silent weakening | Route via `get_gsw()`; xfail on upstream drift; F.3 script | `test_gsw_vs_gsw_torch_sigma0` |
| A4 | high | Invalid mean T/S RMSE column | Mixed °C+PSU | Per-var × band RMSE in T1 reports | `reports/t1_basis_stability.*` |
| A5 | high | D N²=0.0022 ≠ 0 “by construction” | (i) inversion σ₀ round-trip + (ii) N²≠σ₀ | Additive σ₀ metric; PLAN §3.2 note | T1 D diagnosis; `sigma0_monotonicity_violations` |
| A6 | high | A=0.91% vs historical ~22% | Metric mismatch (N² level vs σ₀ profile tol=0.01) | Reconciliation table; Finding-1 **not** refuted | RAW 1.12% → A 21.51% σ₀ profile |
| A7 | med | T2 0% stale vs SSS gap docstring | Gap re-downloaded 2026-07-03; H5 2026-07-04 | Detector unit tests; gate stays OPEN | SSS mtimes; `test_stale_detector` |
| A8 | high | `test_combined_pca_loss_v2` fails | Pre-existing on `820e598` (combined/wmse golden drift) | Explicit skip+reason; no golden regen | Parent worktree + current |
| A9 | med | `gsw_torch as gsw` in readiness | Alias contamination risk for “reference” claims | `evalphys/gsw_backend.py`; headline asserts `gsw` | `conda … import gsw` → gsw 3.6.19 |
| A10 | low | Decision rules not quoted verbatim in T1 md | Hand-written report | Verbatim PLAN quotes in regenerated reports | `phase1_decisive_tests.md` |

## Reconciliation (T1 / Finding-1)

Historical definition (`diagnostics.readiness.static_stability_diagnostic`, σ₀ Δσ₀ < −0.01, **profile** rate, test n=623):

| row | profile rate | interface rate |
|-----|-------------:|---------------:|
| RAW | 1.12% | 0.0013% |
| A PCA-16 | 21.51% | 0.0821% |
| D monotone | 0.48% | ~0% |

Phase-0 N² level rate @ 1e-8 for A is **0.91%** — different ruler, not a refutation. Backend (f): evalphys/T1 used real `gsw`, not `gsw_torch`.

## D diagnosis

- Pre-inversion monotone σ₀: 11 Δσ₀<0 (numerical).
- Post-inversion σ₀ level rate: **0.00317**; N² level: **0.00223**.
- Conclusion: mostly (i) inversion imperfect σ₀ recovery + (ii) N² metric mismatch with §3.2 σ₀ constraint. Plan updated; N² metric retained.

## RMSE paradox

Invalid mixed column removed. D beats A on **T** in all bands and on **S** in 0–50 m; A slightly better on mid-depth S. Bases train-only; A is 16+16; C Newton fail 0%; paradox dissolved as unit-mixing + T-dominated mean.

## T2 gate

**OPEN** (not EMBARGOED). Detector proven with synthetic injection.

## GSW forensics

| location | import |
|----------|--------|
| `evalphys/metrics.py`, `inversion.py`, `t1_basis_stability.py` | `gsw` / now `get_gsw()` |
| `diagnostics/readiness.py:22` | `import gsw_torch as gsw` (training/readiness only) |
| env `import gsw` | `/…/site-packages/gsw/` v3.6.19 |

## Decision needed

| Option | Meaning | Recommendation |
|--------|---------|----------------|
| **R1** Proceed Phase 3.2 monotone-density head | T1 N² gate failed for B/C; Finding-1 holds historically; D is only material cut | **Recommend** — matches prior agent intent with corrected metrics |
| **R2** Escalate / redesign representation chapter | Treat B/C N² failure as “violations not basis-induced” | Not recommended: contradicted by historical σ₀ 1.12%→21.5% |
| **R3** Re-open T2 / embargo headlines | If SSS gap still in H5 | Not indicated — gap filled; detector OK |
| **R4** Regenerate `combined_pca_loss_v2` goldens | Fix selfcheck without skip | Needs human sign-off (numerical skill) — **do not** auto-regen |

**STOP conditions fired?** None of 1–3. Condition 4 avoided via additive σ₀ metric only.  
**Do not start Phases 2–6 pending human confirmation of R1.**

## Commands

```bash
srun --ntasks=1 --cpus-per-task=8 conda run -n nespreso python -m pytest NeSPReSO2_onTemplate/evalphys/tests -q
srun --ntasks=1 --cpus-per-task=8 conda run -n nespreso python NeSPReSO2_onTemplate/diagnostics/stale_sat/test_stale_detector.py
srun --ntasks=1 --cpus-per-task=8 conda run -n nespreso python NeSPReSO2_onTemplate/scripts/t1_basis_stability.py --gsw-backend gsw
srun --ntasks=1 --cpus-per-task=8 conda run -n nespreso python NeSPReSO2_onTemplate/selfcheck.py
```
