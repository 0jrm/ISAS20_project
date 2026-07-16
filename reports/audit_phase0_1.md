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
| A3 | med | gsw vs gsw_torch atol widened to 1e-4 | Silent weakening | Route via `get_gsw()`; xfail on upstream drift; F.3 table in reports | `reports/backend_equivalence.{json,md}` |
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
| B joint EOF-32 | **22.63%** | 0.0802% |
| C density+spice | **21.83%** | 0.0748% |
| D monotone | 0.48% | ~0% |

**Mechanism update (R1):** B does not beat A — load-bearing cause is **truncation**, not T/S separateness. Soft representation changes fail; hard constraint succeeds (21.51%→0.48%).

Phase-0 N² level rate @ 1e-8 for A is **0.91%** — different ruler. Backend equivalence table: [`backend_equivalence.md`](backend_equivalence.md) / `.json` (JOSS upstream evidence).

## Decision (R1 accepted 2026-07-16)

Proceed Phase 3.2 monotone-density ∥ Phase 2.2 error channels. Phase 2.1 satisfied. R4 golden = Phase 5 prerequisite. Inversion fidelity = first-class Phase 3 metric.

**STOP conditions fired?** None.

## Commands

```bash
srun --ntasks=1 --cpus-per-task=8 conda run -n nespreso python -m pytest NeSPReSO2_onTemplate/evalphys/tests -q
srun --ntasks=1 --cpus-per-task=8 conda run -n nespreso python NeSPReSO2_onTemplate/diagnostics/stale_sat/test_stale_detector.py
srun --ntasks=1 --cpus-per-task=8 conda run -n nespreso python NeSPReSO2_onTemplate/scripts/t1_basis_stability.py --gsw-backend gsw
srun --ntasks=1 --cpus-per-task=8 conda run -n nespreso python NeSPReSO2_onTemplate/selfcheck.py
```
