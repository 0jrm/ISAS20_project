# Phase 1 decisive tests — PLAN-v2-recovery

**Date:** 2026-07-16 (audit-corrected; R1 accepted)  
**Branch:** `audit/phase0-1` → merge to working branch  
**Tag:** `evalphys-v1.1.0`

## T2 — Stale-input audit (gate)

See [`stale_by_split.md`](stale_by_split.md).

| Gate | Status |
|------|--------|
| Stale fraction > 5% in val or test (SST, SSH/ADT, SSS) | **OPEN** — all splits 0.0% stale |

**Evidence:** SSS gap `2021-01-01..2022-02-28` re-downloaded 2026-07-03; HDF5 regenerated 2026-07-04; detector unit-tested. **Phase 2.1 satisfied.** Headline metrics not embargoed.

## T1 — Joint vs separate basis reconstruction

See [`t1_basis_stability.md`](t1_basis_stability.md) / [`t1_basis_stability.json`](t1_basis_stability.json).

**Data:** raw ARGO `cache['profiles']`. **Split:** chronological 70/15/15; bases **train-only**.  
**Headline:** `evalphys` N² @ `N2_TOL=1e-8` (reference `gsw`). Also: σ₀-monotonicity; per-variable × depth-band RMSE.

| variant | N² prof | N² level | σ₀ level | T RMSE 0–50 / 50–200 / 200–800 / >800 | S RMSE (same) | dρ/dz | MLD |
|---------|---------|----------|----------|----------------------------------------|---------------|-------|-----|
| A separate PCA-16+16 | 0.766 | 0.00912 | 0.00974 | 0.202 / 0.223 / 0.107 / 0.016 | 0.108 / 0.038 / 0.020 / 0.001 | 0.0067 | 36.6 m |
| B joint EOF-32 | 0.884 | 0.00920 | 0.01105 | 0.227 / 0.232 / 0.099 / 0.017 | 0.087 / 0.032 / 0.015 / 0.001 | 0.0066 | 38.8 m |
| C density+spice PCA | 0.727 | 0.00892 | 0.00945 | 0.181 / 0.197 / 0.100 / 0.015 | 0.115 / 0.057 / 0.017 / 0.001 | 0.0068 | 39.4 m |
| D monotone σ₀ + spice | 0.387 | 0.00223 | 0.00317 | 0.153 / 0.156 / 0.088 / 0.016 | 0.066 / 0.056 / 0.021 / 0.002 | 0.0032 | 1.3 m |

C Newton fail rate: **0%**. D pre-inversion Δσ₀<0 count: **11**; post-inversion σ₀ level **0.317%** (Newton round-trip is the dominant residual).

### Pre-registered decision rules (verbatim from PLAN §1-T1)

1. If B and/or C cut the level violation rate by ≥ 5× vs A at ≤ 10% RMSE cost ⇒ Finding-1 mechanism confirmed; Phase 3 proceeds as planned.
2. If C ≈ A (no improvement) ⇒ the violations are not basis-induced; escalate to human before Phase 3 (the representation chapter framing changes).
3. D should show violation rate ≡ 0 by construction; record its RMSE cost — this is the "price of hard stability" headline number.

| Rule | Outcome |
|------|---------|
| B/C ≥5× N² cut | **NOT MET** — escalate branch fired |
| C ≈ A | **ESCALATE** under N² (and under historical σ₀ — see below) |
| D ≡ 0 | **NOT MET** on post-inversion N²/σ₀; hard constraint is pre-inversion σ₀ on the control grid |
| Human escalation (R1) | **Accepted 2026-07-16:** proceed Phase 3.2 monotone-density head |

### Scientific update (escalation outcome)

The escalate branch fired, and the decision is: **proceed with the hard monotone constraint**. The interesting update is sharper than the original framing: **B (joint EOF) does not fix stability** under the historical σ₀-profile ruler (22.63% vs A's 21.51%). The load-bearing mechanism is **truncation itself**, not “separateness of T/S bases.” Soft representation changes do not buy stability; only the hard constraint does (**21.51% → 0.48%** profile rate on D).

**Honest caveat:** the hard constraint guarantees σ₀ monotonicity **pre-inversion**; the residual ~0.3–0.5% post-inversion comes from the Newton (σ₀,τ)→(T,S) round-trip. Phase 3 must track **inversion fidelity as a first-class metric** — it is now the dominant remaining violation source.

## Reconciliation — historical σ₀ profile rate (tol=0.01 kg/m³)

| row | σ₀ profile rate | interface rate |
|-----|----------------:|---------------:|
| RAW test | **1.12%** | 0.0013% |
| A PCA-16 | **21.51%** | 0.0821% |
| B joint EOF-32 | **22.63%** | 0.0802% |
| C density+spice | **21.83%** | 0.0748% |
| D monotone | **0.48%** | ~0% |

N² level @ 1e-8 for A (~0.91%) is a different ruler — not a refutation of Finding-1. Backend: reference `gsw` 3.6.19. Equivalence vs `gsw_torch`: [`backend_equivalence.md`](backend_equivalence.md).

## T3 — Exclude top 15 m

Exclusion does not change the B/C vs A decision; headline stays full-column at `1e-8`.

---

## Next (R1 accepted)

| Phase | Status |
|-------|--------|
| **0** evalphys v1.1.0 | Done + audited |
| **1** decisive tests | Done + audited; escalate → R1 |
| **2.1** SSS gap | **Satisfied** |
| **2.2** error-channel ingestion | **Next (∥ Phase 3)** |
| **3** monotone density + spice | **Next (R1)** — track inversion fidelity |
| **5** ablation matrix | Prerequisite: root-cause `combined_pca_loss_v2` golden (R4) before launch |
| **4, 6** | Pending |
