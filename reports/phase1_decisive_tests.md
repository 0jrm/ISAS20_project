# Phase 1 decisive tests — PLAN-v2-recovery

**Date:** 2026-07-16 (audit-corrected)  
**Branch:** `audit/phase0-1`  
**Tag baseline:** `pre-audit-phase0-1` → audit → `evalphys-v1.1.0`

## T2 — Stale-input audit (gate)

See [`stale_by_split.md`](stale_by_split.md).

| Gate | Status |
|------|--------|
| Stale fraction > 5% in val or test (SST, SSH/ADT, SSS) | **OPEN** — all splits 0.0% stale |

**Evidence (audit):** SSS gap `2021-01-01..2022-02-28` was re-downloaded 2026-07-03 (424 daily files under `/unity/g2/jmiranda/SubsurfaceFields/Data/CMEMS/SSS/`); satellite HDF5 regenerated 2026-07-04 00:49; cache rebuilt 2026-07-04 15:40. Detector unit tests inject a time-constant patch and assert detection; H5 keys `ostia/analysed_sst`, `ssh/adt`, `sss/sos` present. Test-split SSS temporal std median ≈ 0.035 (frac near-zero = 0). Headline metrics are **not embargoed**.

## T1 — Joint vs separate basis reconstruction

See [`t1_basis_stability.md`](t1_basis_stability.md) and [`t1_basis_stability.json`](t1_basis_stability.json).

**Data:** raw ARGO `cache['profiles']` (not PCA-round-tripped targets).  
**Split:** chronological 70/15/15; **bases fit on train only**.  
**Headline metric:** frozen `evalphys` N² at `N2_TOL=1e-8` (reference `gsw`).  
**Also reported:** σ₀-monotonicity level rate; per-variable × depth-band RMSE.

| variant | N² prof | N² level | σ₀ level | T RMSE 0–50 / 50–200 / 200–800 / >800 | S RMSE (same bands) | dρ/dz | MLD |
|---------|---------|----------|----------|----------------------------------------|---------------------|-------|-----|
| A separate PCA-16+16 | 0.766 | 0.00912 | 0.00974 | 0.202 / 0.223 / 0.107 / 0.016 | 0.108 / 0.038 / 0.020 / 0.001 | 0.0067 | 36.6 m |
| B joint EOF-32 | 0.884 | 0.00920 | 0.01105 | 0.227 / 0.232 / 0.099 / 0.017 | 0.087 / 0.032 / 0.015 / 0.001 | 0.0066 | 38.6 m |
| C density+spice PCA | 0.727 | 0.00892 | 0.00945 | 0.181 / 0.197 / 0.100 / 0.015 | 0.115 / 0.057 / 0.017 / 0.001 | 0.0068 | 39.4 m |
| D monotone σ₀ + spice | 0.387 | 0.00223 | 0.00317 | 0.153 / 0.156 / 0.088 / 0.016 | 0.066 / 0.056 / 0.021 / 0.002 | 0.0032 | 1.3 m |

C Newton fail rate: **0.0%**. D pre-inversion Δσ₀<0 count: **11** (numerical); post-inversion σ₀ level rate **0.00317** (inversion does not perfectly recover σ₀).

### Pre-registered decision rules (verbatim from PLAN §1-T1)

1. If B and/or C cut the level violation rate by ≥ 5× vs A at ≤ 10% RMSE cost ⇒ Finding-1 mechanism confirmed; Phase 3 proceeds as planned.
2. If C ≈ A (no improvement) ⇒ the violations are not basis-induced; escalate to human before Phase 3 (the representation chapter framing changes).
3. D should show violation rate ≡ 0 by construction; record its RMSE cost — this is the "price of hard stability" headline number.

| Rule | Outcome (corrected numbers) |
|------|-------------------------------|
| B or C cuts **level** N² rate ≥5× vs A at ≤10% RMSE cost | **NOT MET** — B/C ≈ A (~0.91%) |
| C ≈ A with no improvement | **ESCALATE** under N² level metric |
| D violation rate ≡ 0 by construction | **NOT MET** on N² (0.00223) or post-inv σ₀ (0.00317); hard constraint is σ₀-monotone on the control grid (PLAN §3.2 note) |
| D RMSE cost | D **improves** T RMSE vs A in all bands; S improves 0–50 m, slightly worse mid-depth — invalid mixed T/S RMSE column removed |

**Interpretation:** Under the **historical** Finding-1 definition (σ₀ profile rate, tol=0.01 kg/m³): RAW test **1.12%** → A PCA-16 **21.51%** — mechanism **confirmed**. Under the Phase-0 **N² level** metric used by the T1 gate, B/C do not buy a 5× cut. Monotone σ₀ (D) is the only representation that materially cuts both N² and σ₀ rates. Phase 3 should proceed on the **monotone-density** path (3.2), with residual N² reported (not assumed zero).

## Reconciliation (why T1 A looked like ~0.91% not ~22%)

| Candidate | Result |
|-----------|--------|
| (a) profile vs level | Dominates the optical illusion: A N² **profile** 76.6% vs **level** 0.91% |
| (b) N2_TOL=0 vs 1e-8 | Negligible for A (0.9157% → 0.9116%) |
| (c) split/cache | Same chronological test n=623, same cache `train_ready_4411c65ee518.pkl` |
| (d) N² vs Δσ₀ method | **Primary for 22%:** historical used readiness σ₀ Δσ₀<-0.01 **profile** rate (1.12%→21.51%) |
| (e) modes / standardization | A is 16+16 separate PCA on per-level z-scores, train-fit only |
| (f) gsw backend | Headline path is reference `gsw` 3.6.19 (not `gsw_torch as gsw`) |

## T3 — Exclude top 15 m

| variant | full-column N² level | exclude top 15 m |
|---------|----------------------|------------------|
| A | 0.0091 | 0.0074 |
| B | 0.0092 | 0.0074 |
| C | 0.0089 | 0.0069 |
| D | 0.0022 | 0.0016 |

Exclusion does **not** change the B/C vs A decision; headline remains full-column at `1e-8`.

---

## Next steps (plan dependency)

| Phase | Status |
|-------|--------|
| **0** evalphys | **Audited** — v1.1.0 (σ₀ metric + gsw backend) |
| **1** decisive tests | **Audited** — this report |
| **2** SSS gap + error channels | Pending (gap already filled; Phase 2.2+ remains) |
| **3** monotone density + spice training | Pending — human confirm after audit Decision needed |
| **4–6** | Pending |
