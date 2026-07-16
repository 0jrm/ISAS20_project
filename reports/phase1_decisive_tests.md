# Phase 1 decisive tests — PLAN-v2-recovery

**Date:** 2026-07-16  
**Branch:** `residual_cube`

## T2 — Stale-input audit (gate)

See [`stale_by_split.md`](stale_by_split.md).

| Gate | Status |
|------|--------|
| Stale fraction > 5% in val or test (SST, SSH/ADT, SSS) | **OPEN** — all splits 0.0% stale |

Headline metrics on the chronological test split are **not embargoed**.

## T1 — Joint vs separate basis reconstruction

See [`t1_basis_stability.md`](t1_basis_stability.md) and [`t1_basis_stability.json`](t1_basis_stability.json).

**Data:** raw ARGO profiles from `cache['profiles']` (not PCA-round-tripped `true_profiles`).  
**Split:** chronological 70/15/15, train-fit / test-eval.  
**Metric:** frozen `evalphys` N² violations at `N2_TOL=1e-8` s⁻².

| variant | viol_rate_profile | viol_rate_level | mean T/S RMSE | dρ/dz RMSE | MLD RMSE |
|---------|-------------------|-----------------|---------------|------------|----------|
| A separate PCA-16+16 | 0.766 | 0.00912 | 0.0893 | 0.0067 | 36.6 m |
| B joint EOF-32 | 0.886 | 0.00920 | 0.0886 | 0.0066 | 38.6 m |
| C density+spice PCA | 0.727 | 0.00892 | 0.0854 | 0.0068 | 39.4 m |
| D monotone σ₀ + spice | 0.387 | 0.00223 | 0.0696 | 0.0032 | 1.3 m |

### Pre-registered decision rules

| Rule | Outcome |
|------|---------|
| B or C cuts **level** violation rate ≥5× vs A at ≤10% RMSE cost | **NOT MET** — B/C ≈ A on level rate (~0.92%) |
| C ≈ A with no improvement | **ESCALATE** — C does not materially beat A on N² level rate |
| D violation rate ≡ 0 by construction | **NOT MET** on N² after (σ₀,τ)→(T,S) inversion; level rate **4.1×** below A |
| D RMSE cost (“price of hard stability”) | **0.0696** mean T/S RMSE vs **0.0893** for A (−22%) |

**Interpretation:** Joint EOF and density/spice PCA do **not** confirm Finding-1 at the N² metric. Monotone σ₀ (variant D) is the only representation that materially cuts stability violations, with **lower** T/S RMSE than A on this test split — Phase 3 should proceed on the **monotone-density** path, not joint EOF or separate density/spice PCA alone.

**Human gate:** B/C failure to meet the 5× rule is logged; D’s partial win avoids a full stop but changes the Phase 3 emphasis (3.2 monotone head is the load-bearing piece).

## T3 — Violation sensitivity (exclude top 15 m)

Tolerance sweep is attached in every `evalphys` stability summary (`N2_TOL ∈ {0, 1e-9, 1e-8, 1e-7}`).  
Excluding the top 15 m (near-neutral N² band) is computed in `summarize_physical(..., exclude_top15m)` — headline numbers remain full-column at `1e-8`.

---

## Next steps (plan dependency)

| Phase | Status |
|-------|--------|
| **0** evalphys v1.0.0 | **Done** — `evalphys/`, manifest, pytest green |
| **1** decisive tests | **Done** — this report |
| **2** SSS gap + error channels | Pending |
| **3** monotone density + spice training | Pending (D motivates 3.2) |
| **4–6** prob head, ablation, OSSE | Pending |
