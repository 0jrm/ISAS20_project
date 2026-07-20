# Phase 6 — Toy OSSE preregistration

**Status:** PRE-REGISTERED 2026-07-17 (before Phase 5 matrix winner)  
**Plan:** [`PLAN-v2-recovery.md`](../PLAN-v2-recovery.md) §6  
**Branch:** `residual_cube`  
**Posture:** written *before* the ablation winner is known so E-table design cannot be shaped by which cell won.

### Changelog

| Date | Change | Why |
|------|--------|-----|
| 2026-07-17 | Initial E0–E5 lock + pins | Pre-winner scientific posture |
| 2026-07-17 | Lock E5 QC rule, LC box, depth bands, OI constants, artifact paths | Close under-specified knobs before physical-space winner lands |
| 2026-07-17 | Winner = A×CRPS; cast-column v1 runner lands (ISAS 2021 grid absent) | E3–E5 use A-path `export_ts_covariance_pca`; R_cal = diag(Σ) in v1 |
| 2026-07-20 | **R_cal v2: full `Σ_T = V diag((ασ)²) Vᵀ` Schur-localized** (Gaussian `L_loc = L_v = 150 m`, reuses locked pin — no new tuned number); `diag(Σ∘ρ)=diag(Σ)` ⇒ σ̄ and E5-τ unchanged. Locked **before** the promotion test number was read. | Closes HANDOFF Next #2. §1/§2 always specified full vertical `Σ_T`; v1 diag was a documented stopgap because raw rank-`n_T` Σ is near-null over `n_z≫n_T` levels (cond(B+R)~2e8) and destabilizes column OI. Localization restores full rank while keeping the structure the CRPS head resolves. Claim rules (E3>E2, E4≥E3) **unchanged**. |

---

## 0. Why now (CPU / matrix wall-clock filler)

Phase 6 E4/E5 (calibrated R vs fixed R) are the dissertation's headline DA claim.
They consume `Σ_T` / `Σ_S` from §4.4 Jacobian export — **gate-critical for Phase 6**, not a Phase 5 leftover.
OSSE prereg + ISOP/MODAS baseline do **not** need the matrix winner; locking them now is the stronger scientific posture.

---

## 1. Frozen experiment table

```
E0  no casts
E1  climatology casts, R_fixed
E2  ISOP/MODAS-class casts, R_fixed
E3  NeSPReSO casts (Phase-5 §3 physical-space winner), R_fixed
E4  NeSPReSO casts, R_cal = Σ_T / Σ_S from dacov export (val-α applied)
E5  NeSPReSO casts, R_cal, QC: keep only casts with mean predicted σ < threshold
```

| ID | casts `y` | R | QC |
|----|-----------|---|-----|
| E0 | none | — | — |
| E1 | train-era monthly clim at cast (x,y,month) | `R_fixed` | — |
| E2 | ISOP/MODAS-class ridge joint-EOF decode | `R_fixed` = diag(E2 per-level test RMSE²) | — |
| E3 | Phase-5 winner μ at cast surface inputs | `R_fixed` = diag(winner per-level test RMSE²) | — |
| E4 | same as E3 | `R_cal` = `dacov.export_ts_covariance_lowrank` (or A/B PC→T/S equiv.), floored | — |
| E5 | same as E4 | `R_cal` | drop casts with `σ̄ > τ` (see §2.1) |

**Primary claims (pre-registered; judged on cell-mean map T RMSE, depth bands in §3):**

| claim | rule |
|-------|------|
| **E3 > E2** | E3 overall subsurface T RMSE strictly lower than E2 |
| **E4 ≥ E3** | E4 overall T RMSE ≤ E3 (calibration does not hurt; report if it helps) |
| **E5** | report (i) overall T RMSE among retained casts / full-domain map after QC, (ii) cast retention fraction; no directional claim locked — exploratory labeled row |

Ties on overall T: lower MLD RMSE, then lower D26 RMSE.

---

## 2. Setup (locked)

| pin | value |
|-----|-------|
| Truth | ISAS20 GoM monthly, **test-era** years only; grid coarsened **0.5°**; depth subsampled to **~60** native levels (same subsample for all E) |
| Domain | GoM box matching ARGO cache / Phase-5 configs (same lat/lon mask) |
| Background `x_b` | train-era monthly climatology (per calendar month) |
| Cast locations | real **2021** ARGO positions from cache; **one analysis / month** |
| Seed | 42 (cast-month iteration order only; no random cast thinning) |
| `B` | vertical only: `σ_clim(z) c(z,z') σ_clim(z')`, Gaussian `L_v = 150 m` |
| `R_fixed` | `diag(depth-dependent test RMSE²)` — Dai et al. convention; **fit RMSE on test once per cast source** (E1 clim / E2 ISOP / E3 NeSPReSO), never retuned after first write |
| `R_cal` | full vertical `Σ_T` / `Σ_S` = `V diag((ασ)²) Vᵀ` (A-path `export_ts_covariance_pca`), **Schur-localized** in depth: `R = (Σ ∘ ρ) + 1e-8·I`, `ρ_ij = exp(-½((z_i−z_j)/L_loc)²)`, `L_loc = L_v = 150 m` (reuse locked B length; no new tuned pin). Val-α applied; val-only global inflation as v1. `dacov.localize_covariance`. `--rcal diag` reproduces the v1 diagonal fallback |
| `H` | truth-grid → cast levels (linear in depth) |
| Horizontal spread | fixed Gaussian `L_h = 100 km` for map scoring (**identical** for all E) |
| OI | column-wise univariate T and S after inversion; no time stepping |

### 2.1 E5 QC threshold (locked procedure — not a free number)

1. On **val** only, for each cast compute `σ̄ = mean_z √diag(Σ_T)` from the same `R_cal` path used at test. (Localization preserves `diag(Σ_T)`, so `σ̄` and `τ` are identical to the v1 diagonal form — the QC rule is unchanged by the v2 upgrade.)
2. Set `τ = median({σ̄_val})` (P50).
3. At test: keep cast iff `σ̄_test ≤ τ`.
4. Report retention % and skill; **do not** retune `τ` toward test skill.

### 2.2 Loop-Current activity index (locked)

| pin | value |
|-----|-------|
| LC box | lat **24–28°N**, lon **88–84°W** |
| Index | monthly SSH-anomaly variance in box (from same SLA product as inputs) |
| Split | months with index ≥ train-era median = **active**; else **quiescent** |

---

## 3. Scoring (locked)

| readout | definition |
|---------|------------|
| Subsurface T RMSE | depth bands: 0–100, 100–300, 300–700, 700–bottom m + overall |
| MLD / D26 | same `evalphys` definitions as Phase 5; coverage reported |
| Strata | LC-active vs quiescent × season (DJF/MAM/JJA/SON) |
| Headline for claims | overall map T RMSE after horizontal spread (E0–E5 comparable) |

---

## 4. ISOP/MODAS-class baseline (§6.2)

Per-location ridge of **joint-EOF PC scores** from `(SLA, SST_anom, month harmonics)` fit on **train** era; decode to T/S via `model.joint_eof.reconstruct_joint_eof`.
Observation error for its casts = its per-level test RMSE (`R_fixed`).
Script: `NeSPReSO2_onTemplate/scripts/isop_modas_baseline.py` (`--selfcheck` now; cache-backed fit with Phase 6 runner).

Order-of-magnitude selfcheck target: GoM T RMSE within factor ~2 of the published NeSPReSO-paper MLR baseline (not a pass/fail gate for E-claims).

---

## 5. Artifacts

| path | role |
|------|------|
| this file | prereg (must precede OSSE runs; git timestamp) |
| `NeSPReSO2_onTemplate/scripts/isop_modas_baseline.py` | §6.2 baseline |
| `NeSPReSO2_onTemplate/dacov/` (`export_ts_covariance_lowrank`) | §4.4 — **required** before E4/E5 |
| `NeSPReSO2_onTemplate/scripts/run_osse.py` | cast-column v1 runner (`--cast-year 2021`); E3–E5 = A×CRPS |
| `reports/osse_results.md` | E-table after cast-column v1 |
| `reports/pseudoobs_error_structure.md` | §6.5 standalone (independent of E outcomes) |

---

## 6. Explicit non-goals

- No time-stepping / full EnKF.
- No horizontal `B` in v1 (column-wise OI; ceiling = add `L_h` into `B`).
- Winner identity does not change E0–E5 design — only which checkpoint fills E3–E5.
- No post-hoc change of `L_v`, `L_h`, E5 `τ` rule, or LC box after test scores exist.
