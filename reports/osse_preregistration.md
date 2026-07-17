# Phase 6 — Toy OSSE preregistration

**Status:** PRE-REGISTERED 2026-07-17 (before Phase 5 matrix winner)  
**Plan:** [`PLAN-v2-recovery.md`](../PLAN-v2-recovery.md) §6  
**Branch:** `residual_cube`  
**Posture:** written *before* the ablation winner is known so E-table design cannot be shaped by which cell won.

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
E3  NeSPReSO casts (Phase-5 winner), R_fixed
E4  NeSPReSO casts, R_cal = Σ_T / Σ_S from dacov export (val-α applied)
E5  NeSPReSO casts, R_cal, QC: keep only casts with mean predicted σ < threshold
```

Primary claims (pre-registered):
- **E3 > E2** — beats operational-class synthetic-cast paradigm
- **E4 ≥ E3** — calibration buys analysis skill
- **E5** — calibrated σ as QC (report both skill and cast retention)

## 2. Setup (locked)

| pin | value |
|-----|-------|
| Truth | ISAS20 GoM monthly, test-era years; grid coarsened 0.5°; ~60 depth levels |
| Background x_b | train-era monthly climatology |
| Cast locations | real 2021 ARGO positions from cache; one analysis / month |
| B | vertical only: σ_clim(z) c(z,z') σ_clim(z'), Gaussian L_v=150 m |
| R_fixed | diag(depth-dependent test RMSE²) — Dai et al. convention |
| R_cal | full vertical `Σ_T`/`Σ_S` from `dacov.export_ts_covariance_lowrank` (or equivalent winner path), floored |
| H | truth-grid → cast levels |
| Horizontal spread | fixed Gaussian L_h=100 km for map scoring (same for all E) |

## 3. Scoring

Subsurface T RMSE by depth band; MLD & D26 error; stratified by Loop-Current-activity index (SSH-anomaly variance in LC box: active vs quiescent) and by season.

## 4. ISOP/MODAS-class baseline (§6.2)

Per-location ridge of joint-EOF PC scores from `(SLA, SST_anom, month harmonics)` fit on train era; decode to profiles.
Observation error for its casts = its per-level test RMSE.
Script target: `NeSPReSO2_onTemplate/scripts/isop_modas_baseline.py` (+ selfcheck vs published MLR GoM RMSE order of magnitude).

## 5. Artifacts

| path | role |
|------|------|
| this file | prereg (must precede OSSE runs; git timestamp) |
| `scripts/isop_modas_baseline.py` | §6.2 baseline |
| `dacov` Σ_T/Σ_S export | §4.4 — **required** before E4/E5 |
| `reports/osse_results.md` | E-table after runs |
| `reports/pseudoobs_error_structure.md` | §6.5 standalone |

## 6. Explicit non-goals

- No time-stepping / full EnKF.
- No horizontal B in v1 (column-wise OI; ceiling noted in PLAN).
- Winner identity does not change E0–E5 design — only which checkpoint fills E3–E5.
