# Session handoff — dissertation data foundation

**Branch:** `residual_cube`  
**Updated:** 2026-07-17  
**Code home:** [`NeSPReSO2_onTemplate/`](NeSPReSO2_onTemplate/)  
**Conda:** `nespreso` (sha `621d0d65…`)

---

## Status: Phase 6 cast-column OSSE **v1 done**

**Winner:** A×CRPS · strata: [`phase5_A_CRPS_physical_strata.md`](reports/phase5_A_CRPS_physical_strata.md)  
**OSSE:** [`osse_results.md`](reports/osse_results.md) · `saved/runs/phase6_osse/cast_column_s42.json`

| claim | result (2021 casts, n=1101) |
|-------|------------------------------|
| E3 > E2 | **FAIL** (0.545 vs 0.541 — ISOP ≈ NeSPReSO) |
| E4 ≥ E3 | **FAIL** (0.546 vs 0.545 — tied within noise; diag R_cal) |
| E5 | retention 0.44; RMSE 1.39 (QC drops → background) |

**Caveats (labeled):** cast-column proxy (no 2021 ISAS grid); E0≡E1; R_cal = **diag(Σ_T)** only (full Σ OI unstable).

### Next

1. Map-level ISAS truth + L_h when 2021 (or test-era) ISAS months available.
2. Full-Σ R_cal localization / sqrt-filter upgrade.
3. Optional `--iso-ensemble` on A×CRPS; quantile / error-channel gated on v3 HDF5.
4. §6.5 `pseudoobs_error_structure.md`.

### Building blocks ready

OSSE prereg · ISOP scaffold · `export_ts_covariance_pca` · `run_osse.py --mode cast_column`
