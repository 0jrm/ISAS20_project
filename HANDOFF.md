# Session handoff — dissertation data foundation

**Branch:** `residual_cube`  
**Updated:** 2026-07-20  
**Code home:** [`NeSPReSO2_onTemplate/`](NeSPReSO2_onTemplate/)  
**Conda:** `nespreso` (sha `621d0d65…`)

---

## Status: Phase 6 cast-column OSSE **v1 done**

**Winner:** A×CRPS · strata: [`phase5_A_CRPS_physical_strata.md`](reports/phase5_A_CRPS_physical_strata.md)  
**OSSE:** [`osse_results.md`](reports/osse_results.md) · `saved/runs/phase6_osse/cast_column_s42.json`

| claim | result (2021 casts, n=1101) |
|-------|------------------------------|
| E3 > E2 | **FAIL** (0.545 vs 0.541 — ISOP ≈ NeSPReSO) |
| E4 ≥ E3 | **FAIL** (0.616 vs 0.545 — R_cal v2 full localized Σ_T) |
| E5 | retention 0.444; RMSE 1.40 (QC drops → background) |

**R_cal v2 finding (2026-07-20):** structured Σ_T = V diag((ασ)²) Vᵀ, Schur-localized (`L_loc = L_v = 150 m`), is now **OI-stable** (raw full Σ blew up, cond(B+R)~2e8). But the CRPS-head cross-level correlations **hurt**: E4 = 0.616 vs **diag-control 0.546** (same code, `--rcal diag`) — off-diagonals degrade the column-OI increment. Diagonal preferred. The head is trained for marginal (per-dim) CRPS; its off-diagonals are basis-induced (shared `V`), not true obs-error structure. `diag(Σ)` preserved by localization ⇒ σ̄/E5-τ unchanged. Prereg locked pre-run (`0422a51`).

**Caveats (labeled):** cast-column proxy (no 2021 ISAS grid); E0≡E1; v1 diag artifacts superseded by v2 full-localized promotion.

### Next

1. Map-level ISAS truth + L_h when 2021 (or test-era) ISAS months available.
2. ~~Full-Σ R_cal localization~~ **done** (v2, negative: off-diagonals hurt). Ceiling: a *learned* joint covariance head (not marginal CRPS) if cross-level structure is wanted; else keep diag.
3. Optional `--iso-ensemble` on A×CRPS; quantile / error-channel gated on v3 HDF5.
4. §6.5 `pseudoobs_error_structure.md`.
5. **A_CRPS_z32** (physical CRPS, PCA-32) is registered: [`reports/A_CRPS_z32.json`](reports/A_CRPS_z32.json). Next ablation prompt: [`reports/NEXT_A_CRPS_z32_roni_ops_heave.md`](reports/NEXT_A_CRPS_z32_roni_ops_heave.md) (RONI, 19 ops, heave vs that baseline).

### Building blocks ready

OSSE prereg · ISOP scaffold · `export_ts_covariance_pca` · `run_osse.py --mode cast_column`
