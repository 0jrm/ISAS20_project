# Phase 6 — Toy OSSE results (cast-column v1)

**Mode:** `cast_column` — year=2021 n=1101
**Casts:** 1101 · levels=60 · L_v=150.0 m
**Winner ckpt:** `/unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project/NeSPReSO2_onTemplate/saved/phase5_matrix/A_CRPS_v2/models/NeSPReSO2_ARGO_GoM_p5_A_CRPS_v2_p5_A_CRPS_v2_s42_s2/p5_A_CRPS_v2_s42_s2/model_best.pth`
**E5 τ (val P50 σ̄):** 0.3449
**R_cal val inflation:** 1.000× (mean diag(Σ) → mean RMSE²)
**R_cal form:** diagonal of Σ_T only (full matrix off-diagonals destabilize v1 column OI)

> Mode `cast_column`: truth = ARGO at cast columns. Map-level ISAS20 + L_h not wired (no 2021 ISAS year on disk).
> E0≡E1 when background is monthly clim and E1 casts are the same clim.

## E-table (overall T RMSE at cast columns)

| E | R | overall T RMSE | retention |
|---|---|----------------|-----------|
| E0 | none | 1.5382 | — |
| E1 | R_fixed_clim | 1.5382 | — |
| E2 | R_fixed_isop | 0.5410 | — |
| E3 | R_fixed_nespreso | 0.5454 | — |
| E4 | R_cal | 0.5463 | — |
| E5 | R_cal_QC_tau=0.3449_keep=0.444 | 1.3934 | 0.444 |

## Claims

- `E3_gt_E2`: **FAIL**
- `E4_ge_E3`: **FAIL**

## By depth band

| E | 0-100 | 100-300 | 300-700 | >700 |
|---|------|------|------|------|
| E0 | 2.1986 | 3.1613 | 2.1013 | 0.3772 |
| E1 | 2.1986 | 3.1613 | 2.1013 | 0.3772 |
| E2 | 1.1934 | 1.0184 | 0.6068 | 0.1523 |
| E3 | 1.1510 | 1.0231 | 0.6343 | 0.1673 |
| E4 | 1.1575 | 1.0203 | 0.6358 | 0.1680 |
| E5 | 1.9420 | 2.8169 | 1.9453 | 0.3525 |
