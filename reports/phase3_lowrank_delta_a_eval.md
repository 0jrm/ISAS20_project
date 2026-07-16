# Phase 3 — low-rank δa v1 (a-space) — FAIL / SUPERSEDED

**Checkpoint:** `saved/argo_densityspice_lowrank/models/NeSPReSO2_ARGO_GoM_densityspice_lowrank/lowrank_delta_a_v1/model_best.pth`  
**Cache:** `../data/cache/train_ready_910be6001098.pkl`  
**Split:** test n=623

**Gate:** FAIL (also vs corrected chrono floor 0.5903)

## Erratum — a-space PCA ceiling

PCA on train `(a_true − a_clim)` with R=16 has σ₀ recon RMSE **0.93 > clim 0.72**
despite ~94% a-space EVR: softplus⁻¹ makes a-space non-linear, so high EVR does not
transfer to σ₀. Pre-inv σ₀ was monotone (softplus path OK); skill never had a chance.
Superseded by σ₀-space PCA (`delta_sigma0_basis`, run `lowrank_sigma0_v2`).

Corrected comparison: overall T **0.8297** vs clean chrono argo16 **0.5367** (floor 0.5903).

**Gate (script, published-random ruler):** FAIL

- σ₀ profile rate: 0.8989 (near-zero: False) — post-inv; pre-inv neg dσ₀ = 0
- N² profile / level @ 1e-8: 0.5072 / 0.003457
- overall T RMSE: 0.8297 vs argo16 0.4158 (ratio 1.995)
- MLD RMSE: 50.247428605928405
- dρ/dz RMSE: 0.009709374493605984
- inversion fail frac: 0.0000

## T/S RMSE by depth band

| band | T RMSE | S RMSE | vs T1-A recon |
|------|--------|--------|---------------|
| 0-50 | 1.5515 | 0.5999 | 7.662 |
| 50-200 | 2.0716 | 0.5905 | 9.302 |
| 200-800 | 0.7777 | 0.1798 | 7.302 |
| >800 | 0.3357 | 0.0535 | — |

_Gate note:_ STOP uses σ₀ near-zero + overall T ≤ argo16×1.10 (trained separate-PCA). By-band vs T1-A is reconstruction floor (prediction>recon expected).

## Phase 2 caveat

T2 stale gate OPEN (0% SSS/SST/SSH on val/test). Full HDF5 lacks v3 error fields (`err_sla` / `analysis_error` / `sos_error` only in `*_err_smoke.h5`). density_spice cache has no `inputs_err` — headline metrics may be SSS-confounded only if stale returns; currently not. Formal product-error channels not in model inputs.
