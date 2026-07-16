# Phase 3 — low-rank δσ₀ v2 (σ₀-space PCA) — near miss

**Checkpoint:** `saved/argo_densityspice_lowrank/models/NeSPReSO2_ARGO_GoM_densityspice_lowrank/lowrank_sigma0_v2/model_best.pth`  
**Cache:** `../data/cache/train_ready_0f6129b27ddb.pkl`  
**Split:** test n=623

**Gate (corrected chrono floor 0.5903):** FAIL by 1.3% — T **0.5981**

Oracle ceiling (true R=16 scores + true spice): T RMSE **0.098**.  
Blame-swap: pred dens + true spice → **0.265**; true dens + pred spice → **0.424**.  
Spice is the binding constraint (mse_spice≈60; dens mse_σ_z≈0.27 ≪ densonly 0.91).

**Gate (script, published-random ruler):** FAIL

- σ₀ profile rate: 0.2584 (near-zero: False) — eval isotonic; pre-inv neg dσ₀ = 0
- N² profile / level @ 1e-8: 0.1172 / 0.001361
- overall T RMSE: 0.5981 vs argo16 0.4158 (ratio 1.438); vs clean chrono 0.5367 (ratio 1.114)
- MLD RMSE: 34.883344722575885
- dρ/dz RMSE: 0.007441934624385057
- inversion fail frac: 0.0000

## T/S RMSE by depth band

| band | T RMSE | S RMSE | vs T1-A recon |
|------|--------|--------|---------------|
| 0-50 | 1.2342 | 0.3292 | 6.095 |
| 50-200 | 1.3972 | 0.1711 | 6.274 |
| 200-800 | 0.6547 | 0.0916 | 6.147 |
| >800 | 0.1347 | 0.0101 | — |

_Gate note:_ STOP uses σ₀ near-zero + overall T ≤ argo16×1.10 (trained separate-PCA). By-band vs T1-A is reconstruction floor (prediction>recon expected).

## Phase 2 caveat

T2 stale gate OPEN (0% SSS/SST/SSH on val/test). Full HDF5 lacks v3 error fields (`err_sla` / `analysis_error` / `sos_error` only in `*_err_smoke.h5`). density_spice cache has no `inputs_err` — headline metrics may be SSS-confounded only if stale returns; currently not. Formal product-error channels not in model inputs.

## Phase 2 caveat

T2 stale gate OPEN (0% SSS/SST/SSH on val/test). Full HDF5 lacks v3 error fields (`err_sla` / `analysis_error` / `sos_error` only in `*_err_smoke.h5`). density_spice cache has no `inputs_err` — headline metrics may be SSS-confounded only if stale returns; currently not. Formal product-error channels not in model inputs.
