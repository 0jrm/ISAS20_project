# Phase 4 — low-rank δσ₀ two-stage CRPS (4.8)

**Checkpoint:** `saved/argo_densityspice_lowrank_crps/models/NeSPReSO2_ARGO_GoM_densityspice_lowrank_crps_lowrank_crps_v1_s2/lowrank_crps_v1_s2/model_best.pth`  
**Cache:** `../data/cache/train_ready_0f6129b27ddb.pkl`  
**Split:** test n=623 (one score); α fitted on val only

**Val-only σ recalibration:** α = RMSE/RMV = **1.1336**

**Anchors (test, calibrated):** MISS (ENCE < 0.20: NO; Spearman ≫ 0.12: yes)

## Calibration (standardized σ₀_ctrl + spice PCs)

| split | recipe | CRPS | ENCE | slope | Spearman |
|-------|--------|------|------|-------|----------|
| val | raw | 0.5846 | 0.2858 | 1.508 | 0.4411 |
| val | σ×α | 0.5867 | 0.2465 | 1.330 | 0.4411 |
| test | raw | 0.7145 | 0.5057 | 1.764 | 0.5189 |
| test | σ×α (headline) | 0.7152 | 0.3611 | 1.556 | 0.5189 |

## Physical (point μ after inversion, test)

- σ₀ profile rate: 0.2857
- N² profile rate: 0.0594
- T RMSE overall: {'0-50': 1.10872326477201, '50-200': 1.17715932962333, '200-800': 0.6474038580157394, '>800': 0.13836299517067788}
- MLD RMSE: 36.43771238759053
- dρ/dz RMSE: 0.0074450838438057665

**Cov export:** `Σ_ρ = V diag(σ_z²) Vᵀ` (score-domain σ; `dacov.density_lowrank_covariance`).

**Caveat:** No inputs_err / input-error tercile stratum (Phase 2.2 full HDF5 blocker). T2 stale gate OPEN. Formal product errors are relative indicators only.
