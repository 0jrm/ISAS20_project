# Phase 3 — blame-split swap test

**Checkpoint:** `saved/argo_densityspice/models/NeSPReSO2_ARGO_GoM_densityspice_densonly/phase3_densonly_v1/model_best.pth`
**Test n:** 623

| Reconstruction | overall T RMSE | vs argo16×1.10 |
|----------------|----------------|----------------|
| pred σ₀ + pred τ (full) | 1.6749 | 3.66× |
| **true σ₀ + pred τ** (spice error) | 1.1868 | 2.59× |
| **pred σ₀ + true τ** (density error) | 0.5468 | 1.20× |
| gate floor (argo16×1.10) | 0.4574 | 1.00× |

**Dominant branch:** `spice` (higher T RMSE when that branch is predicted and the other is truth).

Read: if density-error row ≫ spice-error row, the 0.72→0.46 gap lives in σ₀; decouple / density-only next. If the reverse, spice still owns skill.

