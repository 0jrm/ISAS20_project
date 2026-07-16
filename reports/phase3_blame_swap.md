# Phase 3 — blame-split swap test

**Checkpoint:** `saved/argo_densityspice/models/NeSPReSO2_ARGO_GoM_densityspice/phase3_full_v10/model_best.pth`
**Test n:** 623

| Reconstruction | overall T RMSE | vs argo16×1.10 |
|----------------|----------------|----------------|
| pred σ₀ + pred τ (full) | 0.7236 | 1.58× |
| **true σ₀ + pred τ** (spice error) | 0.3928 | 0.86× |
| **pred σ₀ + true τ** (density error) | 0.5221 | 1.14× |
| gate floor (argo16×1.10) | 0.4574 | 1.00× |

**Dominant branch:** `density` (higher T RMSE when that branch is predicted and the other is truth).

Read: if density-error row ≫ spice-error row, the 0.72→0.46 gap lives in σ₀; decouple / density-only next. If the reverse, spice still owns skill.

