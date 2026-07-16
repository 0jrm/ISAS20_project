# Phase 3 — density-only ablation (λ_τ = 0)

**Date:** 2026-07-16  
**Config:** `config/argo/config_argo_densityspice_densonly.json`  
**Checkpoint:** `saved/argo_densityspice/models/NeSPReSO2_ARGO_GoM_densityspice_densonly/phase3_densonly_v1/model_best.pth`  
**Recipe:** λ_ρ=1, λ_τ=0, `density_weight_scale=1.0`, residual `a=a_clim+δa`, grad_clip 5, 100 epochs  
**Cache / split:** `train_ready_cd9e08b6c630.pkl`, chronological test n=623  

**Purpose:** multi-task interference test after blame-split showed density owns the v10 T gap.

## Headline

| Metric | densonly | v10 (joint, spice-first) |
|--------|----------|---------------------------|
| σ₀ profile rate | **0.000** | **0.000** |
| overall T RMSE | 1.675 (spice unconstrained — not a skill readout) | 0.724 |
| **pred σ₀ + true τ** (density skill) | **0.547** | **0.522** |
| true σ₀ + pred τ (spice skill) | 1.187 (garbage, expected) | 0.393 |

**Verdict:** density alone does **not** beat joint v10 on the fair density readout (pred σ₀ + true τ). Shared-trunk multi-task interference is **not** the smoking gun. Density remains near the clim floor on chrono test even with λ_τ=0; val mse_σ≈0.43 vs test≈0.91 (chrono shift). Next: sequential (freeze v10 spice → train density) or EMA-normalized joint, not another λ_τ=0 retry. §3.6 option 2 still the pre-registered floor if those fail.

## Physical (full pred; spice untrained)

- N² profile / level @ 1e-8: 0.000 / 0.000
- MLD RMSE: 43.33 m
- dρ/dz RMSE: 0.00843
- Gate overall T: **FAIL** (ratio 4.03× argo16) — expected with λ_τ=0

### T/S by band (full pred)

| band | T RMSE | S RMSE |
|------|--------|--------|
| 0-50 | 1.564 | 0.463 |
| 50-200 | 3.374 | 0.319 |
| 200-800 | 2.287 | 0.351 |
| >800 | 0.293 | 0.013 |

## Latent (test)

- mse_σ (std ctrl): ≈0.91 (val ≈0.43)
- mse_τ: ≈191 (untrained)
- |δa| mean: ≈0.09 (moved off zero, but not enough for skill)

## Blame-split detail

See [`phase3_densonly_blame_swap.md`](phase3_densonly_blame_swap.md). Dominant branch under densonly is **spice** (because τ is noise); fair comparison to v10 uses the density-error row only.
