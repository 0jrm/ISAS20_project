# Finding — compress in physical space, constrain after

**Date:** 2026-07-16  
**Status:** citable representation-chapter finding (not a bug report)  
**Candidate arc:** Phase 3 low-rank δσ₀ recovery

## Claim

Linear PCA compression through the *preimage* of a nonlinear monotone parameterization
(softplus⁻¹ of density increments — "a-space") fails to represent physical σ₀ residuals,
even when a-space EVR looks healthy. The same rank in physical σ₀ space reconstructs
cleanly. **Compress in the physical space; apply constraints after.**

## Evidence (GoM ARGO chrono chronological test; K=64 ctrl grid)

| representation | R | σ₀ recon RMSE (test) | vs clim (0.722) | a/σ₀ EVR |
|----------------|---:|---------------------:|----------------:|---------:|
| a-space PCA on `(a − a_clim)` | 16 | **0.925** | worse than clim | a-EVR 0.96 |
| σ₀-space PCA on `(σ₀ − clim)` | 16 | **0.026** | ≪ clim | σ₀-EVR 0.999 |
| increment-space PCA | 16 | 0.119 | better than clim | — |

Ratio of recon errors (a-space / σ₀-space) at R=16: **≈ 35×**.

Full-rank softplus+cumsum encode/decode round-trip on the same profiles: σ₀ RMSE ~1e-6
(the parameterization itself is faithful; the *compression* site is the failure).

## Consequence for the plumbing branch

The `representation_plumbing` call (density signal extractable; argo16 ≪ densonly on
mse_σ) is vindicated by construction: moving the low-rank basis into σ₀ space recovered
density skill (mse_σ_z 0.27 vs densonly 0.91; pred dens + true spice → T 0.250) and,
after spice continue, cleared the corrected chrono skill gate (T 0.562 ≤ floor 0.590).

## What this is not

- Not a failure of softplus+cumsum as a *constraint* — that path remains valid for
  full-rank heads ("stable by construction in the head").
- Not a claim that a-space is always useless — only that *low-rank linear* models of
  a-space do not transfer to σ₀ under this nonlinearity.

## Pointers

- Ceiling study: session diagnostic 2026-07-16 (a vs σ₀ vs incr table).
- Failed train: `reports/phase3_lowrank_delta_a_eval.md` (a-space v1, T=0.830).
- Passing train: `reports/phase3_lowrank_sigma0_spice_eval.md` (σ₀-space + spice continue).
- PLAN: §3.2 low-rank note; §3.6 in-head priority (corrected).
