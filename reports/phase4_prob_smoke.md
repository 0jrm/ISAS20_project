# Phase 4 — probabilistic head (smoke close-out)

**Date:** 2026-07-16  
**Branch:** `residual_cube`

## Acceptance

| Item | Result |
|------|--------|
| `prob_mode=crps` smoke (2 ep) | PASS |
| `prob_mode=nll` smoke (2 ep) | PASS |
| `prob_mode=quantile` smoke (2 ep) | PASS |
| Two-stage launcher (`scripts/train_prob_twostage.py`) | PASS (crps 1+1 ep) |
| `dacov` PSD + MC diagonal (≤15% @ 2000 draws) | PASS (`selfcheck.test_dacov_psd_and_mc`) |
| `scripts/uncertainty_decomposition.py` | PASS → `reports/uncertainty_decomposition.json` |

## What landed

- `PatchConvMLP(probabilistic=True)` → μ+σ or 9 non-crossing quantiles
- `DensitySpiceProbLoss` (`mse` / `crps` / `nll` / `quantile`) behind `loss_config.prob_mode`
- `dacov/`: spice Σ = V diag(σ²) Vᵀ; density-ctrl linearized cumsum map
- Error-channel concat flag on data loader (`use_error_channels`); target-err attenuation hook in loss
- Formal product-error caveat in decomposition output (relative indicators only)

## Caveats / follow-ups before Phase 5 matrix

1. Quantile mode predicts directly in standardized (σ₀_ctrl, spice) space — **hard softplus-a constraint is not applied during quantile training** (documented in loss). Prefer CRPS/NLL winners for the representation chapter.
2. Full HDF5 v3 error fields still not in the density_spice cache used here — 4.5–4.7 end-to-end needs a cache rebuild once SSS/v3 batches are regenerated (Phase 2 remainder).
3. T/S Σ via inversion Jacobian is stubbed as a follow-up note in `export_profile_covariance` (ctrl+spice Σ is the DA R-matrix seed).
4. R4 golden drift remains a **Phase 5 prerequisite**.

## Commands

```bash
cd NeSPReSO2_onTemplate
srun --ntasks=1 --cpus-per-task=8 --gres=gpu:1 conda run -n nespreso \
  python scripts/phase4_prob_smoke.py
srun --ntasks=1 --cpus-per-task=8 --gres=gpu:1 conda run -n nespreso \
  python scripts/train_prob_twostage.py -c config/argo/config_argo_densityspice_prob_smoke.json \
  --prob-mode crps --stage1-epochs 1 --stage2-epochs 1 --parent-tag smoke_twostage
```
