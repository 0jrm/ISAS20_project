# Phase 3 — full density_spice train+eval

**Date:** 2026-07-16  
**Checkpoint:** `saved/argo_densityspice/models/NeSPReSO2_ARGO_GoM_densityspice/phase3_full_v10/model_best.pth`  
**Cache:** `../data/cache/train_ready_cd9e08b6c630.pkl` (n=4145, no `max_samples`)  
**Split:** chronological test n=623  

**Gate:** **FAIL** (skill) — **σ₀ PASS**

## STOP decision (session A)

| Criterion | Result |
|-----------|--------|
| σ₀ profile rate near-zero | **PASS** — 0.0000 |
| Upper-ocean T RMSE ≤ argo16×1.10 (trained separate-PCA overall 0.4158) | **FAIL** — overall T=0.7236 (ratio 1.74) |

Per session instructions: **STOP further headline claims / Phase 5** until skill recovers. Phase 4 CRPS runs below are informational only.

## Physical metrics (frozen evalphys)

- σ₀ profile rate: **0.0000**
- N² profile / level @ 1e-8: **0.0000 / 0.000000**
- overall T RMSE: 0.7236 vs argo16 0.4158 (ratio 1.740)
- MLD RMSE: 42.72 m
- dρ/dz RMSE: 0.00837
- inversion fail frac: 0.0000
- latent: mse_tau≈60 (matches Ridge spice baseline), mse_rho≈1.3 (near clim; density head under-trained)

### T/S RMSE by depth band

| band | T RMSE | S RMSE | vs T1-A recon |
|------|--------|--------|---------------|
| 0-50 | 1.5575 | 0.5509 | 7.691 |
| 50-200 | 1.8699 | 0.5720 | 8.397 |
| 200-800 | 0.6659 | 0.1420 | 6.253 |
| >800 | 0.1765 | 0.0251 | — |

## Training recipe that worked (v10)

Bugs found and fixed this session:

1. **`lambda_rho=8e-6`** from zero-`a` loss-scale derivation starved density (softplus decode of a=0 ⇒ σ₀≈0). Fixed derive to use clim-`a` init; then found spice PCs need **λ_τ≫λ_ρ** early (`0.05/1.0` spice-first) because PC variance ≫1.
2. **Random `a` explodes** softplus/cumsum → residual parameterization `a = a_clim + δa` in `DensitySpiceLoss`.
3. **`model_best.pth` only saved on `epoch % save_period`** → missed best epoch. Fixed in `base_trainer.py`.
4. Density weight×0.01 keeps softplus stable but **blocks density skill** on test (clim floor). Density-only fine-tune improved val ρ but not test (chrono shift).

## Discarded runs (negative results — keep)

- **v8** (`phase3_full_v8`): residual-δa + density weight×0.01, equal-ish λ; completed 100 epochs (val≈83). Spice still stuck (~mse_τ ≈ 180); did not unlock PC learning — discarded for skill, kept as pre–spice-first baseline.
- **v9** (`phase3_full_v9`): full-batch + equal λ_ρ/λ_τ=1; completed 200 epochs (val≈77–79). Same spice stall (~mse_τ ≈ 180); equal λ alone is not enough when PC variance ≫1 — discarded for skill.
- **v10s2e** (`phase3_full_v10s2e`): density fine-tune from v10 with **weight amplify** (undo ×0.01 on μ density rows). Epoch-1 loss ~5×10⁷ / val ~2×10⁶; early-stopped at 67 with no recovery. Pathology: amplifying density weights after residual+×0.01 init blows the fine-tune — comparative-architecture material, not a retry candidate.
- **densonly v1** (`phase3_densonly_v1`, λ_τ=0): fair density skill (pred σ₀+true τ) **0.547 vs v10 0.522** — no improvement. Multi-task interference **refuted**; density still fails chrono test alone. See [`phase3_densonly_eval.md`](phase3_densonly_eval.md).

## Diagnosis (loss whack-a-mole → revised)

Spice unlocked only when v10 set λ_ρ/λ_τ = 0.05/1 (spice-first). Combined with `a = a_clim + δa` and density weight×0.01, joint density sat at clim. Blame-split confirmed density owns the T gap. **But densonly (λ_τ=0) did not fix density** — so the fix is not “remove spice competition”; it is density capacity / chrono generalization / sequential warm-start from v10 spice, or EMA-normalized joint. Prefer procedures over representation-specific magic numbers (Phase 5 fairness). §3.6 option 2 remains the floor.

## Phase 2 caveat

- **T2 stale gate:** OPEN (0% SSS/SST/SSH on val/test) — not embargoed.
- **v3 error fields:** present only in `satellite_NeSPReSO_v2_ARGO_GoM_err_smoke.h5` (16 stations). Full HDF5 lacks `err_sla` / `analysis_error` / `sos_error`. density_spice cache has **no `inputs_err`**.
- **Blocker:** full SSS/v3 batch regen + cache rebuild before `use_error_channels=true` (regen launched; see HANDOFF).
- **Caveat:** test metrics may be SSS-confounded only if stale returns; currently T2 clean. Formal product-error channels not in model inputs.

## Next (before Phase 5)

1. Blame-split swap test (true σ₀ + pred τ vs pred σ₀ + true τ) — cheap, before retrain.
2. Density-only ablation (λ_τ=0) then decoupled/sequential retrain (EMA-normalized losses preferred).
3. Re-gate overall T ≤ argo16×1.10 with σ₀~0; if still FAIL → §3.6 fallback option 2 (isotonic projection at inference).
4. R4 golden remains Phase 5 prerequisite.
