# Phase 3 — density shift diagnostics (eval-only)

No retraining. Checkpoints: densonly v1, v10, argo16_scales. Same chronological split.

## 1. Climatology-only baseline (task hardness)

| era | clim mse_σ | densonly mse_σ | v10 mse_σ | argo16 mse_σ |
|-----|------------|----------------|-----------|--------------|
| val | 1.1386 | 0.4309 | 1.1602 | 0.1463 |
| test | 1.2590 | 0.9133 | 1.7424 | 0.2342 |

- clim test/val ratio: **1.106** (≫1 ⇒ targets drift from train clim; ~1.1 here ⇒ clim hardness alone is not the 2× densonly jump)
- densonly test/val: **2.120**; argo16 test/val: **1.600**
- std-anomaly var test/val: **1.533**

## 2. argo16 control (is the signal in the inputs?)

- argo16 test mse_σ **0.2342** vs densonly **0.9133** → argo16 beats densonly on density.
- Absolute: argo16 val=0.1463 / test=0.2342; densonly val=0.4309 / test=0.9133.
- **Verdict branch:** argo16 density extrapolates far better → signal is in the inputs; monotone / clim-residual plumbing is failing to use it (not a pure informational ceiling).

## 3. Shrinkage  var(σ̂₀ − σ₀_clim) / var(σ₀_true − σ₀_clim)  [σ₀ space]

Prior a-space shrink≈0 contradicted densonly beating clim on val (0.43 vs 1.14) — 
softplus+cumsum Jacobian makes a-space variance ratios uninterpretable. Use σ₀ space.

| era | densonly σ₀-shrink | v10 σ₀-shrink | a-space densonly (do not interpret) |
|-----|--------------------|---------------|--------------------------------------|
| val | 0.318 | 0.133 | 0.0002 |
| test | 0.275 | 0.072 | 0.0002 |

Val densonly σ₀-anom RMSE 0.3647 vs clim 0.8044 (must beat clim if shrink≪1 is false).
argo16 test/val density ratio **1.60** is genuine era shift that hits everyone — plumbing fixes should not be judged against a flat-ratio standard.

Note: `DensitySpiceLoss` already evaluates MSE **post** softplus+cumsum (σ₀ space).

## 4. Test density error vs calendar month

| YYYYMM | n | clim mse | densonly mse | argo16 mse |
|--------|---|----------|--------------|------------|
| 202105 | 52 | 0.5613 | 0.4339 | 0.2054 |
| 202106 | 100 | 1.0620 | 0.8751 | 0.2330 |
| 202107 | 85 | 1.7736 | 1.3509 | 0.4388 |
| 202108 | 68 | 2.0621 | 1.3229 | 0.2644 |
| 202109 | 77 | 1.4281 | 1.0916 | 0.2257 |
| 202110 | 53 | 1.0723 | 0.8671 | 0.1906 |
| 202111 | 51 | 0.8896 | 0.6703 | 0.1359 |
| 202112 | 48 | 1.1546 | 0.7810 | 0.1547 |
| 202201 | 49 | 1.0235 | 0.6010 | 0.1412 |
| 202202 | 40 | 1.0057 | 0.5752 | 0.1964 |

Monotone growth with distance from train era ⇒ nonstationarity fingerprint; flat-then-jump ⇒ input-quality regime (cross-check SSS window).

## Decision (pre-registered)

{
  "clim_test_over_val": 1.1056844394630287,
  "argo16_test_over_val": 1.6002509651410655,
  "densonly_test_over_val": 2.1196537184326045,
  "argo16_beats_densonly_on_test": true,
  "branch": "representation_plumbing",
  "read": "argo16 density \u226a densonly on test and degrades less \u2192 signal is extractable; suspect clim-residual / softplus / month-clim / SSH\u2192density path. Keep v10 spice frozen. \u00a73.6 opt-2 still floor."
}

Keep **v10 spice frozen** as an asset either way (blame-split: true σ₀+pred τ = 0.393).
§3.6 option 2 (isotonic at inference) remains the floor if plumbing + Phase 2 still fail skill.

## Process

- v3 HDF5 regen: confirm resumable progress (batches advancing, not looping).
- Do **not** merge to main until a phase gate passes.

