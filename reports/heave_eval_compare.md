# Heave-residual vs A×CRPS (chrono test, n=623)

Same cache `data/cache/train_ready_3adcff404b0b.pkl`. evalphys 1.2.0. Heave decode is warped residual + clim, not z-PCA.

**Checkpoints**

- **A_CRPS**: Phase 5 winner `p5_A_CRPS_v2_s42_s2/model_best.pth` (z-PCA T/S-16, CRPS).
- **Heave_best**: `heave_residual_s42b/model_best.pth` (val 2.388). s42c never wrote `model_best.pth`.
- **Heave_last**: `heave_residual_s42c/checkpoint.pth` (early-stop epoch 869).

Full-column RMSE is `eval_run.py` (heave path uses `HeaveResidualLoss.physical_ts`). Bands / D20 / D26 / N² are `scripts/thermocline_scorecard.py`. σ_D26 is `scripts/export_heave_tsis.py`.

## Headline

Heave-residual **does not beat A×CRPS** on the metrics this sprint cares about. D26 is a tie. 50–200 m T is slightly worse. Surface T and MLD are much worse. Almost every heave profile violates N². Do not promote; no TSIS insertion.

## Test-split table

| metric | A_CRPS | Heave_best | Heave_last |
|---|---:|---:|---:|
| T RMSE (0–1800 m) | **0.562** | 0.605 | 0.598 |
| S RMSE | **0.091** | 0.093 | 0.094 |
| T 0–50 m | **1.147** | 1.811 | 1.704 |
| T 50–200 m | **1.215** | 1.248 | 1.241 |
| T 200–800 m | 0.658 | **0.636** | 0.644 |
| T >800 m | 0.147 | **0.135** | 0.138 |
| D20 RMSE (m) | 20.05 | **18.72** | 18.75 |
| D26 RMSE (m) | 19.05 | 19.16 | **18.98** |
| MLD RMSE (m) | **36.8** | 51.2 | 50.7 |
| max-N² depth RMSE (m) | 46.7 | **43.9** | 44.6 |
| heave fraction (50–200 T) | 0.18 | 0.00 | 0.08 |
| N² profile viol. @ 1e-8 | **0.385** | 0.998 | 0.992 |
| steric vs *true* (cm RMS) | 7.65 | 7.15 | **6.85** |
| ENCE(σ_D26) | — | 0.546 (fail) | — |
| ENCE(σ_D26) JJA | — | 0.481 (fail) | — |

A×CRPS physical-space ENCE(T)=0.236 already failed the 0.20 gate (`reports/phase5_A_CRPS_physical_strata.md`). Heave σ_D26 fails it harder.

## Why heave lost

1. **MLD collapsed.** Export `mld_m` mean is **10.0 m** on all 623 test casts — the low end of `50 + 40 tanh(·)`. The warp is not predicting mixed-layer depth; it is saturating the logit. MLD RMSE 51 m vs 37 m for A×CRPS follows from that.

2. **Heave is not the leftover error.** Aligning A×CRPS to true D26 only removes 18% of 50–200 m T RMSE². Heave_best heave-fraction is **0**: shifting by predicted D26 does not help. The 50–200 m error is shape, which is what z-PCA-16 already represents well (T1 ceiling **0.116 °C**).

3. **Warp-clim is a worse T1 ceiling than PCA-16.** Truth through warp-clim with *true* MLD+D26: 3.44 °C / D26 61 m. Truth through PCA-16: 0.116 °C / D26 8.3 m. Landmark registration of a climatology is not a tighter representation on this cache. GEM/SLA (1.40 °C) is also worse than A×CRPS (1.22 °C) in 50–200 m.

4. **Unstable columns.** N² violations on 99.8% of heave profiles (level rate still small, ~0.5%) vs 38.5% for A×CRPS. σ₀ monotonicity matches. The unwarp of a 10 m MLD + residual PCs is manufacturing inversions.

5. **σ_D26 is not calibrated.** ENCE 0.55 (gate 0.20). Mean σ_D26 = 12.4 m vs D26 RMSE ~19 m. JJA Spearman is undefined (degenerate pairing). Diagonal R in (η, residual) space is not ready.

## Steric / LC gate

`steric_vs_adt` LC RMS ~1.8e5 cm is **not a real number**: this cache has no `ssh_obs_sla`, so the scorecard fell back to a raw input column. Ignore it.

`summarize_physical` steric-vs-truth is usable: ~7 cm RMS, heave slightly better. That is **not** the 2 cm LC-vs-ADT ship gate. Gate status: **not evaluable** on this cache → no promotion.

## Training note

s42c early-stopped at epoch 869 (`not_improved=501`) without beating s42b's val 2.388. Last weights are not a hidden winner (T 0.598 vs 0.605; still far from 0.562).

## Verdict

Keep A×CRPS as the native-1 m column. Heave-residual as trained here is a **negative** on the thermocline job: it does not take D26 from SLA, it wrecks the mixed layer, and it fails stability and σ_D26 calibration. Next step if this architecture is retried: stop MLD from saturating (different parameterization or a D26-only warp), and only then re-score 50–200 m T / N² / ENCE(σ_D26).
