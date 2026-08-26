# Physical-space A×CRPS, 32+32 PCs (seed 42)

Two-stage protocol v2 (`val_ence` stage-2 stop). Chronological ARGO test, n=623. Cache `train_ready_4ee013852d33.pkl` (32 PCs). Do not pair with the 16-PC A×CRPS cache.

**Checkpoint:** `NeSPReSO2_onTemplate/saved/acrps_phys_pca32/.../acrps_phys_pca32_s42_s2/model_best.pth`

Stage 1: physical MSE, early-stop epoch 780 (patience 500 on val loss). Stage 2: physical CRPS, resume epoch 280, stop epoch 324 (patience 40 on val ENCE).

## Test skill (mean path)

| | T RMSE | S RMSE |
|--|------:|------:|
| A×CRPS s42 (16+16, PC CRPS) | **0.562** | 0.091 |
| this run (32+32, physical CRPS) | 0.581 | **0.090** |

Raw profile RMSE vs cache `profiles` ([`eval_acrps_phys_pca32_s42.json`](eval_acrps_phys_pca32_s42.json) vs [`eval_A_CRPS.json`](eval_A_CRPS.json)). Rank bump is not a like-for-like head ablation.

## Test calibration (analytic Gaussian, decoded μ/σ)

Training metric: closed-form CRPS on `μ_x = μ_z @ V + mean`, `σ_x = sqrt((σ_z²) @ V²)`, vs reconstructed targets.

| | CRPS | ENCE |
|--|------:|------:|
| T+S (level-mean) | 0.122 | 0.206 |
| temperature | 0.215 | 0.289 |
| salinity | 0.030 | 0.226 |

Source: [`eval_acrps_phys_pca32_s42_cal.json`](eval_acrps_phys_pca32_s42_cal.json). ENCE gate 0.20: overall **miss** (0.206); T **miss**.

A×CRPS published physical numbers are **ensemble** decode (M=100), not this analytic diagonal pushforward: phys CRPS **0.119±0.001**, phys ENCE **0.153±0.007** (3-seed table); strata ENCE(T) **0.236**. Do not rank the 0.122 vs 0.119 as the same estimator.

## Takeaway

Physical CRPS training with 32 PCs did not beat A×CRPS s42 on test T RMSE (0.581 vs 0.562) and did not clear the 0.20 physical ENCE gate on the analytic metric the run optimized.
