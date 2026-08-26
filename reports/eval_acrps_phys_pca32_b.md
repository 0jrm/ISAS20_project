# Physical A×CRPS 32+32, balanced — **A_CRPS_z32**

Same 32-PC cache as the first physical run (`train_ready_4ee013852d33.pkl`). Tmux: `acrps_phys_b`. Stage 1 early-stop ~epoch 600 (val loss); stage 2 resume ~235, stop 276 on **val ENCE(T)** (patience 40). Best stage-2 val ENCE(T) was the first logged point (0.139).

**Choices (this run vs phys v1):**

1. Val-only σ α: `none` / `global_var` (one α_T, one α_S) / `depth_band_var` (4 bands × T/S). Pick by val ENCE(T). No per-level α.
2. Equal T/S: `0.5 L_T + 0.5 L_S` (MSE `profile_scales` unused).
3. Stage-2 monitor = ENCE(T) only.
4. Physical CRPS/MSE = mean of four evalphys band-means (0–50, 50–200, 200–800, >800).
5. Keep 32+32 + diagonal V pushforward; add `0.1 ×` PC CRPS (stage 1: PC MSE).
6. Stage 1 uses the same equal/band physical MSE; μ LR × 0.1 in stage 2 unchanged.

Checkpoint: `NeSPReSO2_onTemplate/saved/acrps_phys_pca32_b/.../acrps_phys_pca32_b_s42_s2/model_best.pth`

## Test RMSE (raw Argo profiles, n=623)

| | T | S |
|--|--:|--:|
| A×CRPS s42 (16+16, PC CRPS) | 0.562 | **0.091** |
| phys v1 (profile_scales, concat ENCE) | 0.581 | 0.090 |
| phys balanced | **0.560** | 0.095 |

T mean is now slightly better than frozen A×CRPS s42. S is a bit worse (expected: we stopped starving T).

## Test analytic calibration (decoded μ, diag σ)

Val recipe **depth_band_var**. Headline gate is ENCE(T) < 0.20 (not concat T+S).

| | CRPS(T) | ENCE(T) | CRPS(S) | ENCE(S) |
|--|--------:|--------:|--------:|--------:|
| phys v1 raw | 0.215 | 0.289 | 0.030 | 0.226 |
| balanced raw σ | 0.203 | **0.164** | 0.030 | 1.13 |
| balanced val-α σ | 0.206 | **0.135** | 0.030 | **0.092** |

ENCE(T) raw **passes** 0.20 (0.164). Recalib **passes** (0.135). S raw ENCE is bad because stage-2 ignored S; band α on val repairs it (0.092) without changing μ.

A×CRPS published phys ENCE(T)=0.236 is ensemble/strata, not this analytic diagonal score — still, this run’s analytic ENCE(T) is the first of these heads under 0.20.

## ENCE(T) by depth band (test)

| band | raw | val-α |
|------|----:|------:|
| 0–50 m | 0.582 | 0.491 |
| 50–200 m | 0.634 | 0.415 |
| 200–800 m | 0.163 | 0.131 |
| >800 m | 0.114 | 0.105 |

Pooled ENCE(T) can pass while the surface and thermocline still miss 0.20. Same pattern as frozen A×CRPS strata. Do not ingest CRPS-σ in 0–200 m from this head without more work.

JSON: [`eval_acrps_phys_pca32_b_s42.json`](eval_acrps_phys_pca32_b_s42.json), [`eval_acrps_phys_pca32_b_cal.json`](eval_acrps_phys_pca32_b_cal.json).
