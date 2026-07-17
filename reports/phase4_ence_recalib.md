# Phase 4 — ENCE recovery via val-only σ recalibration

**Checkpoint:** `saved/argo_densityspice_lowrank_crps/models/NeSPReSO2_ARGO_GoM_densityspice_lowrank_crps_lowrank_crps_v1_s2/lowrank_crps_v1_s2/model_best.pth`  
**Cache:** `../data/cache/train_ready_0f6129b27ddb.pkl`  
**Fit split:** val n=621

## Val recipes (fit + score on val)

| recipe | CRPS | ENCE | slope | Spearman | α mean [min,max] | ENCE pass |
|--------|------|------|-------|----------|------------------|-----------|
| none | 0.5846 | 0.2858 | 1.508 | 0.4411 | 1.000 [1.00,1.00] | NO |
| global | 0.5867 | 0.2465 | 1.330 | 0.4411 | 1.134 [1.13,1.13] | NO |
| **per_dim** | 0.5767 | 0.0269 | 1.348 | 0.5807 | 1.231 [0.61,3.51] | yes |
| depth_band | 0.5798 | 0.1113 | 1.342 | 0.5204 | 1.160 [0.75,2.20] | yes |

**Best (val ENCE):** `per_dim` — PASS (ENCE < 0.2)

### Depth-band α (density ctrl)

| band | α |
|------|---|
| 0-50 | 0.949 |
| 50-200 | 2.199 |
| 200-800 | 0.750 |
| >800 | 1.143 |

## Test (val-passing recipes)

| recipe | CRPS | ENCE | slope | Spearman | ENCE pass |
|--------|------|------|-------|----------|-----------|
| per_dim | 0.7108 | **0.2310** | 1.591 | 0.6154 | NO |
| depth_band | 0.7126 | 0.2930 | 1.586 | 0.5854 | NO |

**Test-eval counter:** this s2 test consultation is **#1** of two Phase 4 test peeks (see `phase4_ence_recalib_s2b.md` for #2 after s2b). Mild iteration toward the gate — recorded for hygiene.

vs prior global-α test ENCE **0.361**. Per-dim closes most of the gap but still misses 0.20 by ~0.03.

**Diagnosis:** 50–200 m band needs ~2.2× on val (thermocline under-dispersion). Per-dim α span [0.61, 3.51] overfits val→test transfer.

**Next lever:** longer stage-2 (train-time σ), then re-fit per_dim on val and one new test score.

**Note:** Scales fitted on val only. Spearman invariant to positive σ scale.
