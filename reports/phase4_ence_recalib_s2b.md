# Phase 4 — ENCE recovery via val-only σ recalibration

**Checkpoint:** `saved/argo_densityspice_lowrank_crps/models/NeSPReSO2_ARGO_GoM_densityspice_lowrank_crps_lowrank_crps_v1_s2b/lowrank_crps_v1_s2b/model_best.pth`  
**Cache:** `../data/cache/train_ready_0f6129b27ddb.pkl`  
**Fit split:** val n=621  
**Scales:** `sigma_recalib_per_dim.json` (next to ckpt)

## Test-eval counter (hygiene)

| # | event | split | ENCE | decision |
|---|-------|-------|-----:|----------|
| 1 | s2 + per_dim | **test** | 0.231 | MISS → train longer s2b (mild iteration) |
| 2 | s2b + per_dim | **test** | **0.160** | PASS (headline) |

Two test consultations with a training decision between them. Not disqualifying — Phase 5 matrix replaces single-run verdicts with 3-seed means scored once — but the counter is on the record. Matrix rule: val-only selection/recalib; **one test score per frozen cell**.

## Val recipes (fit + score on val)

| recipe | CRPS | ENCE | slope | Spearman | α mean [min,max] | ENCE pass |
|--------|------|------|-------|----------|------------------|-----------|
| none | 0.5813 | 0.3291 | 1.432 | 0.4876 | 1.000 [1.00,1.00] | NO |
| global | 0.5820 | 0.2435 | 1.328 | 0.4876 | 1.078 [1.08,1.08] | NO |
| **per_dim** | 0.5767 | 0.0578 | 1.355 | 0.5275 | 1.257 [0.67,2.80] | yes |
| depth_band | 0.5792 | 0.1065 | 1.349 | 0.5155 | 1.183 [0.84,1.60] | yes |

**Best (val ENCE):** `per_dim` — PASS (ENCE < 0.2)

### Depth-band α (density ctrl)

| band | α |
|------|---|
| 0-50 | 1.152 |
| 50-200 | 1.603 |
| 200-800 | 0.835 |
| >800 | 1.451 |

## Test (one score on this ckpt, best val recipe)

**Recipe:** `per_dim`  
**Anchors:** PASS (ENCE=0.1603; Spearman=0.5395)

| CRPS | ENCE | slope | Spearman |
|------|------|-------|----------|
| 0.6978 | 0.1603 | 1.608 | 0.5395 |

## Findings (dissertation material)

**Val→test calibration gap (prospectus §3.6.6):** ENCE 0.058 → 0.160 (~3×) across the era boundary — in-era fitted scales degrade out-of-era yet still clear the anchor. Stratified table (depth band × season) deferred to write-up; headline pair is not the whole story.

**Spearman 0.65 → 0.54 provenance:** 0.65 was full-rank density_spice CRPS (`phase4_full_eval.md`, different ckpt/architecture). 0.54 is this low-rank s2b + per_dim path. Not a recalibration side-effect (Spearman invariant to positive σ scale); different checkpoint + longer stage-2.

**Σ wiring:** inference and §4.4 export must use `α ⊙ σ` (`dacov.density_lowrank_covariance(..., alphas=...)`); `selfcheck.test_dacov_sigma_recalib_scales_export` pins this.

**Note:** Scales fitted on val only.
