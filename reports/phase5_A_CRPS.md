# Phase 5 — A × CRPS

## Protocol v2 (val-ENCE stage-2 stop) — scored

**Status:** complete 2026-07-17  
**Head:** `PCAHeteroLoss` on separate T/S PCA-16 (PC space)  
**Test-eval counter:** one consultation per seed.

### Cell mean±std (test, after val per_dim α) — judgment unit

| | test CRPS (PC space) | test ENCE | Spearman |
|--|---------------------:|----------:|---------:|
| **mean±std** | **1.237±0.010** | **0.053±0.004** | **0.728±0.002** |
| clears ENCE&lt;0.20 (on **mean**) | | **yes** (0.053 &lt; 0.20) | |

Per-seed:

| seed | val ENCE | test CRPS | test ENCE | Spearman |
|------|---------:|----------:|----------:|---------:|
| 42 | 0.022 | 1.245 | 0.054 | 0.729 |
| 43 | 0.009 | 1.225 | 0.049 | 0.726 |
| 44 | 0.022 | 1.241 | 0.057 | 0.727 |

**Caveat:** CRPS is in **PC space**, not density/spice space — do not rank against C×CRPS/C×NLL CRPS numbers. ENCE gate is still valid (same definition on predictive σ). Survivor candidate; final §3 pick needs like-for-like CRPS or a pre-registered cross-rep rule.

Artifacts: `saved/phase5_matrix/A_CRPS_v2/`, `eval/p5_A_CRPS_v2_s*_ence.*`, `A_CRPS_v2_summary.json`.
