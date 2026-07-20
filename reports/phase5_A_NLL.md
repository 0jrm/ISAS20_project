# Phase 5 — A × NLL

## Protocol v2 (val-ENCE stage-2 stop) — scored

**Status:** complete 2026-07-17  
**Head:** `PCAHeteroLoss` on separate T/S PCA-16 (PC space)

### Cell mean±std (test, after val per_dim α) — judgment unit

| | test CRPS (PC space) | test ENCE | Spearman |
|--|---------------------:|----------:|---------:|
| **mean±std** | **1.257±0.031** | **0.052±0.004** | **0.729±0.001** |
| clears ENCE&lt;0.20 (on **mean**) | | **yes** (0.052 &lt; 0.20) | |

Per-seed:

| seed | recipe | test CRPS | test ENCE | Spearman |
|------|--------|----------:|----------:|---------:|
| 42 | per_dim | 1.255 | 0.049 | 0.730 |
| 43 | per_dim | 1.289 | 0.056 | 0.729 |
| 44 | per_dim | 1.228 | 0.051 | 0.728 |

**Caveat:** PC-space CRPS — not comparable to C dens/spice CRPS. ENCE gate valid. Survivor.

Artifacts: `saved/phase5_matrix/A_NLL_v2/`, `eval/p5_A_NLL_v2_s*_ence.*`, `A_NLL_v2_summary.json`.
