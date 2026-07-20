# Phase 5 — B × CRPS

## Protocol v2 — scored

**Status:** complete 2026-07-17  
**Head:** joint EOF-32 + `PCAHeteroLoss` (PC space)

### Cell mean±std (test, after val per_dim α)

| | test CRPS (PC) | test ENCE | Spearman |
|--|---------------:|----------:|---------:|
| **mean±std** | **2.761±0.028** | **0.069±0.004** | **0.652±0.001** |
| clears ENCE&lt;0.20 | | **yes** | |

PC-space CRPS (joint dim) — not comparable to A or C. Survivor on ENCE.

Artifacts: `saved/phase5_matrix/B_CRPS_v2/`, `eval/B_CRPS_v2_summary.json`.
