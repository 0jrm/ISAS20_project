# Phase 5 — B × NLL

## Protocol v2 — scored

**Status:** complete 2026-07-17  
**Head:** joint EOF-32 + `PCAHeteroLoss` (PC space)

### Cell mean±std (test, after val per_dim α)

| | test CRPS (PC) | test ENCE | Spearman |
|--|---------------:|----------:|---------:|
| **mean±std** | **2.754±0.008** | **0.082±0.012** | **0.651±0.003** |
| clears ENCE&lt;0.20 | | **yes** | |

Survivor on ENCE. PC-space CRPS not comparable to A/C.

Artifacts: `saved/phase5_matrix/B_NLL_v2/`, `eval/B_NLL_v2_summary.json`.
