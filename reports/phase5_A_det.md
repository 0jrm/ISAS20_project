# Phase 5 — A × det

## Protocol v2 — scored

**Status:** complete 2026-07-17  
**Schedule:** chrono argo16 recipe (early_stop 500 on val loss). Stop epochs: 814 / 1043 / 778.

### Cell mean±std (test T RMSE) — judgment unit

| | overall T RMSE | vs floor 0.5903 |
|--|---------------:|:---------------:|
| **mean±std** | **0.541±0.004** | **yes** (0.541 ≤ 0.590) |

Per-seed: 0.537 / 0.544 / 0.542 — matches clean chrono argo16 (~0.537). **§3 det survivor.**

Artifacts: `saved/phase5_matrix/A_det_v2/`, `eval/p5_A_det_v2_s*.json`, `A_det_v2_summary.json`.
