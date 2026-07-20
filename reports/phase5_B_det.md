# Phase 5 — B × det

## Protocol v2 — scored

**Status:** complete 2026-07-17  
**Schedule:** chrono argo early_stop 500 on val loss (joint EOF-32). Stop epochs: 1123 / 854 / 866.  
**Eval fix:** `eval_run.py` joint_eof path (decode T/S via destandardize; pass `joint_eof_meta` into loss).

### Cell mean±std (test T RMSE) — judgment unit

| | overall T RMSE | vs floor 0.5903 |
|--|---------------:|:---------------:|
| **mean±std** | **0.534±0.001** | **yes** (0.534 ≤ 0.590) |

Per-seed: 0.535 / 0.534 / 0.533 — slightly better than A×det (~0.541) and clean chrono argo16 (~0.537). **§3 det survivor.**

Artifacts: `saved/phase5_matrix/B_det_v2/`, `eval/p5_B_det_v2_s*.json`, `B_det_v2_summary.json`.
