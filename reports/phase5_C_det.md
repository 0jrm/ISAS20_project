# Phase 5 — C × det

## Protocol v2 — scored

**Status:** complete 2026-07-17  
**Schedule:** 150 epochs, early-stop 40 on val loss (deterministic; no σ branch).  
**Launcher fix:** det no longer inherits CRPS `freeze_sigma` / short stage-1 epochs.

### Cell mean±std (test T RMSE) — judgment unit

| | overall T RMSE | vs floor 0.5903 |
|--|---------------:|:---------------:|
| **mean±std** | **0.609±0.012** | **no** (0.609 &gt; 0.590) |

Per-seed:

| seed | T RMSE | ratio vs clean chrono | pre-inv σ₀ neg | post-inv σ₀ profile rate |
|------|-------:|---------------------:|---------------:|-------------------------:|
| 42 | 0.598 | 1.114 | 0 | 0.258 |
| 43 | 0.608 | 1.132 | 0 | 0.191 |
| 44 | 0.622 | 1.158 | 0 | 0.122 |

**Reading:** misses Phase 3 skill floor. Pre-inv monotone (isotonic). Not a §3 survivor. C×NLL remains the only ENCE-clearing prob cell so far.

Artifacts: `saved/phase5_matrix/C_det_v2/`, `saved/runs/phase5_matrix/eval/p5_C_det_v2_s*`, `C_det_v2_summary.json`.
