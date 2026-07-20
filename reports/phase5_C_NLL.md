# Phase 5 — C × NLL

## Protocol v1 — aborted

Mid-stage-2 under val-loss early-stop; not scored. Partial artifacts under `saved/runs/phase5_matrix/protocol_v1_aborted/`. Not a dissertation row.

---

## Protocol v2 (val-ENCE stage-2 stop) — scored

**Status:** complete 2026-07-17  
**Label:** `protocol v2 (val-ENCE early-stop, patience 40)`  
**Test-eval counter:** first consultation per seed (v1 aborted).  
**Stage-2 stop epochs:** s42→218, s43→173, s44→99 (train-time raw `mnt_best` ENCE ≈ 0.043 / 0.042 / 0.045).

### Cell mean±std (test, after val-selected α) — judgment unit

| | test CRPS | test ENCE | Spearman |
|--|----------:|----------:|---------:|
| **mean±std** | **0.774±0.036** | **0.120±0.021** | **0.556±0.062** |
| clears ENCE&lt;0.20 (on **mean**) | | **yes** (0.120 &lt; 0.20) | |

Per-seed detail (not used for pass/fail):

| seed | recipe | val ENCE | test CRPS | test ENCE | Spearman |
|------|--------|---------:|----------:|----------:|---------:|
| 42 | none | 0.043 | 0.795 | **0.107** | 0.487 |
| 43 | global | 0.035 | 0.733 | **0.108** | 0.572 |
| 44 | none | 0.045 | 0.794 | **0.144** | 0.608 |

**Reading:** first probabilistic cell to clear the ENCE gate under protocol v2. NLL head arrives near-calibrated (best val recipe often `none`/`global`, not per_dim). CRPS is worse than C×CRPS (~0.70–0.74) — mean sharpness trades for calibration. Continues as a §3 survivor candidate; final winner waits for remaining cells (det / A / B).

Artifacts: `saved/phase5_matrix/C_NLL_v2/`, `saved/runs/phase5_matrix/eval/p5_C_NLL_v2_s*_ence.*`, `C_NLL_v2_summary.json`, per-ckpt `sigma_recalib_per_dim.json`.
