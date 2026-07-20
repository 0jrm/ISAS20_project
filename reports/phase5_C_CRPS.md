# Phase 5 — C × CRPS

## Protocol v1 (short stage-2 / val-loss early-stop) — retained evidence

**Status:** complete 2026-07-17 — **not** the dissertation headline cell  
**Label:** `protocol v1 (short stage-2)`  
**Why kept:** three-seed replication that under-trained stage-2 fails calibration (matches Phase 4 s2 test ENCE 0.231). UQ-chapter finding.  
**Test-eval counter:** one consultation per seed under v1; protocol v2 is the second per seed (logged).

### Cell mean±std (test, after val per_dim α) — judgment unit

| | test CRPS | test ENCE | Spearman |
|--|----------:|----------:|---------:|
| **mean±std** | **0.703±0.003** | **0.225±0.020** | **0.607±0.014** |
| clears ENCE&lt;0.20 (on **mean**) | | **no** (0.225 ≥ 0.20) | |

Per-seed detail (not used for pass/fail):

| seed | val ENCE | test CRPS | test ENCE | Spearman |
|------|----------|-----------|-----------|----------|
| 42 | 0.045 | 0.705 | 0.231 | 0.619 |
| 43 | 0.056 | 0.700 | 0.242 | 0.609 |
| 44 | 0.051 | 0.705 | 0.203 | 0.591 |

Note: CRPS 0.703 ≈ Phase 4 s2b 0.698; Spearman clears — mean is fine, variance schedule was immature under v1.

Artifacts: `saved/phase5_matrix/C_CRPS/`, `saved/runs/phase5_matrix/eval/p5_C_CRPS_s*_ence.*`, `C_CRPS_summary.json`.

---

## Protocol v2 (val-ENCE stage-2 stop) — scored

**Status:** complete 2026-07-17  
**Label:** `protocol v2 (val-ENCE early-stop, patience 40)`  
**Test-eval counter:** second consultation per seed (after v1).  
**Stage-2 stop epochs:** s42→196, s43→154, s44→93 (train-time raw `mnt_best` ENCE 0.119 / 0.101 / 0.156).

### Cell mean±std (test, after val per_dim α) — judgment unit

| | test CRPS | test ENCE | Spearman |
|--|----------:|----------:|---------:|
| **mean±std** | **0.742±0.048** | **0.248±0.021** | **0.632±0.019** |
| clears ENCE&lt;0.20 (on **mean**) | | **no** (0.248 ≥ 0.20) | |

Per-seed detail (not used for pass/fail):

| seed | val ENCE (post-α) | test CRPS | test ENCE | Spearman |
|------|------------------:|----------:|----------:|---------:|
| 42 | 0.024 | 0.702 | 0.240 | 0.634 |
| 43 | 0.031 | 0.728 | 0.233 | 0.650 |
| 44 | 0.025 | 0.795 | 0.272 | 0.612 |

**Reading (do not re-tune):** val ENCE after per-dim α is excellent (~0.02–0.03) but test ENCE stays in the Phase-4 short-s2 band (~0.23–0.27). Stopping stage-2 on *train-time raw* val ENCE did **not** recover Phase 4 s2b (test ENCE 0.160). CRPS mean also degraded vs v1 (0.742 vs 0.703), driven by early-stopped s44. Cell does not clear the 0.20 gate; §3.1 fallback remains in force for the full matrix.

Artifacts: `saved/phase5_matrix/C_CRPS_v2/`, `saved/runs/phase5_matrix/eval/p5_C_CRPS_v2_s*_ence.*`, `C_CRPS_v2_summary.json`, per-ckpt `sigma_recalib_per_dim.json`.
