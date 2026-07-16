# Gate floor provenance — corrected chronological ruler

**Date:** 2026-07-16  
**Purpose:** dissertation-defensible chain for the Phase 3 skill gate constant.

## Constants (side by side)

| constant | value | derivation |
|----------|------:|------------|
| published argo16 T (random split) | 0.4158 | `saved/eval_argo16_test.json`; `argo16_scales` has **no `split_mode`** → random |
| leaked chrono eval of same ckpt | 0.514 | train set overlaps 2021–2022 test era |
| **clean chrono argo16 raw T** | **0.5367** | `argo16_chrono_clean` (same config + `split_mode: chronological`, early stop ep 814) |
| published-random floor (×1.10) | 0.4574 | **do not use** for chrono candidates (like-for-like violation) |
| **corrected same-split floor** | **0.5903** | clean chrono raw × 1.10 |

Gate intent (restated): *within 10% of the argo16 baseline on the **same** split*.

## Errata / clean re-runs (must stay linked)

1. **Leakage erratum** on [`phase3_density_shift_diag.md`](phase3_density_shift_diag.md): argo16
   mse_σ test 0.21 was leaked-optimistic.
2. **Clean density diag** [`phase3_density_shift_diag_clean.md`](phase3_density_shift_diag_clean.md):
   clean argo16 mse_σ test **0.234** vs densonly 0.913 → `representation_plumbing` **survives**.
3. **Clean isotonic gate** [`phase3_argo16_isotonic_gate_clean.md`](phase3_argo16_isotonic_gate_clean.md):
   raw T 0.5367, +isotonic T 0.5367, pre-inv σ₀=0, proj cost ~0.0014 °C → opt-2 PASS on
   corrected floor (stability half); skill half is the low-rank candidate below.
4. PLAN-v2-recovery changelog 2026-07-16 "Ruler repair + leakage erratum".

## Skill-gate candidate (low-rank δσ₀)

See [`phase3_lowrank_sigma0_spice_eval.md`](phase3_lowrank_sigma0_spice_eval.md).
Pass on corrected floor is **admission to the Phase 5 matrix**, not the dissertation
headline number (single seed; test-eval iteration logged there).
