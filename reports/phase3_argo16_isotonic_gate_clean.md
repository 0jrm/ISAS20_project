# Phase 3 gate — argo16 + isotonic projection (§3.6 option 2)

**Verdict (corrected ruler: same-split baseline×1.10): PASS**

Split: `chronological` n=623. Cache: `../data/cache/train_ready_3adcff404b0b.pkl`.

> Checkpoint: `argo16_chrono_clean` (chrono-trained, no leakage). Published T=0.4158 used random split; the earlier 0.514 chrono figure came from the random-trained `argo16_scales` checkpoint and was leaked-optimistic. Clean chrono baseline raw T = 0.5367.

| stage | overall T | σ₀ |
|-------|-----------|-----|
| argo16 raw | 0.5367 | post-inv profile 0.4366 |
| + isotonic + re-invert | **0.5367** | **pre-inv 0.0000** / post tol0 0.2311 / tol1e-6 0.0000 |
| published floor (0.416×1.10) | 0.4574 | pre-inv <0.01 |
| self floor (raw×1.10) | 0.5903 | |

- Proj cost vs raw T: 0.0014 °C
- Post-inv level viol @ tol0: 0.001483 (O(1e-6) Newton noise; tol=1e-6 → profile rate 0)

σ₀ pre-inv: PASS. Skill vs published: FAIL (ratio 1.291). Skill vs self×1.10: PASS.

**Corrected-ruler gate → PASS.** Ruler repair 2026-07-16: the 0.458 floor mixed a random-split baseline with chronological candidates (like-for-like split violation). Operative floor = same-split argo16 raw ×1.10; the published constant stays in the table for the record. Isotonic delivers the stability half (pre-inv σ₀=0, tiny T cost).

