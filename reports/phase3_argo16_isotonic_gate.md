# Phase 3 gate — argo16 + isotonic projection (§3.6 option 2)

**Verdict (coded vs published 0.416×1.10): FAIL**

Split: `chronological` n=623. Cache: `../data/cache/train_ready_651f62a4b596.pkl`.

> Published T=0.4158 (eval_argo16_test.json) used random split; ckpt config has no split_mode. Chronological test ≈0.514.

| stage | overall T | σ₀ |
|-------|-----------|-----|
| argo16 raw | 0.5143 | post-inv profile 0.4896 |
| + isotonic + re-invert | **0.5143** | **pre-inv 0.0000** / post tol0 0.3082 / tol1e-6 0.0000 |
| published floor (0.416×1.10) | 0.4574 | pre-inv <0.01 |
| self floor (raw×1.10) | 0.5657 | |

- Proj cost vs raw T: 0.0015 °C
- Post-inv level viol @ tol0: 0.001522 (O(1e-6) Newton noise; tol=1e-6 → profile rate 0)

σ₀ pre-inv: PASS. Skill vs published: FAIL (ratio 1.237). Skill vs self×1.10: PASS.

**Coded gate → FAIL.** Chronological argo16 (0.514) already exceeds the published 0.458 floor; isotonic cannot invent skill. It does deliver the stability half (pre-inv σ₀=0, tiny T cost). No merge to main. In-head priority: low-rank-δa PCA on increment anomalies, then month-clim. Loss already in σ₀ space (post softplus+cumsum).

