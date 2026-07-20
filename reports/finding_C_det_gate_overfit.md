# Finding — C×det misses the skill floor under the fair 3-seed protocol

**Date:** 2026-07-17  
**Status:** locked finding (matrix evidence)

## Claim

Phase 3’s single-run low-rank δσ₀ + spice continue pass (**T 0.562 ≤ floor 0.590**, ~5% margin) **did not replicate** as matrix cell **C×det** under protocol v2: cell-mean test T RMSE **0.609±0.012** (seeds 0.598 / 0.608 / 0.622) — **MISS**.

## Why this matters

The Phase 3 pass admitted C to the matrix. It was **not** the final dissertation number. Riding a thin margin after iterating toward the gate is exactly the failure mode the test-eval counter and 3-seed cell-mean rule were built to catch. The matrix did its job: an over-fit gate pass did not survive fair replication.

## Retroactive framing

| artifact | role |
|----------|------|
| `phase3_lowrank_sigma0_spice_eval.md` (T 0.562) | **admission** to Phase 5 matrix C |
| `phase5_C_det.md` (T 0.609±0.012) | **fair skill** under locked seeds/schedule |

Do not cite 0.562 as the C det headline. Cite 0.609±0.012 (or successor physical-space table).

## Related

C×NLL remains the only C probabilistic ENCE survivor under latent protocol v2 (0.120). Physical-space rescoring (prereg amendment 2026-07-17) decides the cross-rep default.
