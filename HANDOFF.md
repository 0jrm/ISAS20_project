# Session handoff — dissertation data foundation

**Branch:** `residual_cube` (merge `audit/phase0-1` after push)  
**Base:** legacy ISAS production on `nespreso-v2-port` — not replaced by dissertation work  
**Updated:** 2026-07-16  
**Code home:** [`NeSPReSO2_onTemplate/`](NeSPReSO2_onTemplate/)

**v2 recovery — R1 ACCEPTED (2026-07-16):** Audit on `audit/phase0-1`, tags `pre-audit-phase0-1` + **`evalphys-v1.1.0`**. See [`reports/audit_phase0_1.md`](reports/audit_phase0_1.md), [`reports/phase1_decisive_tests.md`](reports/phase1_decisive_tests.md).

- **Mechanism update:** B joint EOF ≈ A under historical σ₀ (22.63% vs 21.51%) — truncation itself, not T/S separateness; only hard monotone constraint cuts to **0.48%**. Soft bases do not buy stability.
- **Phase 3 caveat:** hard constraint = σ₀ mono **pre-inversion**; residual ~0.3–0.5% is Newton round-trip → track **inversion fidelity** as first-class.
- **Phase 2.1:** SSS gap filled + detector proven → **satisfied**.
- **Next:** Phase **3** (monotone head) ∥ Phase **2.2** (error channels). Phase 5 matrix blocked until R4 golden root-cause (`combined_pca_loss_v2`).
- **gsw_torch JOSS evidence:** [`reports/backend_equivalence.md`](reports/backend_equivalence.md) (σ₀ max|Δ|≈9.0e-5, 14 N² flips @1e-8).

**Known selfcheck skip (R4):** `test_combined_pca_loss_v2` combined/wmse golden — Phase 5 prerequisite, not blocking Phase 3.

**Read first:** [`PLAN-v2-recovery.md`](PLAN-v2-recovery.md) (changelog), [`PLAN.md`](PLAN.md).
