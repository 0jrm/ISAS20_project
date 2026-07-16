# Session handoff — dissertation data foundation

**Branch to use:** `audit/phase0-1` @ `7901d1b` (tagged `evalphys-v1.1.0`)  
**`residual_cube`:** still @ `820e598` — **merge `audit/phase0-1` before Phase 3** (merge was interrupted 2026-07-16).  
**Updated:** 2026-07-16  
**Code home:** [`NeSPReSO2_onTemplate/`](NeSPReSO2_onTemplate/)

**Full session record:** [`HANDOFF-2026-07-16-v2-recovery-audit.md`](HANDOFF-2026-07-16-v2-recovery-audit.md)

---

## v2 recovery — Phase 0/1 audited; R1 accepted

Plan: [`PLAN-v2-recovery.md`](PLAN-v2-recovery.md) (see Changelog).  
Audit: [`reports/audit_phase0_1.md`](reports/audit_phase0_1.md).  
Phase 1: [`reports/phase1_decisive_tests.md`](reports/phase1_decisive_tests.md).

| Gate | Result |
|------|--------|
| Finding-1 (σ₀ profile tol=0.01) | Confirmed: RAW **1.12%** → A **21.51%** |
| B / C under same ruler | **22.63% / 21.83%** — joint EOF does **not** help |
| D monotone | **0.48%** — only material cut |
| T1 N² ≥5× for B/C | Not met → escalate → **R1: Phase 3.2** |
| T2 stale | **OPEN**; Phase **2.1 satisfied** (SSS re-downloaded 2026-07-03) |

**Dissertation claim:** mechanism is **truncation**, not T/S separateness. Soft bases fail; hard σ₀ constraint succeeds. Residual post-inversion violations (~0.3–0.5%) ⇒ track **inversion fidelity** in Phase 3.

**gsw_torch JOSS table:** [`reports/backend_equivalence.md`](reports/backend_equivalence.md) (σ₀ max|Δ|≈9e-5; 14 N² flips @1e-8). Headline = reference `gsw`.

**R4:** `combined_pca_loss_v2` golden drift — skip with reason; **Phase 5 prerequisite**, not blocking Phase 3.

---

## Next (strict order)

1. **Merge** `audit/phase0-1` → `residual_cube` and push (clear `.git/*.lock` if needed).
2. **Phase 3** (monotone density + spice + inversion-fidelity metrics) **∥ Phase 2.2** (error channels). Skip redoing 2.1.
3. Do **not** start Phase 5 ablation until R4 golden is root-caused.

```bash
rm -f .git/packed-refs.lock .git/index.lock
git checkout residual_cube
git merge --no-ff audit/phase0-1 -m "merge(audit): Phase 0/1 audit (evalphys-v1.1.0, R1)"
git push origin residual_cube
```

---

## Older context (still true)

**Read also:** [`PLAN.md`](PLAN.md), [`PLAN-dissertation-data-foundation.md`](PLAN-dissertation-data-foundation.md).  
**Close-out (2026-07-15):** [`HANDOFF-2026-07-15-agentic-close-out.md`](HANDOFF-2026-07-15-agentic-close-out.md) — "nature's 24.7%" was PCA-16 target; Phase 8 CLOSED.  
**Conda:** `nespreso`.
