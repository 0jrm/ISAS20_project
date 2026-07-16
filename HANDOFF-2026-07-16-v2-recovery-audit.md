# Session handoff — 2026-07-16 v2 recovery Phase 0/1 audit

**Authoritative branch:** `audit/phase0-1` @ `7901d1b`  
**Tags:** `pre-audit-phase0-1` (unaudited snapshot), `evalphys-v1.1.0` (audited metrics)  
**Working branch `residual_cube`:** still at `820e598` — **merge not done** (interrupted). Do this first before Phase 3.

**Conda:** `nespreso`. CPU: `srun --ntasks=1 --cpus-per-task=8 …`

---

## Status (one glance)

| Item | State |
|------|--------|
| Phase 0 `evalphys/` | **Done + audited** → v1.1.0 |
| Phase 1 T1/T2/T3 | **Done + audited** |
| T2 stale gate | **OPEN** (Phase 2.1 SSS gap **satisfied**) |
| Decision R1 | **Accepted** — Phase 3.2 monotone head |
| Merge into `residual_cube` | **PENDING** |
| Full `selfcheck.py` | Phase-boundary suite green; full run stalls on L3 altimetry NFS crawl (`test_stratified_eval_l3_cache_smoke`) |

---

## What happened

1. Prior agent left Phase 0/1 uncommitted on `residual_cube`. Snapshot commit + tag `pre-audit-phase0-1`, then audited on `audit/phase0-1`.
2. Audit fixed: inverted `exclude_top_m`, invalid mixed T/S RMSE, σ₀ metric + PLAN note, `gsw_backend`, stale detector tests, golden skip (R4).
3. Completeness pass (user): B/C historical σ₀ rows; F.3 table in `reports/backend_equivalence.*`; mechanism narrative; R1 accepted.

### Headline science (dissertation claim)

Under historical σ₀ **profile** rate (tol=0.01 kg/m³, test n=623):

| row | rate |
|-----|-----:|
| RAW | 1.12% |
| A PCA-16 | 21.51% |
| B joint EOF | **22.63%** (does not help) |
| C density+spice | **21.83%** (≈ A) |
| D monotone | **0.48%** |

**Claim:** load-bearing cause is **truncation itself**, not T/S separateness. Soft bases fail; only the **hard monotone σ₀ constraint** buys stability (21.51% → 0.48%).

**Caveat:** hard constraint = σ₀ mono **pre-inversion**. Residual ~0.3–0.5% post-inversion is Newton round-trip → Phase 3 must track **inversion fidelity** as first-class.

T1 N² ≥5× rule for B/C: **not met** (escalate fired) → human R1 = proceed monotone anyway.

### gsw_torch JOSS evidence

[`reports/backend_equivalence.md`](reports/backend_equivalence.md): σ₀ max|Δ|≈9.0e-5, RMS≈1.1e-5; 14 N² flips @1e-8. Headline metrics stay on reference `gsw` 3.6.19.

---

## Read first (order)

1. This file / [`HANDOFF.md`](HANDOFF.md)
2. [`PLAN-v2-recovery.md`](PLAN-v2-recovery.md) — **Changelog** + §3.2 σ₀-vs-N² note
3. [`reports/audit_phase0_1.md`](reports/audit_phase0_1.md)
4. [`reports/phase1_decisive_tests.md`](reports/phase1_decisive_tests.md)
5. [`reports/t1_basis_stability.md`](reports/t1_basis_stability.md), [`reports/backend_equivalence.md`](reports/backend_equivalence.md)

---

## Next actions (do in order)

### 0. Finish interrupted merge

```bash
cd /unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project
# clear stale lock if present
rm -f .git/packed-refs.lock .git/index.lock
git checkout residual_cube
git merge --no-ff audit/phase0-1 -m "merge(audit): Phase 0/1 audit into residual_cube (evalphys-v1.1.0, R1)"
git push origin residual_cube
```

### 1. Execute R1 — Phase 3 ∥ Phase 2.2

Per plan dependency graph: **Phase 3** (monotone density head + spice + inversion fidelity metrics) **in parallel with Phase 2.2** (error-channel ingestion). Phase **2.1 is done** — do not re-download SSS.

### 2. Do not forget

| ID | Note |
|----|------|
| R4 | `test_combined_pca_loss_v2` combined/wmse golden drift (pre-existing on `820e598`) — **Phase 5 prerequisite**, not blocking Phase 3. No silent regen. |
| Full selfcheck | Avoid/fix `test_stratified_eval_l3_cache_smoke` walking all of `data/raw/altimetry_l3` on NFS; phase-boundary suite is the practical gate. |
| Phases 4–6 | Still gated as in PLAN; do not skip to Phase 5 matrix before R4. |

---

## Key paths

| Path | Role |
|------|------|
| `NeSPReSO2_onTemplate/evalphys/` | Frozen metrics v1.1.0 |
| `NeSPReSO2_onTemplate/evalphys/gsw_backend.py` | Headline = reference `gsw` |
| `NeSPReSO2_onTemplate/scripts/t1_basis_stability.py` | T1 |
| `NeSPReSO2_onTemplate/scripts/backend_equivalence.py` | F.3 → reports |
| `NeSPReSO2_onTemplate/diagnostics/stale_sat/` | T2 + unit tests |
| `data/cache/train_ready_4411c65ee518.pkl` | ARGO cache used for T1 |

## Commands

```bash
cd NeSPReSO2_onTemplate
srun --ntasks=1 --cpus-per-task=8 conda run -n nespreso python -m pytest evalphys/tests -q
srun --ntasks=1 --cpus-per-task=8 conda run -n nespreso python diagnostics/stale_sat/test_stale_detector.py
srun --ntasks=1 --cpus-per-task=8 conda run -n nespreso python scripts/t1_basis_stability.py --gsw-backend gsw
```

---

## Commit stack on `audit/phase0-1` (after snapshot)

```
7901d1b docs(phase1): B/C historical rows, mechanism update, F.3 equivalence table
91d32cc docs(reports): audit_phase0_1 findings and HANDOFF next task
7c1ede5 docs(plan): changelog + sigma0-vs-N2 note for Phase 3.2
8e75796 fix(selfcheck): skip pre-existing combined_pca golden; wire audit gates
b33df1e test(stale): synthetic time-constant injection and H5 key checks
c248528 fix(t1): per-variable RMSE, reconciliation, and D diagnosis
7c7ada3 feat(evalphys): configurable gsw backend + sigma0 metric (v1.1.0)
7f9a043 chore(phase0-1): snapshot pre-audit implementation (UNAUDITED)
```
