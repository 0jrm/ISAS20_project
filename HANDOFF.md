# Session handoff — dissertation data foundation

**Branch:** `residual_cube`  
**Updated:** 2026-07-16  
**Code home:** [`NeSPReSO2_onTemplate/`](NeSPReSO2_onTemplate/)  
**Conda:** `nespreso`

---

## Status: §3.6 opt-2 gate FAIL on chronological — in-head still critical path

| Phase | Status |
|-------|--------|
| 0–1 | Done |
| 2 | v3 HDF5 regen advancing (batches written through 300+) |
| 3 | Soft gate FAIL; shift diags + argo16+isotonic gate done |
| 4–6 | Blocked on skill (or a future chrono-trained baseline + projection) |

**Do not merge to main.**

---

## Shrinkage fix (σ₀ space)

a-space shrink≈0 contradicted densonly beating clim on val. Recomputed:

| era | densonly σ₀-shrink | densonly mse_σ | clim mse_σ |
|-----|--------------------|----------------|------------|
| val | **0.318** | 0.43 | 1.14 |
| test | **0.275** | 0.91 | 1.26 |

Under-correction still true (~0.3× anomaly variance), but not “δa≈0”. Loss already post softplus+cumsum.

---

## argo16 + isotonic gate (§3.6 opt-2)

Report: [`reports/phase3_argo16_isotonic_gate.md`](reports/phase3_argo16_isotonic_gate.md)

| | chronological | random (published regime) |
|--|---------------|---------------------------|
| argo16 raw T | **0.514** | **0.416** |
| +isotonic T | 0.514 | ~0.416 |
| pre-inv σ₀ | **0.000** | 0.000 |
| vs published 0.458 floor | FAIL | PASS skill |

**Published 0.416 was random-split.** Chronological argo16 already sits above the coded floor; projection cannot invent skill. Stability half works (pre-inv σ₀=0, proj cost ~0.0015°C; post-inv profile rate at tol=0 is O(1e-6) Newton noise).

---

## Next (reordered)

1. **Low-rank δa:** PCA on train `(a_true − a_clim)` → ~16 scores → decode δa → softplus+cumsum (restore coordination argo16 gets for free). Keep v10 spice frozen.
2. **Month-resolved / harmonic a_clim** (JJA clim mse 1.8–2.1).
3. Confirm loss stays in σ₀ space (already does).
4. SSH→density ablation last (argo16 already proves inputs carry signal).
5. v3 HDF5 continues in background for Phase 4.5 when skill returns.

```bash
tmux attach -t v3_hdf5_regen
srun --ntasks=1 --cpus-per-task=8 conda run -n nespreso \
  python scripts/eval_argo16_isotonic_gate.py --split-mode chronological
```
