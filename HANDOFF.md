# Session handoff — dissertation data foundation

**Branch:** `residual_cube`  
**Updated:** 2026-07-16  
**Code home:** [`NeSPReSO2_onTemplate/`](NeSPReSO2_onTemplate/)  
**Conda:** `nespreso`

---

## Status: densonly ablation done — interference NOT the cause

| Phase | Status |
|-------|--------|
| 0–1 | Done |
| 2 | Partial — T2 OPEN; v3 full HDF5 regen **in progress** (tmux `v3_hdf5_regen`) |
| 3 | Soft gate **FAIL**; blame-split + densonly complete — see below |
| 4 | Informational — Spearman 0.65 PASS / ENCE 0.33 MISS |
| 5–6 | Blocked until skill recovery (or §3.6 option-2) + R4 golden |

---

## Results this session (keep)

### Blame-split (v10)

| Reconstruction | T RMSE |
|----------------|--------|
| pred both | 0.724 |
| true σ₀ + pred τ | **0.393** (≤ gate 0.457) |
| pred σ₀ + true τ | **0.522** |

Density owns the joint gap; spice with true density already clears the skill floor.

### Density-only (λ_τ=0) — [`reports/phase3_densonly_eval.md`](reports/phase3_densonly_eval.md)

| Readout | densonly | v10 |
|---------|----------|-----|
| pred σ₀ + true τ (fair density) | **0.547** | **0.522** |
| overall T | 1.675 (spice noise) | 0.724 |
| σ₀ rate | 0.000 | 0.000 |

**Discussion:** Turning off spice does not improve density skill. Multi-task interference / shared trunk is not the failure mode. Density still under-learns on chrono test (val mse_σ≈0.43, test≈0.91). Prefer sequential fine-tune of density with **v10 spice frozen**, or EMA-normalized joint — not another λ sweep. §3.6 option 2 (isotonic at inference) remains the pre-registered floor.

Discarded negatives (in Phase 3 report): v8, v9 (spice stall), v10s2e (weight-amplify blow-up).

---

## Best deterministic checkpoint (still)

`.../phase3_full_v10/model_best.pth` — overall T 0.724, σ₀=0, spice≈Ridge.

---

## Next (ordered)

1. ~~Blame-split~~ DONE  
2. ~~Density-only ablation~~ DONE — interference refuted  
3. **Sequential:** freeze v10 spice head + trunk (or trunk+spice), train density δa only with λ_ρ=1 — without weight amplify  
4. Or EMA-normalized dual-branch joint (procedure, not magic λ)  
5. Re-CRPS + scalar σ calib after mean recovers  
6. If still FAIL → §3.6 option 2 → Phase 5  
7. After v3 HDF5: cache rebuild with `inputs_err` → Phase 4.5

```bash
# v3 regen status
tmux attach -t v3_hdf5_regen
tail -f data/NeSPReSO_v2_ARGO_GoM_sat/v3_err_regen.log
```
