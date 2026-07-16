# Session handoff — dissertation data foundation

**Branch:** `residual_cube`  
**Updated:** 2026-07-16  
**Code home:** [`NeSPReSO2_onTemplate/`](NeSPReSO2_onTemplate/)  
**Conda:** `nespreso`

---

## Status: low-rank δσ₀ PASS — matrix admission (inference-isotonic claim)

| Phase | Status |
|-------|--------|
| 0–1 | Done |
| 2 | v3 HDF5 regen advancing |
| 3 | **PASS** (corrected floor) — see architecture claim below |
| 4 | **Next** — two-stage CRPS on this μ; cov export is `Σ_ρ = V diag(σ_z²) Vᵀ` |
| 5–6 | Pass = Phase 5 matrix admission; dissertation number = 3-seed mean±std |

---

## Architecture claim (do not silent)

Winning path is **σ₀-space** low-rank: `σ̂₀ = clim + scores @ V`.  
**Not** monotone in the head. Pre-isotonic test: **45.6%** profiles violate.  
**Guarantee:** *stable by construction at inference* via mandatory isotonic
(`project_monotone_sigma0_ctrl`) — §3.6 opt-2 / T1-D. Cost ΔT ≈ −0.0002 °C.  
Full-rank softplus+cumsum retains the stronger "hard constraint in the head" claim.
PLAN §3.2 updated. Eval reports `stability_guarantee: inference_isotonic`.

---

## Headline numbers

| quantity | value |
|----------|------:|
| chrono T RMSE (spice_v3) | **0.562** |
| clean chrono argo16 | 0.5367 |
| same-split floor (×1.10) | **0.590** |
| ratio | 1.047 |
| pre-inv σ₀ (after isotonic) | 0 |
| pred dens + true spice | 0.250 |
| true dens + pred spice | 0.404 |

**Ckpt:** `saved/.../lowrank_sigma0_spice_v3/model_best.pth`  
**Cache:** `../data/cache/train_ready_0f6129b27ddb.pkl`  
**Report:** [`reports/phase3_lowrank_sigma0_spice_eval.md`](reports/phase3_lowrank_sigma0_spice_eval.md)

---

## Record (hygiene)

| doc | role |
|-----|------|
| [`reports/gate_floor_provenance.md`](reports/gate_floor_provenance.md) | floor chain: leak → clean retrain → 0.5903 |
| [`reports/finding_compress_physical_space.md`](reports/finding_compress_physical_space.md) | a-space PCA failure as citable finding (35× gap) |
| [`reports/phase3_density_shift_diag.md`](reports/phase3_density_shift_diag.md) | leakage erratum (argo16 mse_σ 0.21) |
| [`reports/phase3_density_shift_diag_clean.md`](reports/phase3_density_shift_diag_clean.md) | clean re-run; plumbing survives |
| eval report `eval_hygiene.test_evals_consumed` | **2** selection evals on σ₀ family (v2→v3); forking-paths noted |

---

## Next

1. **Phase 4** on this μ: two-stage CRPS; val-only σ recalibration for ENCE; iterate on val, one test score per matrix cell.
2. Month-clim: defer (Phase 5 variant only if free).
3. Merge to main: defensible **only with** inference-isotonic wiring + floor provenance in the same merge (both now in tree — commit when asked).

```bash
tmux attach -t v3_hdf5_regen
```
