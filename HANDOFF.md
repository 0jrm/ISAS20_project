# Session handoff — dissertation data foundation

**Branch:** `residual_cube` (merged → `master` @ `56f1e18`)  
**Updated:** 2026-07-16  
**Code home:** [`NeSPReSO2_onTemplate/`](NeSPReSO2_onTemplate/)  
**Conda:** `nespreso`

---

## Status: Phase 4 low-rank CRPS run complete — ENCE miss / Spearman pass

| Phase | Status |
|-------|--------|
| 0–1 | Done |
| 2 | v3 HDF5 regen advancing |
| 3 | **PASS** (merged to master) — inference-isotonic claim |
| 4 | **Run done** — low-rank CRPS; Spearman PASS, ENCE MISS after val σ×α; §4.4 PSD+MC on score-σ export green |
| 5–6 | Pass = Phase 5 matrix admission; dissertation number = 3-seed mean±std |

---

## Architecture claim (do not silent)

Winning path is **σ₀-space** low-rank: `σ̂₀ = clim + scores @ V`.  
**Not** monotone in the head. Pre-isotonic test: **45.6%** profiles violate.  
**Guarantee:** *stable by construction at inference* via mandatory isotonic
(`project_monotone_sigma0_ctrl`) — §3.6 opt-2 / T1-D. Cost ΔT ≈ −0.0002 °C.  
Prob σ lives on scores; cov export `Σ_ρ = V diag(σ_z²) Vᵀ`.

---

## Headline numbers

### Phase 3 μ (deterministic)

| quantity | value |
|----------|------:|
| chrono T RMSE (spice_v3) | **0.562** |
| same-split floor (×1.10) | **0.590** |
| ratio | 1.047 |

### Phase 4 low-rank CRPS (`lowrank_crps_v1`)

| quantity | val raw | val σ×α | test raw | test σ×α |
|----------|--------:|--------:|---------:|---------:|
| CRPS | 0.585 | 0.587 | 0.715 | **0.715** |
| ENCE | 0.286 | 0.246 | 0.506 | **0.361** |
| Spearman | 0.441 | 0.441 | 0.519 | **0.519** |
| slope | 1.51 | 1.33 | 1.76 | 1.56 |

α = 1.134 (val RMSE/RMV). Anchors: Spearman **PASS**; ENCE **MISS** (<0.20).  
vs prior full-rank FAIL μ CRPS 1.15 / Spearman 0.65 — mean recovered, ranking still strong.

**Ckpt s2:** `saved/.../lowrank_crps_v1_s2/model_best.pth`  
**Cache:** `../data/cache/train_ready_0f6129b27ddb.pkl`  
**Report:** [`reports/phase4_lowrank_crps_eval.md`](reports/phase4_lowrank_crps_eval.md)  
**§4.4 score-σ export:** `dacov.density_lowrank_covariance` — PSD + MC agreement green
(`mc_vs_diag_agreement_lowrank`, n_draw=2000, rtol=0.15, max_rel≈0.057).

---

## Record (hygiene)

| doc | role |
|-----|------|
| [`reports/gate_floor_provenance.md`](reports/gate_floor_provenance.md) | floor chain → 0.5903 |
| [`reports/finding_compress_physical_space.md`](reports/finding_compress_physical_space.md) | a-space PCA failure |
| [`reports/phase4_lowrank_crps_val.md`](reports/phase4_lowrank_crps_val.md) | val-only iteration |
| eval hygiene | Phase 4: 1 test score after val α fit |

---

## Next

1. **ENCE recovery options (val only):** per-dim / depth-band σ scales; or longer stage-2; do not burn more test scores.
2. Phase 5 matrix preregistration when ENCE path chosen (or admit Spearman as DA-ranking claim with ENCE caveat).
3. Month-clim: defer. Phase 2 v3 HDF5 still open for error-channel stratum.

```bash
tmux attach -t v3_hdf5_regen
```
