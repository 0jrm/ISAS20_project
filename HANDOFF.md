# Session handoff — dissertation data foundation

**Branch:** `residual_cube`  
**Updated:** 2026-07-16  
**Code home:** [`NeSPReSO2_onTemplate/`](NeSPReSO2_onTemplate/)  
**Conda:** `nespreso`

---

## Status: Phase 3–4 FAIL-state committed — recover density, then re-gate

Full-cache density_spice + CRPS two-stage are on disk and in git (reports/code). **Gates govern phase progression, not whether work is committed.** Skill gate FAIL vs argo16; hard σ₀ stability holds. Checkpoints live under `saved/` (gitignored) on the shared FS.

| Phase | Status |
|-------|--------|
| 0–1 | Done |
| 2 | Partial — **T2 OPEN**; v3 full HDF5 regen **launched** (`config_argo_patch_l4_err.json` → `satellite_NeSPReSO_v2_ARGO_GoM_err.h5`) |
| 3 | Soft gate **FAIL** — σ₀=0 but overall T RMSE 0.72 vs argo16 0.42 ([`reports/phase3_full_train_eval.md`](reports/phase3_full_train_eval.md)) |
| 4 | Informational — ENCE 0.33 **MISS**; Spearman 0.65 **PASS** (≫0.12) ([`reports/phase4_full_eval.md`](reports/phase4_full_eval.md)) |
| 5–6 | Blocked until skill recovery (or §3.6 option-2 fallback) + R4 golden |

---

## Diagnosis (Phase 3)

Spice was stuck (mse_τ ≈ 180) until v10 set λ_ρ/λ_τ = 0.05/1 spice-first — then density sat at climatology (mse_ρ≈1.3). With λ_ρ=0.05 **and** `a=a_clim+δa` **and** density weight×0.01, the density branch was barely asked to move δa. Classic two-branch whack-a-mole — fix structurally, not with another λ sweep. Prefer EMA-normalized / sequential procedures over representation-specific magic (Phase 5 fairness).

Discarded negatives (two lines each in the Phase 3 report): **v8**, **v9** (spice stall), **v10s2e** (weight-amplify fine-tune blow-up).

---

## Best deterministic checkpoint

`NeSPReSO2_onTemplate/saved/argo_densityspice/models/NeSPReSO2_ARGO_GoM_densityspice/phase3_full_v10/model_best.pth`  
Cache: `data/cache/train_ready_cd9e08b6c630.pkl`  
Recipe: residual `a=a_clim+δa`, spice-first `λ_ρ=0.05` / `λ_τ=1.0`, density weight×0.01, grad_clip 5.

| Metric | Value |
|--------|-------|
| σ₀ profile rate | **0.000** |
| N² profile rate | **0.000** |
| overall T RMSE | 0.724 (argo16=0.416, ratio 1.74) |
| spice mse_τ | ≈60 (≈Ridge) |

---

## Best CRPS two-stage

`.../phase4_crps_v2_s2/model_best.pth` (parent `phase4_crps_v2`)  
Config: `config/argo/config_argo_densityspice_crps.json`  
Launcher: `scripts/train_prob_twostage.py --stage1-epochs 50 --stage2-epochs 50`  
Headline: Spearman 0.65 (ranking works). Defer scalar σ calib + ENCE re-judge until mean recovers.

---

## Code fixes landed this session

- `derive_loss_scales.py`: clim-`a` init (not zero-`a`)
- `DensitySpiceLoss`: residual `a_clim + δa`
- `base_trainer.py`: save `model_best` on every improvement (not only `save_period`); tolerate missing optimizer state
- `trainer.py`: optional `grad_clip_norm`
- Eval: `scripts/eval_density_spice.py`, `scripts/eval_phase4_crps.py`

---

## Next (ordered)

1. **Blame-split swap test** (cheap): reconstruct T via (true σ₀, pred τ) and (pred σ₀, true τ) — locate the 0.724→0.458 gap.
2. **Decouple branches:** density-only (λ_τ=0) ablation; then EMA-normalized per-branch losses or two small models; sequential density→freeze→spice as fallback.
3. **Re-CRPS stage-2** after mean recovers → optional val scalar σ calib → re-judge ENCE.
4. If skill still FAIL → fire **§3.6 fallback option 2** (isotonic σ₀ projection at inference; T1-D). Then Phase 5.
5. After v3 HDF5 finishes: rebuild density_spice cache with `inputs_err` → Phase 4.5 error-channel conditioning.

```bash
# Phase 2 v3 regen (tmux; resumable)
cd /unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project
srun --ntasks=1 --cpus-per-task=8 conda run -n nespreso --no-capture-output \
  python utils/generate_argo_satellite_data.py \
  -c NeSPReSO2_onTemplate/config/argo/config_argo_patch_l4_err.json --batch-size 100

# Deterministic eval
cd NeSPReSO2_onTemplate
srun --ntasks=1 --cpus-per-task=8 conda run -n nespreso \
  python scripts/eval_density_spice.py -c config/argo/config_argo_densityspice.json \
  -r saved/argo_densityspice/models/NeSPReSO2_ARGO_GoM_densityspice/phase3_full_v10/model_best.pth
```
