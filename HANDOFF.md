# ISAS20_project — session handoff

**Branch:** `nespreso-v2-port`  
**Updated:** 2026-06-18  
**Code home:** [`NeSPReSO2_onTemplate/`](NeSPReSO2_onTemplate/)

Read this first. Detailed history: [`PLAN-patch-arch-handoff.md`](PLAN-patch-arch-handoff.md), [`PLAN-phase5.md`](PLAN-phase5.md), [`PLAN-phase6.md`](PLAN-phase6.md), [`PLAN.md`](PLAN.md).

---

## What this repo is

Offline **PyTorch batch training** for NeSPReSO v2 (surface → latent → T/S profiles). **Not** a web app.

| Tag | Data | Config | Arch |
|-----|------|--------|------|
| `isas20` | ISAS HDF5 + newsat patches | `config_isas_patch.json` | `PatchConvMLP` |
| `argo_v2` | v2 pickle + COAPS | `config_argo.json` | `PatchConvMLP` (point mode) |

---

## Production (ISAS GoM)

**Use PCA-16:** `config_isas_patch.json` → checkpoint `saved/models/NeSPReSO2_ISAS_GoM_patch/patch16_scales/model_best.pth`

Test `raw_profile_rmse`: **T 1.016 / S 5.318**

Phase 5 learned-decoder pipeline is **closed for ISAS production** (science appendix only — see [`PLAN-phase5.md`](PLAN-phase5.md)).

**Next:** Phase 6 — GoM diagnostics + results narrative ([`PLAN-phase6.md`](PLAN-phase6.md)). **Global model dropped.**

### Phase 6 (GoM-only)

| Task | Status | Artifact |
|------|--------|----------|
| GoM ML diagnostics | **done** | `scripts/gom_diagnostics.py` → `saved/gom_diagnostics/` |
| Results table | **done** | `scripts/results_table.py` → `saved/results/eval_table.md` |
| Decoder eval `loss: NaN` | **done** | `DecoderProfileLoss` uses `nanmean` on profile MSE |

```bash
srun --ntasks=1 --cpus-per-task=8 --gres=gpu:1 python3 scripts/gom_diagnostics.py
srun --ntasks=1 --cpus-per-task=8 python3 scripts/results_table.py
```

---

## Current status (GoM, Jun 2026)

### Done (Phases 0–4)

- v2 port, dual caches, `PatchConvMLP`, 16-PC bases, loss scales, agent train monitor
- Phase 4b **exhausted** — no GoM lever hits ≥10% full-step speedup

### Done (Phase 5 — closed)

| Item | Path / notes |
|------|----------------|
| Stage A AE training | `scripts/train_profile_ae.py` — `Autoencoder`, `ResAutoencoder`, `--layer-scale`, `--arch-tag` |
| Stage A2 latent export | `scripts/export_ae_latents.py` — cache keys `ae_targets_dim32`, `_res`, `_satres`, … |
| Stage B decoder loss | `model/loss.py` — `DecoderProfileLoss`, `loss_config.mode: decoder` |
| Stage B configs | `config_isas_patch_decoder*.json`, `config_isas_patch_decoder_dim32_satres.json` |
| Surface-residual decode | `ResAutoencoder` + SST/SSS broadcast skip; wired through Stage B loss |
| Residual MLP trunk | `PatchConvMLP(residual=True)` |
| Eval | `eval_run.py` — decoder mode; trust `raw_profile_rmse` only (`loss: NaN` known) |
| Selfcheck | decoder loss, residual AE/MLP, surface-residual path |

### Phase 5 verdict — **CLOSED (ISAS)**

**Go/no-go #1 (test `raw_profile_rmse` within 5% of PCA): FAILED** on all converged Stage B runs.

| Model | T RMSE | S RMSE | vs PCA T / S | Notes |
|-------|-------:|-------:|--------------|-------|
| **PCA prod** | **1.016** | **5.318** | — | `patch16_scales` |
| decoder16_ae | 1.314 | 7.078 | +29% / +33% | early-stopped ep 7822 |
| decoder32_ae | 1.287 | 6.416 | +27% / +21% | best ep 3703 |
| decoder32_res_ae | 26.45 | 31.37 | — | **abandoned ep 138** — not converged; do not headline |
| decoder32_satres_2ep | — | — | — | 2-epoch smoke only; not eval'd on test |

**Go/no-go #2 (ARGO speed ≥10%): not demonstrated.** ISAS decoder step ~3% *slower* than PCA at dim16.

**Root cause:** Stage A profile AE recon is strong (salinity especially), but **surface → AE-latent** does not transfer; frozen decoder amplifies latent error. More MLP capacity and residual blocks improved val_loss, not test profiles.

**Surface-residual experiment (satres):** Architecturally valid (SST/SSS skip at decode), but Stage A regressed vs best ResAutoencoder (T 0.394 / S 0.508 vs 0.202 / 0.181 val recon). Not pursued further on ISAS.

**What worked (science appendix):**
- `ResAutoencoder` Stage A (hidden residual blocks): T **0.202**, S **0.181** val recon
- dim32 MLP lower val_loss than dim16

**Known eval caveats (accepted):**
- `loss: NaN` in decoder eval — use `raw_profile_rmse` only
- `latent_target_rmse` not comparable across dim16/32/AE
- Pin `cache_path` to `train_ready_e6f936bdc80a.pkl` for decoder configs

---

## If revisiting (not scheduled)

| Idea | When |
|------|------|
| ARGO full-step speed benchmark | Only if perf on 1801 levels still matters |
| Train surface on **PCA latents**, decode with AE at eval | Diagnostic — separates latent target from surface head |
| Joint fine-tune last decoder layers | New hypothesis only |

---

## Eval numbers (test split — within-tag only)

### PCA production

| Model | temp RMSE | sal RMSE | Checkpoint |
|-------|----------:|---------:|------------|
| PatchConvMLP ISAS patch 16-PC | **1.016** | **5.318** | `patch16_scales/model_best.pth` |
| PatchConvMLP ARGO 16-PC | **0.416** | **0.072** | `argo16_scales/model_best.pth` |

### Phase 5 decoder runs (ISAS, cache `e6f936bdc80a`)

| Model | temp RMSE | sal RMSE | Eval JSON |
|-------|----------:|---------:|-----------|
| decoder16_ae | 1.314 | 7.078 | `saved/eval_isas_decoder16_test.json` |
| decoder32_ae | 1.287 | 6.416 | `saved/eval_isas_decoder32_test.json` |
| decoder32_res_ae | 26.45 | 31.37 | `saved/eval_isas_decoder32_res_test.json` (immature ckpt) |

Cross-tag: [`eval_matched.py`](NeSPReSO2_onTemplate/eval_matched.py) only.

---

## Quick commands

```bash
cd NeSPReSO2_onTemplate

srun --ntasks=1 --cpus-per-task=8 python3 selfcheck.py

# Production train / eval (PCA)
srun --ntasks=1 --cpus-per-task=8 --gres=gpu:1 \
  python3 train.py -c config_isas_patch.json -id my_run
srun --ntasks=1 --cpus-per-task=8 --gres=gpu:1 \
  python3 eval_run.py -c config_isas_patch.json \
  -r saved/models/NeSPReSO2_ISAS_GoM_patch/patch16_scales/model_best.pth \
  --out saved/eval_isas_patch16_pca_test.json

# Phase 5 appendix (AE + decoder) — see PLAN-phase5.md
```

---

## File map (Phase 5)

| Path | Role |
|------|------|
| `config_isas_patch_decoder*.json` | Stage B configs |
| `config_isas_patch_decoder_dim32_satres.json` | Surface-residual Stage B (not production) |
| `scripts/train_profile_ae.py` | Stage A per-variable AE |
| `scripts/export_ae_latents.py` | AE targets → cache |
| `saved/decoders/isas20/` | Frozen profile decoders |
| `saved/plots/decoder16_vs_decoder32_train_curves.png` | Stage B curves |

---

## Related plans

| Doc | Purpose |
|-----|---------|
| [`PLAN-phase6.md`](PLAN-phase6.md) | GoM diagnostics and results |
| [`PLAN-phase5.md`](PLAN-phase5.md) | Phase 5 close-out + results |
| [`PLAN-patch-arch-handoff.md`](PLAN-patch-arch-handoff.md) | Phases 1–4b |
| [`AGENTS.md`](AGENTS.md) | Agent + ponytail rules |
