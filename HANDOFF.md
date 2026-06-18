# ISAS20_project — session handoff

**Branch:** `nespreso-v2-port`  
**Updated:** 2026-06-18  
**Code home:** [`NeSPReSO2_onTemplate/`](NeSPReSO2_onTemplate/)

Read this first. Detailed history: [`PLAN-patch-arch-handoff.md`](PLAN-patch-arch-handoff.md), [`PLAN-phase5.md`](PLAN-phase5.md), [`PLAN.md`](PLAN.md).

---

## What this repo is

Offline **PyTorch batch training** for NeSPReSO v2 (surface → latent → T/S profiles). **Not** a web app.

| Tag | Data | Config | Arch |
|-----|------|--------|------|
| `isas20` | ISAS HDF5 + newsat patches | `config_isas_patch.json` | `PatchConvMLP` |
| `argo_v2` | v2 pickle + COAPS | `config_argo.json` | `PatchConvMLP` (point mode) |

---

## Current status (GoM, Jun 2026)

### Done (Phases 0–4)

- v2 port, dual caches, `PatchConvMLP`, 16-PC bases, loss scales, agent train monitor
- Phase 4b **exhausted** — no GoM lever hits ≥10% full-step speedup
- PCA production baseline: **T 1.016 / S 5.318** test `raw_profile_rmse` (`patch16_scales`)

### Done (Phase 5 — this session)

| Item | Path / notes |
|------|----------------|
| Stage A AE training | `scripts/train_profile_ae.py` — `Autoencoder`, `ResAutoencoder`, `--layer-scale`, `--arch-tag` |
| Stage A2 latent export | `scripts/export_ae_latents.py` — separate cache keys (`ae_targets_dim32`, `_res`, …) |
| Stage B decoder loss | `model/loss.py` — `DecoderProfileLoss`, `loss_config.mode: decoder` |
| Stage B configs | `config_isas_patch_decoder.json`, `_dim32.json`, `_dim32_res.json` |
| Data loader keys | `target_key` / `weight_key` in `NeSPReSO2DataLoader` |
| Residual / skip arch | `ResidualConvBlock`, `ResidualLinearBlock`, `ResAutoencoder`, `PatchConvMLP(residual=True)` |
| Eval fixes | `eval_run.py` — decoder axis indexing, decoder-mode latent RMSE, profile fallback |
| Training curve plot | `scripts/plot_decoder_train_curves.py` → `saved/plots/decoder16_vs_decoder32_train_curves.png` |
| Selfcheck | decoder loss, residual AE/MLP, `raw_profile_rmse` indexing |

### Active training (check `status.json`)

| Run ID | Config | Epoch (approx) | Best val_loss | GPU |
|--------|--------|----------------:|--------------:|-----|
| `decoder16_ae` | `config_isas_patch_decoder.json` | ~7600 / 8000 | **0.566** | 0 |
| `decoder32_ae` | `config_isas_patch_decoder_dim32.json` | ~3400 / 8000 | **0.488** | 1 |
| `decoder32_res_ae` | `config_isas_patch_decoder_dim32_res.json` | ~70 / 8000 | 0.607 (early) | 2 |

**None have finished or early-stopped yet.** decoder16 is near max epochs but `not_improved_count` was ~265 — may still run to 8000.

### Brutally honest Phase 5 verdict (so far)

**Go/no-go #1 (test `raw_profile_rmse` within 5% of PCA): FAILED for every Stage B run evaluated.**

| Model | T RMSE | S RMSE | vs PCA T / S |
|-------|-------:|-------:|--------------|
| **PCA baseline** | **1.016** | **5.318** | — |
| decoder16_ae | 1.314 | 7.078 | +29% / +33% |
| decoder32_ae | 1.287 | 6.416 | +27% / +21% |
| decoder32_res_ae | *(not eval'd yet)* | | |

**The science bet did not transfer:** Stage A AE salinity recon crushes PCA (0.18–0.21 vs 1.17), but the full **surface → AE-latent → frozen decoder** pipeline is **worse than PCA-16 on both variables**. Bigger MLP (4× params) helps val_loss and salinity RMSE modestly; still nowhere near PCA.

**Go/no-go #2 (ARGO speed): not re-tested** — ISAS decoder step ~same as PCA (~3% slower at dim16). ARGO is where speed might matter; ARGO AE still loses to PCA at 200 epochs.

**What actually works:**
- `ResAutoencoder` Stage A: T **0.202**, S **0.181** vs plain `Autoencoder` dim32 (0.327 / 0.207) — residual hidden blocks are the clearest win this session.
- dim32 MLP trains to lower val_loss faster than dim16 (0.49 vs 0.57 at ~3k epochs).

**What's broken / sloppy (fix or accept):**
- `eval_run.py` reports `loss: NaN` for decoder eval — `DecoderProfileLoss` uses PCA-reconstructed `true_profiles` for training but eval compares against raw cache profiles; salinity NaNs in raw truth are fine for `raw_profile_rmse` but poison the combined loss metric. **Headline on `raw_profile_rmse` only.**
- `latent_target_rmse` in decoder eval is AE-latent MSE, not comparable across dim16/32 — ignore it.
- Cache has multiple AE target keys (`ae_targets`, `ae_targets_dim32`, `ae_targets_dim32_res`); **pin `cache_path`** in decoder configs to `train_ready_e6f936bdc80a.pkl` or `ensure_cache` hash drift creates orphan pickles.
- First `decoder32_res_ae` launch failed (raw-profile `true_profiles` → NaN loss); fixed via `true_profiles_numpy()` in `train.py`.

---

## Recommendations (priority order)

1. **Let `decoder32_res_ae` run** — only variant combining better AE (ResAutoencoder) + residual MLP; re-eval at val_loss &lt; 0.50. If test RMSE still &gt;5% vs PCA, **stop Phase 5 for ISAS production** and keep PCA-16.
2. **Do not throw more params at ISAS** without a new idea (e.g. train surface model on PCA latents but decode with AE only at eval; or joint fine-tune decoder last layers). Capacity alone is not closing the gap.
3. **If pursuing speed:** benchmark decoder loss on **ARGO** only; ISAS 187 levels will never justify the complexity.
4. **Fix eval `loss: NaN`** — mask-aware profile MSE in `DecoderProfileLoss` or skip reporting combined loss in decoder mode (small diff).
5. **Security TODO** unchanged — allowlist config `type` fields; document pickle trust.

---

## Eval numbers (test split — within-tag only)

### PCA production (Phase 4)

| Model | temp RMSE | sal RMSE | Checkpoint |
|-------|----------:|---------:|------------|
| PatchConvMLP ISAS patch 16-PC | **1.016** | **5.318** | `patch16_scales/model_best.pth` |
| PatchConvMLP ARGO 16-PC | **0.416** | **0.072** | `argo16_scales/model_best.pth` |

### Phase 5 decoder runs (ISAS, same cache `e6f936bdc80a`)

| Model | temp RMSE | sal RMSE | Checkpoint |
|-------|----------:|---------:|------------|
| decoder16_ae | 1.314 | 7.078 | `decoder16_ae/model_best.pth` |
| decoder32_ae | 1.287 | 6.416 | `decoder32_ae/model_best.pth` |

Cross-tag: [`eval_matched.py`](NeSPReSO2_onTemplate/eval_matched.py) only.

---

## Quick commands

```bash
cd NeSPReSO2_onTemplate

srun --ntasks=1 --cpus-per-task=8 python3 selfcheck.py

# Stage A — profile AE
srun --ntasks=1 --cpus-per-task=8 --gres=gpu:1 \
  python3 scripts/train_profile_ae.py -c config_isas_patch.json \
  --arch ResAutoencoder --encoding-dim 32 --layer-scale 2 \
  --arch-tag ResAutoencoder_dim32

# Stage A2 — export latents (pin cache!)
srun --ntasks=1 --cpus-per-task=8 --gres=gpu:1 \
  python3 scripts/export_ae_latents.py -c config_isas_patch_decoder_dim32_res.json \
  --cache ../data/cache/train_ready_e6f936bdc80a.pkl \
  --decoder-dir saved/decoders/isas20/ResAutoencoder_dim32 \
  --target-key ae_targets_dim32_res --weight-key ae_weights_dim32_res

# Stage B — surface model
srun --ntasks=1 --cpus-per-task=8 --gres=gpu:1 \
  python3 train.py -c config_isas_patch_decoder_dim32_res.json -id decoder32_res_ae

# Eval (decoder mode — trust raw_profile_rmse only)
srun --ntasks=1 --cpus-per-task=8 --gres=gpu:1 \
  python3 eval_run.py -c config_isas_patch_decoder_dim32.json \
  -r saved/models/NeSPReSO2_ISAS_GoM_patch_decoder_dim32/decoder32_ae/model_best.pth \
  --out saved/eval_isas_decoder32_test.json

# Training curves
python3 scripts/plot_decoder_train_curves.py \
  --run decoder16 saved/log/NeSPReSO2_ISAS_GoM_patch_decoder/decoder16_ae/info.log \
  --run decoder32 saved/log/NeSPReSO2_ISAS_GoM_patch_decoder_dim32/decoder32_ae/info.log \
  -o saved/plots/decoder16_vs_decoder32_train_curves.png

python3 scripts/train_monitor.py --once --manifest saved/runs/<RUN_ID>/manifest.json
```

---

## File map (Phase 5 additions)

| Path | Role |
|------|------|
| `config_isas_patch_decoder*.json` | Stage B configs (dim16 / dim32 / dim32+residual) |
| `scripts/export_ae_latents.py` | Write AE targets into cache |
| `scripts/train_profile_ae.py` | Stage A per-variable AE |
| `scripts/plot_decoder_train_curves.py` | Parse `info.log` → PNG |
| `model/model.py` | `ResAutoencoder`, residual `PatchConvMLP` |
| `saved/decoders/isas20/` | Frozen profile decoders |
| `saved/plots/` | Training curve comparisons |

---

## Related plans

| Doc | Purpose |
|-----|---------|
| [`PLAN-phase5.md`](PLAN-phase5.md) | AE/decoder roadmap + session results |
| [`PLAN-patch-arch-handoff.md`](PLAN-patch-arch-handoff.md) | Phases 1–4b |
| [`AGENTS.md`](AGENTS.md) | Agent + ponytail rules |
