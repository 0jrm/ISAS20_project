# ISAS20_project — session handoff

**Branch:** `nespreso-v2-port`  
**Updated:** 2026-06-17  
**Code home:** [`NeSPReSO2_onTemplate/`](NeSPReSO2_onTemplate/)

Read this first. Detailed history lives in [`PLAN-patch-arch-handoff.md`](PLAN-patch-arch-handoff.md), [`PLAN-phase5.md`](PLAN-phase5.md), [`PLAN.md`](PLAN.md).

---

## What this repo is

Offline **PyTorch batch training** for NeSPReSO v2 (surface → PCA latent → T/S profiles). **Not** a web app — no HTTP endpoints, LiveViews, or database.

| Tag | Data | Config | Arch |
|-----|------|--------|------|
| `isas20` | ISAS HDF5 + newsat patches | `config_isas.json`, `config_isas_patch.json` | `PredictionModel` or `PatchConvMLP` |
| `argo_v2` | v2 pickle + COAPS | `config_argo.json` | `PatchConvMLP` (point mode) |

---

## Current status (GoM, Jun 2026)

### Done

- Phases 0–4: v2 port, dual-dataset caches, `PatchConvMLP`, 16-PC bases, loss scales, agent train monitor
- Phase 4b **exhausted** at GoM scale — no lever hits 10% full-step speedup (batch, `compile_loss`, `pred_profile_cached`, combined stack)
- GoM training runs complete; test eval JSON in `NeSPReSO2_onTemplate/saved/eval_*_test.json`
- Notebook comparison surface: `notebooks/compare_v2_vs_template.ipynb` + `run_compare.py` headless smoke
- Phase 5 Stage A: `train_profile_ae.py`, `benchmark_profile_ae_dims.py` — ISAS sal AE beats PCA; ARGO still PCA-dominant at 200 epochs

### Active / next

| Priority | Task | Doc |
|----------|------|-----|
| 1 | Notebook comparison rewrite (eval surface) | `notebooks/`, `nb_metrics.py` |
| 2 | Phase 5 Stage B: `DecoderProfileLoss`, latent cache export | [`PLAN-phase5.md`](PLAN-phase5.md) |
| 3 | Security hardening (allowlist config types, pickle trust doc) | see **Trust boundaries** below |
| 4 | Phase 4c (optional): ISAS global + DDP | only if leaving GoM |

### Do not do (GoM)

- Enable `combo_phase4b_all` — **~10% slower** than baseline
- Chase `torch.compile` / bf16 on GoM epochs (~25 ms/epoch)
- Compare raw `eval_run.py` RMSE across ISAS vs ARGO tags without caveats

---

## Eval numbers (test split — within-tag only)

| Model | temp RMSE | sal RMSE | Checkpoint |
|-------|----------:|---------:|------------|
| PatchConvMLP ISAS patch 16-PC | 1.016 | **5.32** | `patch16_scales/model_best.pth` |
| PredictionModel ISAS point 15-PC | **1.002** | 5.53 | `baseline15pc/model_best.pth` |
| PatchConvMLP ARGO 16-PC | **0.416** | **0.072** | `argo16_scales/model_best.pth` |

Cross-tag comparison: use [`eval_matched.py`](NeSPReSO2_onTemplate/eval_matched.py) (~2k colocated profiles), not raw RMSE.

---

## Bottleneck (performance)

**#1:** `CombinedPCALoss` — PCA profile reconstruction (`pcs @ components + mean`), especially ARGO **1801** depth levels. MLP forward is ~283K params; VRAM &lt;1% on A100 at GoM batch sizes.

Profiler: `Adam.step` + `aten::mm` in loss, not model forward. See README ML benchmarks section.

---

## Trust boundaries (security)

Attack surface = **shared HPC filesystem**, not internet.

| Risk | Location | Mitigation (not yet all done) |
|------|----------|-------------------------------|
| `pickle.load` on caches | `data_loaders.py`, `train.py`, preproc | Only load caches you built; migrate to npz+json later |
| `torch.load(weights_only=False)` | `eval_run.py`, checkpoints | Prefer `weights_only=True` where state_dict only |
| Config-driven `getattr` for arch/loader | `parse_config.py` | **TODO:** allowlist `type` fields in `validate_config` |
| `io.v2_src` path injection | `export_v2_cache.py` | Pin path in env, not writable JSON on shared FS |
| TensorBoard on login node | README | `tensorboard --host 127.0.0.1` + SSH tunnel |

---

## Quick commands

```bash
cd NeSPReSO2_onTemplate

# Health (no data required for core checks)
srun --ntasks=1 --cpus-per-task=8 python3 selfcheck.py

# Train
srun --ntasks=1 --cpus-per-task=8 --gres=gpu:1 python3 train.py -c config_isas_patch.json
srun --ntasks=1 --cpus-per-task=8 --gres=gpu:1 python3 train.py -c config_argo.json

# Eval (pair checkpoint with its cache)
srun --ntasks=1 --cpus-per-task=8 --gres=gpu:1 \
  python3 eval_run.py -c config_argo.json \
  -r saved/models/NeSPReSO2_ARGO_GoM/argo16_scales/model_best.pth --out /tmp/eval.json

# Agent dual-run monitor
python3 scripts/train_monitor.py --once --manifest saved/runs/<RUN_ID>/manifest.json

# Notebook smoke
srun --ntasks=1 --cpus-per-task=8 --gres=gpu:1 python3 notebooks/run_compare.py
```

Always wrap CPU work: `srun --ntasks=1 --cpus-per-task=8 ...`

---

## File map

| Path | Role |
|------|------|
| `train.py` | Cache ensure, batch resolve, train entry |
| `eval_run.py` / `eval_matched.py` | Test-split eval / cross-tag matched RMSE |
| `model/loss.py` | `CombinedPCALoss`, `loss_config.mode` |
| `preproc/preproc_isas_sat.py` | ISAS cache build |
| `preproc/export_v2_cache.py` | ARGO v2 pickle → cache |
| `scripts/train_monitor.py` | Agent status from `status.json` |
| `selfcheck.py` | v2 equivalence + one runnable check per non-trivial feature |

---

## Related plans

| Doc | Purpose |
|-----|---------|
| [`PLAN.md`](PLAN.md) | Original v2 port (Phases 0–7 done) |
| [`PLAN-patch-arch-handoff.md`](PLAN-patch-arch-handoff.md) | Phases 1–4b detail + benchmarks |
| [`PLAN-phase5.md`](PLAN-phase5.md) | AE/KAN decoder roadmap |
| [`PLAN-agent-train-monitor.md`](PLAN-agent-train-monitor.md) | Agent dual-run spec (implemented) |
| [`NeSPReSO2_onTemplate/README.md`](NeSPReSO2_onTemplate/README.md) | Ops, batch size, ML opt tables |
| [`AGENTS.md`](AGENTS.md) | Agent + ponytail instructions |
| [`eng-principles-pack/INDEX.md`](eng-principles-pack/INDEX.md) | On-demand engineering skills |
