# NeSPReSO v2 on ISAS template

Port of [v2-nespreso](/unity/g2/jmiranda/v2-nespreso) into the victoresque training template. See [SOURCES.md](../SOURCES.md) for module mapping.

## Quick start

```bash
cd NeSPReSO2_onTemplate
pip install -r requirements.txt

# v2 equivalence + PCA checks (no data required)
srun --ntasks=1 --cpus-per-task=8 python3 selfcheck.py

# build both train-ready caches
srun --ntasks=1 --cpus-per-task=8 python3 preproc/preproc_isas_sat.py cache config_isas.json --force
srun --ntasks=1 --cpus-per-task=8 python3 preproc/export_v2_cache.py -c config_argo.json --force

# train (identical hyperparams; ISAS20+newsat vs ARGO+COAPS)
srun --ntasks=1 --cpus-per-task=8 python3 train.py -c config_isas.json
srun --ntasks=1 --cpus-per-task=8 python3 train.py -c config_argo.json

# test-split eval (raw profile RMSE — use matching config + checkpoint)
srun --ntasks=1 --cpus-per-task=8 python3 eval_run.py -c config_isas.json -r saved/models/.../checkpoint.pth
```

## Dual-dataset comparison

| Config | `io.dataset_tag` | Profiles | Depth grid |
|--------|-------------------|----------|------------|
| `config_isas.json` | `isas20` | ISAS HDF5 | 187 levels |
| `config_argo.json` | `argo_v2` | v2 pickle | 1801 m (0–1800) |

Both use `seed=42`, **chronological split** (ARGO default; ISAS still random legacy), v2-matched trainer settings (`early_stop=500`). Batch size defaults vary by config — see **Batch size** below.

## Batch size

GoM runs barely use GPU VRAM at the default `batch_size=512`. Use the batch benchmark to find throughput, or set **`batch_size: 0`** to auto-pick the largest batch that fits.

| Config key | Meaning |
|---|---|
| `batch_size: 512` | Fixed batch (capped by train-set size) |
| **`batch_size: 0`** | **Auto: probe largest batch that fits in VRAM** (also capped by train-set size) |
| `batch_size_safety: 0.95` | Shrink auto-probed max by 5% headroom (default) |

CLI override: `python3 train.py -c config_isas_patch.json --bs 0`

```bash
# Sweep powers-of-two up to max-fit; writes JSON under saved/benchmarks/
srun --ntasks=1 --cpus-per-task=8 --gres=gpu:1 \
  python3 scripts/benchmark_batch_size.py -c config_isas_patch.json

srun --ntasks=1 --cpus-per-task=8 --gres=gpu:1 \
  python3 scripts/benchmark_batch_size.py -c config_argo.json
```

**Findings (A100 80GB, GoM, Jun 2026):** VRAM stays under ~1% even at max batch — the bottleneck is **FLOPs/step**, not memory. Per-step throughput rises monotonically with batch size (512 → 222k samples/s; max 2755 → 773k samples/s on ARGO). Full-epoch wall-clock gain is modest (~4% ARGO) because max batch collapses to **1 step/epoch** (~2.9k ARGO / ~3.5k ISAS train samples). On GoM, `batch_size=0` therefore resolves to one batch per epoch (best per-step samples/s, but fewer optimizer steps per epoch). For fixed-batch training comparable to v2, keep `batch_size=512` on ARGO; ISAS patch ships with `batch_size: 0`.

| Config | Train N | Max fit | Best throughput batch | Peak VRAM | Epoch speedup vs 512 |
|---|---:|---:|---:|---:|---:|
| `config_isas_patch.json` (PatchConvMLP) | 3530 | 3353 | 3353 | ~335 MiB | — |
| `config_isas.json` (PredictionModel) | 3530 | 3353 | 3353 | ~68 MiB | — |
| `config_argo.json` (PatchConvMLP + PCA loss) | 2901 | 2755 | 2755 | ~215 MiB | ~4% (0.0233→0.0223 s/epoch) |

`config_isas_patch.json` ships with `batch_size: 0` (auto max). `config_argo.json` keeps `batch_size: 512` for v2 parity (max batch benchmarked but below 10% full-step bar).
Edit paths in `config_argo.json` (`v2_pickle`, `v2_src`) for your machine.

**Eval rule:** always pair checkpoint with the cache it was trained on (`eval_run.py`).

**Cross-tag metrics:** native `eval_run.py` numbers are **not** apples-to-apples (different test splits, depth grids, PCA bases, truths). Use `eval_matched.py` for comparable colocation RMSE on ~2k matched profiles (see `preproc/overlap.py`).

## Config highlights

| Key | Meaning |
|-----|---------|
| `input_params` | Feature flags (`timecos`…`ssh`, `sat`); `input_dim = sum(flags) - 1` when `sat` is true |
| `io.dataset_tag` | `isas20` or `argo_v2` — selects cache builder in `train.py` |
| `io.spatial_pad`, `io.temporal_pad` | `0,0` = center-pixel SST/SSS/SSH (v2 point inputs) |
| `outputs` | Ordered map `{name: n_components}`; `output_dim = sum(values)` |
| `data_loader.train_frac` / `val_frac` / `test_frac` | Split fractions (default 0.7/0.15/0.15) |
| `data_loader.split_mode` | `chronological` (dissertation default for ARGO) or `random` (legacy) |
| `data_loader.split_config` | Optional explicit date ranges per split |

## v2 equivalence

With `spatial_pad=0`, `temporal_pad=0`, and `outputs: {temperature: 15, salinity: 15}`, `PredictionModel` + `CombinedPCALoss` match v2 forward passes within `1e-6` (`selfcheck.py`).

## Notes

- **Loss scales** (`37.86`, `0.28`, `2.8294`, `0.0255`) are GoM temp/sal defaults; re-derive for new outputs or regions.
- **PCA** is fit per cache build; each dataset tag has its own PCA in its pickle.
- **Density penalty** is wired but off by default (`density.enabled: false`); enable in both configs for parity with v2 training.
- **Cross-tag RMSE** is not directly comparable (different depth grids and profile populations).

## Agent dual-run workflow

```bash
cd NeSPReSO2_onTemplate
RUN_ID=$(date +%Y%m%d_%H%M%S)
mkdir -p saved/runs/$RUN_ID
# write manifest before launch (agent fills pids after background start)
cat > saved/runs/$RUN_ID/manifest.json <<EOF
{"run_id":"$RUN_ID","runs":[
  {"tag":"isas20","config":"config_isas.json","pid":null},
  {"tag":"argo_v2","config":"config_argo.json","pid":null}
]}
EOF

# parallel if >=2 idle GPUs (ponytail: nvidia-smi util < 10%)
IDLE=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | awk '$1<10{c++} END{print c+0}')
srun --ntasks=1 --cpus-per-task=8 --gres=gpu:1 python3 train.py -c config_isas.json -id ${RUN_ID}_isas20 &
srun --ntasks=1 --cpus-per-task=8 --gres=gpu:1 python3 train.py -c config_argo.json -id ${RUN_ID}_argo_v2 &

python3 scripts/train_monitor.py --once --manifest saved/runs/$RUN_ID/manifest.json
# exit 0=done 1=running 2=failed/stalled; add --kill on exit 2

# when both done:
python3 eval_run.py -c config_isas.json -r saved/models/NeSPReSO2_ISAS_GoM/${RUN_ID}_isas20/model_best.pth --out saved/runs/$RUN_ID/eval_isas.json
python3 eval_run.py -c config_argo.json -r saved/models/NeSPReSO2_ARGO_GoM/${RUN_ID}_argo_v2/model_best.pth --out saved/runs/$RUN_ID/eval_argo.json
```

Stdout sentinels: `NESPRO_TRAIN_EPOCH`, `NESPRO_TRAIN_DONE`, `NESPRO_TRAIN_FAIL`. Per-run `status.json` lives under each `save_dir`.

### GoM eval (Jun 2026, test split, `eval_run.py`)

| Model | config | temp RMSE | sal RMSE |
|---|---|---:|---:|
| PatchConvMLP 16-PC ISAS patch | `config_isas_patch.json` | 1.016 | **5.32** |
| PredictionModel 15-PC ISAS point | `config_isas.json` | **1.002** | 5.53 |
| PatchConvMLP 16-PC ARGO point | `config_argo.json` | **0.416** | **0.072** |

ISAS bases differ (16 vs 15 PCs); not strictly apples-to-apples. JSON: `saved/eval_*_test.json`.

**TensorBoard** (enabled in GoM configs): `tensorboard --logdir saved/log --port 6006`. Past runs with `tensorboard: false` only have `info.log` — no event files.

## ML optimization benchmarks

NeSPReSO trains a small MLP (`9→512→512→30`, ~283K params), not an LLM. Many GPU tricks that help transformers are **testable but rarely faster** here because epochs are short (~6–7 batches) and PCA loss matmuls dominate.

### Applicability

| Optimization | Testable? | Result on this repo |
|---|---|---|
| float32 (baseline) | Yes | Fastest on ISAS; default |
| bfloat16 / float16 (`autocast`) | Yes | ~0.82–0.89× (slower) |
| `torch.set_float32_matmul_precision` | Yes | ~0.91–0.96× |
| `torch.compile` | Yes | ~0.86–0.88× |
| Fused Adam | Yes | ~0.94–1.01× |
| `cudnn.benchmark` | Yes | ~0.89–1.00× |
| DDP (2 GPU) | Yes | ~1.39× faster per epoch (ISAS smoke; each rank sees half the batches) |
| Flash attention | No | No attention layers |
| Pad to power-of-2 | Marginal | `batch=512`, `hidden=512` already Po2 |
| Model/tensor parallelism, ZeRO | No | Model fits in KB of VRAM |
| Quantization, KV-caching | No | Inference/LLM only |

### Run benchmarks

```bash
cd NeSPReSO2_onTemplate

# Full single-GPU sweep (10 warmup + 100 timed epochs per variant)
srun --ntasks=1 --cpus-per-task=8 --gres=gpu:1 \
  python3 scripts/benchmark_ml_opts.py -c config_isas.json --profile

# Or use the launcher (single-GPU + optional DDP)
bash scripts/launch_benchmark.sh config_isas.json

# DDP only (2 GPUs)
srun --ntasks=1 --cpus-per-task=8 --gres=gpu:2 \
  torchrun --nproc_per_node=2 scripts/benchmark_ml_opts.py \
  -c config_isas.json --variant ddp
```

Results JSON: `saved/benchmarks/ml_opts_*.json`.

### Measured speedups (Jun 2026, `bfs-v13-skynet`, 100 timed epochs)

**ISAS (`config_isas.json`)** — baseline `0.0256 s/epoch`:

| variant | sec/epoch | speedup | val_loss |
|---|---:|---:|---:|
| baseline | 0.0256 | 1.00× | 138.39 |
| fused_adam | 0.0272 | 0.94× | 138.47 |
| matmul_highest | 0.0278 | 0.92× | 138.39 |
| autocast_bf16 | 0.0312 | 0.82× | 138.14 |
| combo_best | 0.0293 | 0.87× | 138.49 |

**ARGO (`config_argo.json`)** — baseline `0.0214 s/epoch` (deeper PCA loss, 1801 levels):

| variant | sec/epoch | speedup | val_loss |
|---|---:|---:|---:|
| baseline | 0.0214 | 1.00× | 0.776 |
| matmul_highest | 0.0212 | 1.01× | 0.776 |
| fused_adam | 0.0212 | 1.01× | 0.767 |
| autocast_bf16 | 0.0241 | 0.89× | 0.753 |
| compile_model | 0.0243 | 0.88× | 0.734 |
| compile_loss | 0.0226 | 1.02× | 0.137 |
| compile_both | 0.0224 | 1.03× | 0.139 |

**Phase 4b verdict (Jun 2026, GoM ARGO):** All three levers tested — max batch (~4%), `compile_loss` (~2%), `pred_profile_cached` (~2%) — **below the 10% full-step bar**. Combined stack (`combo_phase4b_all`: max batch + compile model+loss + `pred_profile_cached`) is **~10% slower** than baseline (0.0272 vs 0.0246 s/epoch) — compile overhead dominates at 1 batch/epoch. GoM ARGO is fast enough; reserve loss optimizations for global scale. Config defaults unchanged (`batch_size: 512`, `loss_config.mode: combined`).

Optional faster loss (same objective as profile MSE branch):

```json
"loss_config": { "mode": "pred_profile_cached" }
```

**Phase 5 (in progress):** learned profile decoders — see [PLAN-phase5.md](../PLAN-phase5.md). Stage A: `scripts/train_profile_ae.py`; dim sweep: `scripts/benchmark_profile_ae_dims.py` (dims 16–256 vs PCA-X). ISAS salinity AE beats PCA at every dim (best 0.202 @ dim 128); ARGO still PCA-dominant at 200 epochs.

**DDP (2 GPU, ISAS smoke, 20 timed epochs):** `0.0184 s/epoch` vs `0.0256` 1-GPU baseline — **~1.39× faster** per epoch (each rank processes half the batches).

## Dissertation data foundation

See [`../PLAN-dissertation-data-foundation.md`](../PLAN-dissertation-data-foundation.md) and [`../HANDOFF.md`](../HANDOFF.md).

```bash
# Data census + split design (writes ../reports/)
srun --ntasks=1 --cpus-per-task=8 python3 scripts/data_census.py -c config_argo.json

# L3 download scaffolding (requires copernicusmarine / podaac / cdsapi credentials)
python3 scripts/download_l3_products.py --product all_scaffold
```

ARGO configs use `split_mode: chronological`. Explicit date ranges: `config_argo_chrono_dates.json`.

## Comparison notebook

[`notebooks/compare_v2_vs_template.ipynb`](notebooks/compare_v2_vs_template.ipynb) is the interactive comparison surface for:

- **ISAS20** vs **ARGO** (`isas_point`, `isas_patch`, `argo_point` inline configs)
- **PCA vs AE** profile reconstruction on the test split
- Smoke **training** (2 epochs) and **inference** via `train.main` / `eval_run.main`
- Depth curves and spatial maps on a **common 0–1800 m grid** (10 m steps)

All statistics are defined in [`notebooks/nb_metrics.py`](notebooks/nb_metrics.py) (`STATISTICS` contract). Summary tables use `raw_profile_rmse_common`; native-grid RMSE matches `eval_run.py`.

```bash
cd NeSPReSO2_onTemplate
jupyter notebook notebooks/compare_v2_vs_template.ipynb

# headless smoke (PCA/AE repr + cache stats; inference if notebook checkpoints exist)
srun --ntasks=1 --cpus-per-task=8 --gres=gpu:1 python3 notebooks/run_compare.py
```

Regenerate the notebook from [`notebooks/build_notebook.py`](notebooks/build_notebook.py) after editing cell sources there.

Profiler (ISAS baseline): top CUDA time in `Adam.step` and `aten::mm` (PCA reconstruction), not the MLP forward.

**Recommendation:** keep FP32 defaults. Optional `performance` block in config if you want to experiment:

```json
"performance": {
  "cudnn_benchmark": false,
  "matmul_precision": null,
  "autocast": false,
  "autocast_dtype": "bfloat16",
  "compile": false,
  "compile_loss": false,
  "fused_optimizer": false
}
```
