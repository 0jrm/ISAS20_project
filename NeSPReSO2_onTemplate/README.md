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

Both use `seed=42`, `70/15/15` split, v2-matched trainer settings (`batch=512`, `early_stop=500`).
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
| `data_loader.train_frac` / `val_frac` / `test_frac` | `torch.random_split` fractions (default 0.7/0.15/0.15) |

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

**DDP (2 GPU, ISAS smoke, 20 timed epochs):** `0.0184 s/epoch` vs `0.0256` 1-GPU baseline — **~1.39× faster** per epoch (each rank processes half the batches).

Profiler (ISAS baseline): top CUDA time in `Adam.step` and `aten::mm` (PCA reconstruction), not the MLP forward.

**Recommendation:** keep FP32 defaults. Optional `performance` block in config if you want to experiment:

```json
"performance": {
  "cudnn_benchmark": false,
  "matmul_precision": null,
  "autocast": false,
  "autocast_dtype": "bfloat16",
  "compile": false,
  "fused_optimizer": false
}
```
