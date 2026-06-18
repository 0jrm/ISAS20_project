---
name: nespreso-train-eval
description: Train, cache-build, eval, and agent-monitor NeSPReSO dual-dataset runs (isas20 vs argo_v2). Use when running train.py, eval_run.py, building caches, dual GPU runs, or train_monitor.py.
---

# NeSPReSO train & eval

## Before training

1. Read [`HANDOFF.md`](../../../HANDOFF.md) and [`PLAN-dissertation-data-foundation.md`](../../../PLAN-dissertation-data-foundation.md).
2. Edit machine paths in config JSON (`io.data_path`, `io.v2_pickle`, `io.v2_src`) — never hardcode in Python.
3. Run data census if splits are uncertain: `python3 scripts/data_census.py -c config_argo.json`
4. Build caches if missing:

```bash
cd NeSPReSO2_onTemplate
srun --ntasks=1 --cpus-per-task=8 python3 preproc/preproc_isas_sat.py cache config_isas.json --force
srun --ntasks=1 --cpus-per-task=8 python3 preproc/export_v2_cache.py -c config_argo.json --force
```

`train.py` calls `ensure_cache()` automatically; `--force` only when inputs or `config_hash` inputs change.

## Train

```bash
srun --ntasks=1 --cpus-per-task=8 --gres=gpu:1 python3 train.py -c config_isas_patch.json -id my_run_id
srun --ntasks=1 --cpus-per-task=8 --gres=gpu:1 python3 train.py -c config_argo.json -id my_run_id
```

- `batch_size: 0` auto-probes VRAM; GoM often → 1 batch/epoch.
- `config_argo.json` keeps `batch_size: 512` for v2 parity unless user says otherwise.
- Stdout sentinels: `NESPRO_TRAIN_EPOCH`, `NESPRO_TRAIN_DONE`, `NESPRO_TRAIN_FAIL`.
- Status: `{save_dir}/status.json`.

## Dual-run agent workflow

```bash
RUN_ID=$(date +%Y%m%d_%H%M%S)
# manifest under saved/runs/$RUN_ID/ — see README "Agent dual-run workflow"
python3 scripts/train_monitor.py --once --manifest saved/runs/$RUN_ID/manifest.json
# exit 0=done, 1=running, 2=failed/stalled
```

ponytail: parallel launch when ≥2 idle GPUs (`nvidia-smi` util < 10%); upgrade path = Slurm `squeue`.

## Eval

```bash
srun --ntasks=1 --cpus-per-task=8 --gres=gpu:1 \
  python3 eval_run.py -c config_argo.json \
  -r saved/models/NeSPReSO2_ARGO_GoM/my_run_id/model_best.pth \
  --out /tmp/eval.json
```

**Rule:** checkpoint PCA must match cache. `eval_run.py` needs `profiles` in cache.

Cross-tag comparison: `eval_matched.py` — not raw `eval_run.py` across tags.

## Do not

- Use random split for dissertation results (use `split_mode: chronological`).
- Enable `combo_phase4b_all` on GoM (slower).
- Compare ISAS vs ARGO RMSE without `eval_matched.py`.
- Add W&B/MLflow — JSON + `status.json` is enough.
