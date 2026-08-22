# Train

Train lets a user fit a model from a config JSON, write checkpoints under a named run id, and observe epoch progress until done or failed.

## Sub-features

- `train-argo` starts `train.py` with `config/argo/config_argo.json` and a unique `-id`.
- `train-status` shows `{save_dir}/status.json` moving through running to done or failed.
- `train-sentinels` emits `NESPRO_TRAIN_EPOCH` / `NESPRO_TRAIN_DONE` / `NESPRO_TRAIN_FAIL` on stdout.

## How to get to it (user POV)

- From `NeSPReSO2_onTemplate/`, run `srun --ntasks=1 --cpus-per-task=8 --gres=gpu:1 python3 train.py -c config/argo/config_argo.json -id <run_id>`.
- Watch `{trainer.save_dir}/models/<name>/<run_id>/status.json`.
- Optionally wrap two configs with `scripts/train_monitor.py --once --manifest saved/runs/<id>/manifest.json`.

## Driving it with control-nespreso

Preconditions:

- `control-nespreso doctor` exits 0.
- An idle GPU exists (`nvidia-smi` utilization under 10%), or skip and report the unmet GPU precondition.
- Cache for that config already exists or `train.py` is allowed to build it (`ensure_cache()`). Building a cache is a long side effect; do not `--force` rebuild a shared cache.
- `NESPRO_VERIFY_ID` is unique and not an existing production run id under `saved/models/`.

- **Start ARGO train.** Run `control-nespreso cli --gpu -- train.py -c config/argo/config_argo.json -id "$NESPRO_VERIFY_ID"`. The process writes `saved/models/NeSPReSO2_ARGO_GoM/$NESPRO_VERIFY_ID/`.
- **Epoch progress.** Read `stdout.log` for `NESPRO_TRAIN_EPOCH` lines and `status.json` for `"state": "running"` plus an increasing `epoch`.
- **Completion.** Wait until stdout contains `NESPRO_TRAIN_DONE` or `NESPRO_TRAIN_FAIL`, or `status.json` `"state"` is `done` or `failed`. Exit code `0` only on done.
- **Proof.** Capture `artifacts/$NESPRO_VERIFY_ID/stdout.log` and a copy of `status.json`. A second read of `status.json` after the process exits still shows the terminal state. `model_best.pth` exists only on a successful train.

## Gotchas

- Never reuse a production `-id`. Colliding ids overwrite `status.json` and checkpoints.
- Do not enable `combo_phase4b_all`, bf16, or `torch.compile` on GoM without a measured ≥10% full-step gain.
- Dissertation ARGO configs must keep `data_loader.split_mode` as `chronological`. A passing train with `random` is not dissertation proof.
- Two `--gpu` trains on one GPU corrupt both runs. If fewer than two idle GPUs, do not start a dual-run.
- `train_monitor.py` requires `--once`. Without `--once` it errors. `--kill` only kills pids listed in the manifest.
