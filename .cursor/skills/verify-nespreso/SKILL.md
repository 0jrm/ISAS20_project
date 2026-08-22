---
name: verify-nespreso
description: Drive NeSPReSO v2 as a user does — HPC CLI (selfcheck, train, eval, census), not a web UI. Use when proving a code change, checking the conda/srun environment, or capturing a transcript that the real entry points still work.
---

# Verify NeSPReSO

Offline PyTorch batch ML. The user-facing surface is the CLI under `NeSPReSO2_onTemplate/`, run on Skynet with conda env `nespreso` and CPU-capped `srun`. There is no app server. TensorBoard (`tensorboard --logdir saved/log --port 6006`) is a log viewer only.

Never mix a checkpoint with a cache it was not trained on. Never headline raw `eval_run.py` RMSE across `isas20` vs `argo_v2`; that path is `eval_matched.py`.

## Launch

No daemon. Launch means: interpreter + working directory ready, then each drive in its own `control-nespreso` run.

```bash
REPO=<checkout>
CTRL="$REPO/.cursor/skills/verify-nespreso/scripts/control-nespreso"
chmod +x "$CTRL"   # once
export NESPRO_VERIFY_ID=verify-$(date +%Y%m%d_%H%M%S)
"$CTRL" doctor
```

Ready when `doctor` prints `ok python=...`, `ok selfcheck=...`, and `ok torch=...` and exits 0.

Default CPU cap is 8 (`NESPRO_VERIFY_CPUS`). If already inside a Slurm allocation (`SLURM_JOB_ID` set), the helper does **not** nest `srun`. Set `NESPRO_VERIFY_SRUN=0` to force a local run (`OMP_NUM_THREADS` only).

Teardown: `"$CTRL" cleanup` (kills the pid this `NESPRO_VERIFY_ID` started, not by process name).

## Doctor

Read-only. Run first whenever anything looks off.

```bash
"$CTRL" doctor
```

Require: `NeSPReSO2_onTemplate/{selfcheck,train,eval_run}.py` exist; conda env `nespreso` python imports `torch`, `numpy`, `sklearn`; git HEAD printed. `srun` missing is a warning, not a doctor failure. Do not drive if doctor exits non-zero.

## Drive

Harness: `control-nespreso` (repo-local bash). Always `cd` is handled by the helper; pass paths relative to `NeSPReSO2_onTemplate/`.

```bash
"$CTRL" cli -- selfcheck.py
"$CTRL" cli -- selfcheck.py test_prediction_model_v2 test_combined_pca_loss_v2
"$CTRL" cli -- scripts/data_census.py -c config/argo/config_argo.json --reports-dir /tmp/nespro-verify-census-$NESPRO_VERIFY_ID
"$CTRL" cli --gpu -- train.py -c config/argo/config_argo.json -id "$NESPRO_VERIFY_ID"
"$CTRL" cli --gpu -- eval_run.py -c config/argo/config_argo.json -r saved/models/NeSPReSO2_ARGO_GoM/<run_id>/model_best.pth --out /tmp/nespro-verify-eval-$NESPRO_VERIFY_ID.json
```

Read the matching file in `features/` before driving. Prefer named CLI flags and config JSON paths over editing Python.

Two GPU trains must not share a GPU or a `trainer.save_dir` run id. Two selfchecks may run side by side (CPU; `selfcheck.py` forces `CUDA_VISIBLE_DEVICES=""`). Census verification must use a disposable `--reports-dir`, never `../reports`.

## Evidence

Proof lives under `.cursor/skills/verify-nespreso/artifacts/$NESPRO_VERIFY_ID/` (`meta.txt`, `command.txt`, `stdout.log`, `stderr.log`, `exit_code`, `pid`). That directory is gitignored; do not delete it during cleanup.

Standards:

- Drive the real CLI (`selfcheck.py`, `train.py`, `eval_run.py`, `scripts/data_census.py`, `eval_matched.py`), not internal setters or importing a test helper as a substitute for the entry point.
- Capture the command and the resulting stdout/stderr/exit code, not a paraphrase.
- Side effects: `status.json` / `model_best.pth` under the unique `-id` for train; JSON `--out` for eval; census JSON+md under the disposable reports dir. Re-read those files; do not trust the process name or a “dry run” label.
- `selfcheck.py` success line is `selfcheck: all assertions passed` with exit 0. Per-test lines: `selfcheck ok <name>` / `selfcheck FAIL <name>`.
- Train success sentinels: stdout `NESPRO_TRAIN_DONE` and `{save_dir}/status.json` `"state": "done"`. Failure: `NESPRO_TRAIN_FAIL` or status `failed`.
- Mocks only at production boundaries that already isolate (none on the selfcheck path).

## Cleanup

```bash
"$CTRL" cleanup
```

Sends SIGTERM then SIGKILL to the pid recorded for this `NESPRO_VERIFY_ID` only. Leaves `artifacts/` intact. Do not `pkill python` / `pkill srun`. Do not delete the user’s `saved/models/` or `reports/` trees. Disposable census dirs under `/tmp/nespro-verify-*` may be removed after their contents are copied into `artifacts/`.

## Helpers

```bash
.cursor/skills/verify-nespreso/scripts/control-nespreso doctor
.cursor/skills/verify-nespreso/scripts/control-nespreso cli [--gpu] [--cpus N] -- <args...>
.cursor/skills/verify-nespreso/scripts/control-nespreso cleanup
```

The helper resolves repo root from its path, finds conda env `nespreso` python, wraps `srun --ntasks=1 --cpus-per-task=8` when appropriate, and tees evidence into `artifacts/$NESPRO_VERIFY_ID/`.
