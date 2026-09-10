---
name: verify-nespreso-da
description: Drive NeSPReSO vs HYCOM xb / Argo comparison and DA-candidate pipeline asserts on the HPC CLI (compare_xb_argo, residual rho/blend, heave-vs-shape, OSSE selfcheck). Use when proving compare_xb_argo.py, profiles_xb.nc, profiles_nes.nc, stats.json hashing, assimilation-target experiments, or the three-point OSE sanity ladder. Train, eval_run, and census stay in verify-nespreso.
---

# Verify NeSPReSO DA

Offline scoring of NeSPReSO casts against TSIS Argo / xb / xa. The user-facing surface is the CLI under `NeSPReSO2_onTemplate/`, conda env `nespreso`, CPU-capped `srun`. There is no app server.

A NeSPReSO cast is not a new measurement. It is a nonlinear observation operator on SST/SSS/SSH the reanalysis already saw, plus a learned vertical prior. xb at an Argo location is a forecast from an analysis that likely assimilated that same float days earlier. Do not headline pooled RMSE as a DA trophy. Do not treat nes and xb errors as independent without measuring rho.

Train and `eval_run.py` proofs belong in [verify-nespreso](../verify-nespreso/SKILL.md). This skill does not mix a checkpoint with a cache it was not trained on.

Hypotheses, experiment order, and skip rules live in [hypotheses.md](hypotheses.md). Read the matching file in `features/` before a drive.

## Launch

No daemon. Interpreter plus working directory, then each drive in its own `control-nespreso-da` run.

```bash
REPO=<checkout>
CTRL="$REPO/.cursor/skills/verify-nespreso-da/scripts/control-nespreso-da"
chmod +x "$CTRL" "$REPO/.cursor/skills/verify-nespreso/scripts/control-nespreso"
export NESPRO_VERIFY_ID=verify-da-$(date +%Y%m%d_%H%M%S)
"$CTRL" doctor
```

Ready when `doctor` prints `ok compare_xb_argo=...`, the nested `ok python=...` / `ok torch=...` lines, and exits 0.

Default CPU cap is 8 (`NESPRO_VERIFY_CPUS`). If `SLURM_JOB_ID` is set, the helper does not nest `srun`. Set `NESPRO_VERIFY_SRUN=0` to force a local run.

Teardown runs `"$CTRL" cleanup`. It kills the pid this `NESPRO_VERIFY_ID` started, not a process name.

Two full `--mode all` predicts must not share a GPU or the production `--out`. Two `--selfcheck` runs may share a node. Each `cli` invocation overwrites `artifacts/$NESPRO_VERIFY_ID/`. Use a fresh `NESPRO_VERIFY_ID` per drive.

## Doctor

Read-only. Run first whenever anything looks off.

```bash
"$CTRL" doctor
```

Require `compare_xb_argo.py`, `run_osse.py`, `offline_asserts.py`, and the nested verify-nespreso doctor (selfcheck/train/eval files, conda `nespreso` imports). Missing `profiles_xb.nc` or `reports/xb_argo_compare/{profiles_nes.nc,stats.json}` is a warning. Do not drive xb-rescore or live blend until those files exist. `srun` missing is a warning, not a doctor failure. Do not drive if doctor exits non-zero.

## Drive

The harness is `control-nespreso-da`. It sets `NESPRO_VERIFY_ARTIFACTS` to this skill's `artifacts/` and execs `control-nespreso` for `cli` / `cleanup`. Paths after `--` are relative to `NeSPReSO2_onTemplate/` unless absolute.

```bash
"$CTRL" cli -- scripts/compare_xb_argo.py --selfcheck
"$CTRL" cli -- "$REPO/.cursor/skills/verify-nespreso-da/scripts/offline_asserts.py" --selfcheck
"$CTRL" cli -- scripts/run_osse.py --selfcheck
"$CTRL" cli -- scripts/compare_xb_argo.py --mode score --nc "$REPO/profiles_xb.nc" --out /tmp/nespro-verify-da-$NESPRO_VERIFY_ID
```

Never pass the default `--out` (`reports/xb_argo_compare`) on a verify run. `--mode score` and `--from-stats` also rewrite `reports/eval_xb_argo.md`. Copy the disposable `stats.json` into artifacts, then delete only the `/tmp/nespro-verify-da-*` tree.

GPU predict (`--mode predict` / `--mode all`) needs idle GPU and matching checkpoint/cache pairs listed in `compare_xb_argo.py` `MODELS`. If a checkpoint is missing, skip that model. Do not swap caches.

## Evidence

Proof lives under `.cursor/skills/verify-nespreso-da/artifacts/$NESPRO_VERIFY_ID/` (`meta.txt`, `command.txt`, `stdout.log`, `stderr.log`, `exit_code`, `pid`). Gitignored. Cleanup must not delete it.

Standards:

- Drive the real CLI (`scripts/compare_xb_argo.py`, `scripts/run_osse.py`, `offline_asserts.py`), not importing a helper as a substitute for the entry point.
- Capture command, stdout, stderr, exit code.
- Side effects. Disposable `--out/stats.json` and its `sha256` from `offline_asserts.py --stats`. Re-read those files.
- `compare_xb_argo.py --selfcheck` success line is `selfcheck: compare_xb_argo helpers ok` with exit 0.
- `offline_asserts.py --selfcheck` success line is `selfcheck: offline_asserts ok` with exit 0.
- `run_osse.py --selfcheck` prints `run_osse selfcheck OK` with exit 0.
- Mocks only at production boundaries that already isolate (none on the selfcheck path).

## Cleanup

```bash
"$CTRL" cleanup
```

SIGTERM then SIGKILL to the pid recorded for this `NESPRO_VERIFY_ID` only. Leaves `artifacts/` intact. Do not `pkill python`. Do not delete `profiles_xb.nc`, `reports/xb_argo_compare/`, or `saved/`. Disposable `/tmp/nespro-verify-da-*` dirs may be removed after their contents are copied into `artifacts/`.

## Helpers

```bash
.cursor/skills/verify-nespreso-da/scripts/control-nespreso-da doctor
.cursor/skills/verify-nespreso-da/scripts/control-nespreso-da cli [--gpu] [--cpus N] -- <args...>
.cursor/skills/verify-nespreso-da/scripts/control-nespreso-da cleanup
.cursor/skills/verify-nespreso-da/scripts/offline_asserts.py --selfcheck
.cursor/skills/verify-nespreso-da/scripts/offline_asserts.py --stats <stats.json> --model A_CRPS_z32
.cursor/skills/verify-nespreso-da/scripts/offline_asserts.py --xb <profiles_xb.nc> --nes <profiles_nes.nc> --model A_CRPS_z32
```

`offline_asserts.py` is the rho / optimal-blend / stats-hash lever. `--selfcheck` prints `selfcheck: offline_asserts ok`. `--stats` prints `sha256` plus per-slice `n`, T RMSE, and `heave_vs_shape` for `all`/`2024`/`ood`/`lc`. `--xb`+`--nes` prints nested JSON (`model`, `bands`, `slices`, `desroziers`, `dof`, `in_file_argo_age_km100`, `autopsy_2024`, `complementarity_50_200`, `estimator_blend_hits`). Run it through `control-nespreso-da cli` so evidence lands in `artifacts/`.
