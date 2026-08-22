# NeSPReSO verification map

This directory is the maintained source for verifying user-facing CLI behavior of NeSPReSO v2. Read the index before driving, then use the matching feature file as the recipe.

## Baseline preconditions

- Repo checkout with `NeSPReSO2_onTemplate/` present.
- Conda env `nespreso` (HPC path typically `/conda/jmiranda/miniconda/envs/nespreso/bin/python3`).
- `control-nespreso` on the invocation path: `.cursor/skills/verify-nespreso/scripts/control-nespreso`.
- `export NESPRO_VERIFY_ID=verify-<unique>` so artifacts do not collide.
- Run `control-nespreso doctor` and require exit 0.
- Never drive a GPU job against another session's `-id` or checkpoint.

## Driving conventions

- Start every recipe from doctor-green unless its preconditions say otherwise.
- Treat commands as literal. Keep `-c`, `-r`, `-id`, `--out`, `--reports-dir` unchanged except for the unique verify id.
- CPU commands: `control-nespreso cli -- <script> ...`
- GPU commands: `control-nespreso cli --gpu -- <script> ...`
- Restore nothing in `saved/` that the user already owns. Verification trains use a unique `-id`. Copy disposable census/eval JSON into `artifacts/` then delete only the `/tmp/nespro-verify-*` copy.

## Proof and skip reporting

- Capture the user command, stdout, stderr, and exit code.
- Mutation proof includes a second read of the written file (`status.json`, eval JSON, census JSON).
- Record the feature ID and entry point with every artifact (`meta.txt` + `command.txt`).
- Report an unreachable path with the attempted command and the unmet precondition (missing cache, missing checkpoint, no idle GPU).
- Do not report a skipped entry point as verified through a different path. In particular, `eval_run.py` success is not `eval_matched.py` proof.

## Feature entry contract

Each feature file starts with an H1 title and one paragraph describing the user-visible behavior. It then uses exactly four H2 sections in this order.

1. `Sub-features` lists short IDs with one line for each behavior.
2. `How to get to it (user POV)` lists every user entry point.
3. `Driving it with control-nespreso` starts with `Preconditions:` and uses labeled bullets that pair each user action with an exact command and observable result.
4. `Gotchas` lists traps that can waste or invalidate a verification run.

Keep implementation details out of the map. Name only user paths, stable handles, required state, commands, and observable proof.

## Features

- [Selfcheck](./selfcheck.md) covers the no-data CPU suite and v2-parity named tests.
- [Train](./train.md) covers single-config training with a unique run id and status sentinels.
- [Eval](./eval.md) covers test-split `eval_run.py` with matching checkpoint and cache.
- [Data census](./data-census.md) covers ARGO census and split-design reports into a disposable directory.
- [Matched eval](./eval-matched.md) covers the only valid ISAS-vs-ARGO RMSE comparison.
