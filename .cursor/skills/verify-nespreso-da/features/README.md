# NeSPReSO DA verification map

This directory is the maintained source for verifying user-facing NeSPReSO vs xb / DA-candidate CLI behavior. Read the index before driving, then use the matching feature file as the recipe.

## Baseline preconditions

- Repo checkout with `NeSPReSO2_onTemplate/` present.
- Conda env `nespreso`.
- `control-nespreso-da` at `.cursor/skills/verify-nespreso-da/scripts/control-nespreso-da`.
- `export NESPRO_VERIFY_ID=verify-da-<unique>` so artifacts do not collide.
- Run `control-nespreso-da doctor` and require exit 0.
- Never write verify `--out` to `reports/xb_argo_compare`.
- Train/eval/census recipes stay in `../verify-nespreso/features/`.

## Driving conventions

- Start every recipe from doctor-green unless its preconditions say otherwise.
- Treat commands as literal. Keep `--nc`, `--out`, `--mode`, `--model` unchanged except for the unique verify id and disposable `--out`.
- CPU commands. `control-nespreso-da cli -- <script> ...`
- GPU predict. `control-nespreso-da cli --gpu -- ...` only when the feature says so.
- Copy disposable score JSON into `artifacts/` then delete only the `/tmp/nespro-verify-da-*` copy.

## Proof and skip reporting

- Capture the user command, stdout, stderr, and exit code.
- Mutation proof includes a second read of the written `stats.json` and its sha256.
- Record the feature ID and entry point with every artifact (`meta.txt` + `command.txt`).
- Report an unreachable path with the attempted command and the unmet precondition (missing `profiles_xb.nc`, missing TSIS obs log, missing checkpoint).
- Do not report `eval_run.py` success as xb-compare proof. Do not report `run_osse.py --selfcheck` as a TSIS OSE.

## Feature entry contract

Each feature file starts with an H1 title and one paragraph describing the user-visible behavior. It then uses exactly four H2 sections in this order.

1. `Sub-features` lists short IDs with one line for each behavior.
2. `How to get to it (user POV)` lists every user entry point.
3. `Driving it with control-nespreso-da` starts with `Preconditions:` and uses labeled bullets that pair each user action with an exact command and observable result.
4. `Gotchas` lists traps that can waste or invalidate a verification run.

Keep implementation details out of the map. Name only user paths, stable handles, required state, commands, and observable proof.

## Features

- [Compare helpers](./compare-helpers.md) covers interp/score toys and pair `dt>0`.
- [xb rescore](./xb-rescore.md) covers `--mode score` from existing `profiles_nes.nc` plus the 2024 slice.
- [Error blend](./error-blend.md) covers ρ(e_nes, e_xb) and optimal-blend RMSE (P2).
- [Heave vs shape](./heave-shape.md) covers `heave_vs_shape` on xb and on NeSPReSO (P3).
- [Pipeline guards](./pipeline-guards.md) covers OSSE selfcheck and the OSE asserts that are skip until TSIS dumps exist.
