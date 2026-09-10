# Compare helpers

Compare helpers let a user prove `compare_xb_argo.py` interpolation, RMSE/bias, 1° binning, OHC, and pair lookup without reading `profiles_xb.nc` or a GPU.

## Sub-features

- `helpers-selfcheck` runs the in-file toy asserts.
- `pair-dt` fails the script if a pair source is not strictly earlier (enforced in `build_xb_pairs` when pairs are built).

## How to get to it (user POV)

- From `NeSPReSO2_onTemplate/`, run `srun --ntasks=1 --cpus-per-task=8 python3 scripts/compare_xb_argo.py --selfcheck`.

## Driving it with control-nespreso-da

Preconditions:

- `control-nespreso-da doctor` exits 0.
- `NESPRO_VERIFY_ID` is set.
- No GPU. Do not pass `--gpu`.
- `profiles_xb.nc` is not required.

- **Helpers.** Run `control-nespreso-da cli -- scripts/compare_xb_argo.py --selfcheck`. Exit code `0`. stdout contains `selfcheck: compare_xb_argo helpers ok`.
- **Proof.** Copy `artifacts/$NESPRO_VERIFY_ID/{command.txt,stdout.log,stderr.log,exit_code}` as the record. `exit_code` is `0`.

## Gotchas

- `--selfcheck` never writes `stats.json` or figures. Success here is not xb-rescore proof.
- Pair `dt>0` during a full predict is a runtime raise, not this toy. This drive only proves the helper that `nearest_eval_pairs` returns `dt_days > 0` on a 3-point fixture.
- Do not import `compare_xb_argo` from a notebook as a substitute for the CLI.
