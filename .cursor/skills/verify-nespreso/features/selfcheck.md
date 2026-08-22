# Selfcheck

Selfcheck lets a user prove the template still matches v2 numerics and that split/config guards hold, without reading caches or GPUs.

## Sub-features

- `selfcheck-full` runs every test in `selfcheck.py`.
- `selfcheck-help` lists test names.
- `selfcheck-v2-parity` runs the named PredictionModel / CombinedPCALoss pins.
- `selfcheck-split` runs the chronological-split leakage guard.

## How to get to it (user POV)

- From `NeSPReSO2_onTemplate/`, run `srun --ntasks=1 --cpus-per-task=8 python3 selfcheck.py`.
- Run `python3 selfcheck.py --help` to list tests.
- Run `python3 selfcheck.py test_prediction_model_v2 test_combined_pca_loss_v2` for the v2 pins only.
- Run `python3 selfcheck.py test_chronological_split_no_leakage` for the dissertation split guard.

## Driving it with control-nespreso

Preconditions:

- `control-nespreso doctor` exits 0.
- `NESPRO_VERIFY_ID` is set to a unique value.
- No GPU is required. Do not pass `--gpu`.

- **Help.** List tests. Run `control-nespreso cli -- selfcheck.py --help`. Exit code `0` and stdout contain `selfcheck.py [test_name ...]` plus `test_prediction_model_v2`.
- **V2 parity.** Run the named pins. Run `control-nespreso cli -- selfcheck.py test_prediction_model_v2 test_combined_pca_loss_v2`. Exit code `0`. stdout contains `selfcheck ok test_prediction_model_v2`, `selfcheck ok test_combined_pca_loss_v2`, and `selfcheck: all assertions passed`.
- **Split guard.** Run `control-nespreso cli -- selfcheck.py test_chronological_split_no_leakage`. Exit code `0` and stdout contain `selfcheck ok test_chronological_split_no_leakage`.
- **Full suite.** Run `control-nespreso cli -- selfcheck.py`. Exit code `0` and stdout end with `selfcheck: all assertions passed`. Any `selfcheck FAIL` line is a failed proof.
- **Proof.** Copy `artifacts/$NESPRO_VERIFY_ID/{command.txt,stdout.log,stderr.log,exit_code}` as the record. `exit_code` is `0`. stdout includes the command's selected tests and the all-passed line.

## Gotchas

- `selfcheck.py` forces `CUDA_VISIBLE_DEVICES=""`. A `--gpu` wrap does not make these tests use CUDA.
- Per-test wall is `SELFCHECK_TIMEOUT` seconds (default 120). A hang prints `selfcheck FAIL <name>` then raises.
- Named tests that are not in the `TESTS` tuple exit non-zero with `unknown selfcheck tests:`.
- Passing the full suite is the only proof of `selfcheck-full`. Named-test success does not cover omitted tests.
- Do not import `selfcheck` from a notebook and call one function as a substitute for the CLI.
