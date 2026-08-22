# Eval

Eval lets a user score a trained checkpoint on a split and write a JSON report of profile RMSE for that dataset tag.

## Sub-features

- `eval-test` runs `eval_run.py` on the test split.
- `eval-out` writes the JSON report to `--out`.
- `eval-pair` refuses (or is invalid proof if forced) a checkpoint paired with a different cache/PCA.

## How to get to it (user POV)

- From `NeSPReSO2_onTemplate/`, run `srun --ntasks=1 --cpus-per-task=8 --gres=gpu:1 python3 eval_run.py -c config/argo/config_argo.json -r saved/models/NeSPReSO2_ARGO_GoM/<run_id>/model_best.pth --out <json>`.
- Pass `--split train|val|test` (default `test`).

## Driving it with control-nespreso

Preconditions:

- `control-nespreso doctor` exits 0.
- Checkpoint `-r` was trained with the same `-c` config / cache (same `dataset_tag` and PCA). If that pair is missing, skip and report the unmet precondition; do not eval a convenient other checkpoint.
- `NESPRO_VERIFY_ID` is set.

- **Test split.** Run `control-nespreso cli --gpu -- eval_run.py -c config/argo/config_argo.json -r saved/models/NeSPReSO2_ARGO_GoM/<run_id>/model_best.pth --out /tmp/nespro-verify-eval-$NESPRO_VERIFY_ID.json`. Exit code `0`. stdout is a JSON object with `raw_profile_rmse` (and `n_samples`).
- **Persisted report.** Copy `/tmp/nespro-verify-eval-$NESPRO_VERIFY_ID.json` into `artifacts/$NESPRO_VERIFY_ID/eval.json`. A second `python3 -c "import json; json.load(open('...'))"` succeeds and `n_samples` > 0.
- **Proof.** `exit_code` is `0` and the copied JSON contains `raw_profile_rmse`. Record the exact `-c` and `-r` in `command.txt`.

## Gotchas

- `eval_run.py` RMSE is within-tag only. Do not compare ISAS and ARGO numbers from this command; that is the matched-eval feature.
- Missing `profiles` in the cache makes eval unusable. That is a skip with the cache path named, not a failed model.
- `--out` is optional; without it the JSON is only on stdout. Verification always passes `--out` so the side effect can be re-read.
- CPU eval is possible but slow; use `--gpu` when a GPU is idle.
