# Matched eval

Matched eval lets a user compare ISAS and ARGO models on a shared depth grid at colocated sites, which is the only valid cross-tag RMSE.

## Sub-features

- `matched-run` runs `eval_matched.py` with both configs and both checkpoints.
- `matched-out` writes the balanced JSON report.
- `matched-notes` keeps the native `eval_run.py` files out of the headline.

## How to get to it (user POV)

- From `NeSPReSO2_onTemplate/`, run `python3 eval_matched.py --isas-config config/isas/config_isas.json --isas-checkpoint <isas.pth> --argo-config config/argo/config_argo.json --argo-checkpoint <argo.pth> --out <json>`.

## Driving it with control-nespreso

Preconditions:

- `control-nespreso doctor` exits 0.
- Both checkpoints exist and each matches its own cache/PCA. If either is missing, skip and name the paths.
- `NESPRO_VERIFY_ID` is set.

- **Balanced compare.** Run `control-nespreso cli --gpu -- eval_matched.py --isas-config config/isas/config_isas.json --isas-checkpoint <isas.pth> --argo-config config/argo/config_argo.json --argo-checkpoint <argo.pth> --out /tmp/nespro-verify-matched-$NESPRO_VERIFY_ID.json`. Exit code `0`.
- **Report shape.** Copy the `--out` file into `artifacts/$NESPRO_VERIFY_ID/eval_matched.json`. JSON contains `"eval_type": "balanced_apples_to_apples"` plus `A_shared_sites_balanced_truth` and `B_pooled_holdout_50_50`.
- **Proof.** A second read of the copied JSON still has both A and B blocks. Native `eval_*.json` files are not used as the comparison.

## Gotchas

- Raw `eval_run.py` RMSE across tags is invalid proof of this feature even if both evals succeeded.
- Depth grids differ (ISAS 187 levels vs ARGO 0–1800 m). Matched eval interpolates to 0–1800 m at 10 m; do not compare native grids by eye.
- `--dt-max-days` defaults to `1.0`. Changing it changes the match set; record the flag in `command.txt`.
