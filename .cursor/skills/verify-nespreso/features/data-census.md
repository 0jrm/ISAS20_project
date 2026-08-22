# Data census

Data census lets a user write ARGO profile-count and chronological-split design reports from a config JSON, without training.

## Sub-features

- `census-json` writes `data_census.json` and `split_design.json`.
- `census-md` writes the markdown companions.
- `census-default` prints the recommended dissertation split on stdout.

## How to get to it (user POV)

- From `NeSPReSO2_onTemplate/`, run `srun --ntasks=1 --cpus-per-task=8 python3 scripts/data_census.py -c config/argo/config_argo.json`.
- Override output with `--reports-dir <dir>`.

## Driving it with control-nespreso

Preconditions:

- `control-nespreso doctor` exits 0.
- The ARGO pickle path in `config/argo/config_argo.json` (`io.v2_pickle`) is readable. If not, skip and name the missing path.
- `NESPRO_VERIFY_ID` is set.
- `--reports-dir` is a disposable directory, not `../reports`.

- **Census.** Run `control-nespreso cli -- scripts/data_census.py -c config/argo/config_argo.json --reports-dir /tmp/nespro-verify-census-$NESPRO_VERIFY_ID`. Exit code `0`. stdout contains `census:` and `split:` paths plus `default split:`.
- **Files.** `/tmp/nespro-verify-census-$NESPRO_VERIFY_ID/data_census.json` and `split_design.json` exist. Copy both into `artifacts/$NESPRO_VERIFY_ID/`.
- **Proof.** Re-read `split_design.json` and confirm `recommendation.default_dissertation_split` is present. Then delete only the `/tmp/nespro-verify-census-*` directory, not `artifacts/`.

## Gotchas

- Default `--reports-dir` is `../reports` (repo `reports/`). Driving without an override overwrites shared dissertation reports. Always pass a `/tmp/nespro-verify-census-*` dir.
- Census reads the v2 pickle; it is not a synthetic selfcheck. Missing pickle is skip, not a code-failure proof.
- Do not treat census markdown as a train or eval result.
