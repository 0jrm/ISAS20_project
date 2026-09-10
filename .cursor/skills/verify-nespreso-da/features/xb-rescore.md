# xb rescore

xb rescore lets a user score already-written NeSPReSO profiles on inov z against Argo / xb / xa and persist `stats.json`, including the 2024 era slice, without a new GPU predict.

## Sub-features

- `score-mode` runs `--mode score` into a disposable `--out`.
- `hash-stats` prints sha256 of `stats.json` via `offline_asserts.py --stats`.
- `era-2024` reads `slices.2024` (autopsy of the n=166 tell).

## How to get to it (user POV)

- From `NeSPReSO2_onTemplate/`, run `python3 scripts/compare_xb_argo.py --mode score --nc <profiles_xb.nc> --out <dir>` when `<dir>/profiles_nes.nc` and `<dir>/sat_scalars.npz` already exist.
- Hash with `offline_asserts.py --stats <dir>/stats.json`.

## Driving it with control-nespreso-da

Preconditions:

- `control-nespreso-da doctor` exits 0.
- `profiles_xb.nc` exists at the repo root.
- `reports/xb_argo_compare/profiles_nes.nc` and `sat_scalars.npz` exist. If either is missing, skip and name the path. Do not run `--mode all`.
- Disposable `--out` directory. Never `reports/xb_argo_compare`.
- `NESPRO_VERIFY_ID` is set.
- No GPU for `--mode score`.

- **Stage inputs.** Copy `reports/xb_argo_compare/{profiles_nes.nc,sat_scalars.npz}` into `/tmp/nespro-verify-da-$NESPRO_VERIFY_ID/` (read-only source, write only the tmp dir).
- **Score.** Run `control-nespreso-da cli -- scripts/compare_xb_argo.py --mode score --nc "$REPO/profiles_xb.nc" --out /tmp/nespro-verify-da-$NESPRO_VERIFY_ID`. Exit code `0`. stdout contains `scored `.
- **Hash.** Run `control-nespreso-da cli -- "$REPO/.cursor/skills/verify-nespreso-da/scripts/offline_asserts.py" --stats /tmp/nespro-verify-da-$NESPRO_VERIFY_ID/stats.json --model A_CRPS_z32`. Exit code `0`. stdout JSON contains `sha256` and `"n": 6780` on the full file (or the n of the staged copy).
- **2024 slice.** `python3 -c` reading the disposable `stats.json` must show `slices["2024"]["n"]` and both `slices["2024"]["xb"]["T"]["rmse"]` and `slices["2024"]["models"]["A_CRPS_z32"]["T"]["rmse"]`. Copy that JSON fragment into `artifacts/$NESPRO_VERIFY_ID/era_2024.json`.
- **Proof.** `artifacts/$NESPRO_VERIFY_ID/` holds `stats.json` copy, hash stdout, and `exit_code` 0. Production `reports/xb_argo_compare/stats.json` byte contents are unchanged if you never passed that path as `--out` (the script still rewrites `reports/eval_xb_argo.md`; see gotchas).

## Gotchas

- `--mode score` and `--from-stats` rewrite `$REPO/reports/eval_xb_argo.md` unconditionally. Treat that rewrite as pollution. Prefer proving hash + 2024 keys from a **copy** of the existing `reports/xb_argo_compare/stats.json` when you only need the autopsy, and skip `--mode score`.
- Autopsy-only drive. `control-nespreso-da cli -- "$REPO/.cursor/skills/verify-nespreso-da/scripts/offline_asserts.py" --stats "$REPO/reports/xb_argo_compare/stats.json" --model A_CRPS_z32` plus reading `slices.2024`. That does not rewrite figures. It is valid proof of `hash-stats` and `era-2024` when `--mode score` would clobber the markdown.
- `--plots-only` needs the same staged nc/npz pair and also skips new predict.
- Do not compare these RMSE numbers to `eval_run.py` on the 2015–2022 cache. Different era, grid, and truth.
