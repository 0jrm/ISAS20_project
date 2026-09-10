# Pipeline guards

Pipeline guards separate a scientific DA result from a plumbing artifact. The OSSE toy is driveable. TSIS OSE asserts are skip until cycle dumps exist.

## Sub-features

- `osse-selfcheck` runs `run_osse.py --selfcheck` (column OI shapes, not HYCOM).
- `ose-three-point` is the Argo-as-nes / climatology / R→∞ ladder. Skip without a TSIS cycle driver.
- `ose-leakage` requires withheld Argo ⊄ pair neighbors. Offline `build_xb_pairs` already refuses `dt_days <= 0`.
- `ose-innovation` matches TSIS O−B to compare_xb_argo residuals. Skip without OmB files.

## How to get to it (user POV)

- OSSE toy. From `NeSPReSO2_onTemplate/`, `python3 scripts/run_osse.py --selfcheck`.
- OSE ladder. There is no CLI in this repo. Do not fake it with `eval_run.py`.

## Driving it with control-nespreso-da

Preconditions:

- `control-nespreso-da doctor` exits 0.
- `NESPRO_VERIFY_ID` is set.
- No GPU for `--selfcheck`.
- OSE sub-features need a TSIS cycle working directory the user names. If unset, report skip. Do not invent paths.

- **OSSE toy.** Run `control-nespreso-da cli -- scripts/run_osse.py --selfcheck`. Exit code `0`. stdout contains `run_osse selfcheck OK`.
- **Three-point.** Skip unless a cycle script exists. When it does, (a) withheld Argo fed as NeSPReSO reproduces the ARGO arm within the cycle's documented tolerance, (b) climatology-as-NeSPReSO degrades toward climatology, (c) R→∞ equals CTRL bitwise. Failure of (a) is operator/interp. Failure of (c) is a leak.
- **Proof for this checkout.** OSSE selfcheck transcript only. Record skipped OSE bullets with the missing artifact name (`TSIS obs log`, cycle dump, OmB/OmA).

## Gotchas

- July cast-column OSSE already failed E3>E2 and E4≥E3. Re-running `--selfcheck` does not revive those claims.
- Diagonal R is the default after that OSSE. Do not re-enable CRPS-head full Σ because this skill mentions C1.
- `run_osse.py` is not TSIS. Innovation consistency against `compare_xb_argo` cannot be proven from OSSE JSON.
- Free-run vs reanalysis (experiment D) has no entry point here. Skip.
