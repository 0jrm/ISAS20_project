# Error blend

Error blend lets a user measure ρ(e_nes, e_xb) and the optimal blend weight w on T residuals, first on a 2-level toy, then on live `profiles_nes.nc` + `profiles_xb.nc`. P2 lives or dies on that ρ.

## Sub-features

- `blend-toy` runs uncorrelated-vs-perfectly-correlated asserts.
- `blend-live` prints nested JSON: `bands` is the `all` slice; `slices` has per-slice per-band n, rho, rmse, bias, debiased RMSE, w_nes, rmse_blend; plus `desroziers`, `dof`, `in_file_argo_age_km100`, `autopsy_2024`, `complementarity_50_200`, `estimator_blend_hits`.

## How to get to it (user POV)

- Toy. `python3 .cursor/skills/verify-nespreso-da/scripts/offline_asserts.py --selfcheck`
- Live. `python3 .cursor/skills/verify-nespreso-da/scripts/offline_asserts.py --xb profiles_xb.nc --nes reports/xb_argo_compare/profiles_nes.nc --model A_CRPS_z32`

## Driving it with control-nespreso-da

Preconditions:

- `control-nespreso-da doctor` exits 0.
- `NESPRO_VERIFY_ID` is set.
- No GPU.
- Live drive also needs `profiles_xb.nc` and `reports/xb_argo_compare/profiles_nes.nc`. If either is missing, stop after `blend-toy` and report the skip.

- **Toy.** Run `control-nespreso-da cli -- "$REPO/.cursor/skills/verify-nespreso-da/scripts/offline_asserts.py" --selfcheck`. Exit code `0`. stdout contains `selfcheck: offline_asserts ok`.
- **Live.** Run `control-nespreso-da cli -- "$REPO/.cursor/skills/verify-nespreso-da/scripts/offline_asserts.py" --xb "$REPO/profiles_xb.nc" --nes "$REPO/reports/xb_argo_compare/profiles_nes.nc" --model A_CRPS_z32`. Exit code `0`. stdout is JSON with `bands["50-200"]` keys `rho`, `w_nes`, `rmse_blend`, `rmse_xb`. Same stdout also has `slices`, `desroziers`, `dof`, `in_file_argo_age_km100`, `complementarity_50_200`, `estimator_blend_hits`.
- **Decision.** Copy stdout to `artifacts/$NESPRO_VERIFY_ID/blend.json`. Read `complementarity_50_200` first. `all_high` means little complementary information at floats. `low_le_0p4` names complementarity slices, not a TSIS win. `w_nes` / `rmse_blend` / `estimator_blend_hits` are linear-estimator diagnostics. Beating xb RMSE is neither necessary nor sufficient for DA value. Do not start or skip an OSE from that comparison.
- **Proof.** `exit_code` is `0` and the JSON parses. Toy success is not live ρ.

## Gotchas

- Live load reads full T cubes. Keep `NESPRO_VERIFY_CPUS` at 8. Do not add a GPU wrap.
- `w_nes` is clipped to [0, 1]. A negative unconstrained weight means nes is dominated; the clip reports 0.
- Slice ρ (all, 2024, 2025, in_bbox, ood, lc) is in this helper under `slices`. Read those keys; do not re-pool by hand.
- Pair-track models (`stoch_eof_pair`) have a different valid-cast mask. Pass `--model` explicitly.
