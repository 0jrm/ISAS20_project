# Heave vs shape

Heave vs shape lets a user see how much 50–200 m T RMSE is an isotherm-depth error versus a thermocline-shape error, for xb and for a NeSPReSO cell, from `stats.json` or from `evalphys` toys.

## Sub-features

- `heave-evalphys` runs the frozen metric test.
- `heave-stats` requires `heave_vs_shape` on xb and on `A_CRPS_z32` in `stats.json`.

## How to get to it (user POV)

- From `NeSPReSO2_onTemplate/`, run `python3 -m pytest evalphys/tests/test_evalphys.py::test_max_n2_and_heave_split`.
- From scored output, read `slices.all.xb.heave_vs_shape` and `slices.all.models.<name>.heave_vs_shape`.

## Driving it with control-nespreso-da

Preconditions:

- `control-nespreso-da doctor` exits 0.
- `NESPRO_VERIFY_ID` is set.
- No GPU.
- `heave-stats` needs `reports/xb_argo_compare/stats.json`. Skip with that path if missing.

- **Metric pin.** Run `control-nespreso-da cli -- -m pytest evalphys/tests/test_evalphys.py::test_max_n2_and_heave_split`. Exit code `0`.
- **Stats keys.** Run `control-nespreso-da cli -- "$REPO/.cursor/skills/verify-nespreso-da/scripts/offline_asserts.py" --stats "$REPO/reports/xb_argo_compare/stats.json" --model A_CRPS_z32`. Exit code `0`. stdout JSON keeps `sha256` and prints `n`, xb T RMSE, model T RMSE, and `heave_vs_shape` dicts for slices `all`/`2024`/`ood`/`lc` when present.
- **Read the split.** Copy those heave dicts (or the same keys from `stats.json`) into `artifacts/$NESPRO_VERIFY_ID/heave_vs_shape.json`. Record `heave_fraction`, `rmse_50_200`, and `rmse_50_200_heave_aligned` for `xb` and for `A_CRPS_z32`. P3 is supported only if the two `heave_fraction` values differ in a way that leaves complementary residuals. Matching fractions with nes still worse after alignment is not a shape win.
- **Proof.** pytest passed and the JSON fragment exists. Do not treat D26 RMSE alone as this feature.

## Gotchas

- `heave_vs_shape_split` shifts pred so D26 matches truth, then rescores 50–200 m. It is not an isopycnal remap.
- xb already has this field in `score_vs_argo`. A missing key means an old `stats.json`, not a missing physical effect.
- `evalphys` version is pinned in `evalphys/METRICS_MANIFEST.json`. A heave number from a different manifest is not this proof.
