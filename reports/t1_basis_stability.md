# Phase 1 decisive tests — T1 basis stability

Cache: `/unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project/data/cache/train_ready_4411c65ee518.pkl`  |  train n=2901  test n=623  |  gsw=`gsw`
Bases fit on: **train_split_only** (leakage check: train only).

| variant | N² prof | N² level | σ₀ level | T/S RMSE by band | dρ/dz | MLD |
|---------|---------|----------|----------|------------------|-------|-----|
| A_separate_pca | 0.7657 | 0.0091 | 0.0097 | T:0-50=0.202,50-200=0.223,200-800=0.107,>800=0.016; S:0-50=0.108,50-200=0.038,200-800=0.020,>800=0.001 | 0.0067 | 36.6067 |
| B_joint_eof | 0.8844 | 0.0092 | 0.0110 | T:0-50=0.227,50-200=0.232,200-800=0.099,>800=0.017; S:0-50=0.087,50-200=0.032,200-800=0.015,>800=0.001 | 0.0066 | 38.6022 |
| C_density_spice_pca | 0.7271 | 0.0089 | 0.0094 | T:0-50=0.181,50-200=0.197,200-800=0.100,>800=0.015; S:0-50=0.115,50-200=0.057,200-800=0.017,>800=0.001 | 0.0068 | 39.4356 |
| D_monotone_density | 0.3868 | 0.0022 | 0.0032 | T:0-50=0.153,50-200=0.156,200-800=0.088,>800=0.016; S:0-50=0.066,50-200=0.056,200-800=0.021,>800=0.002 | 0.0032 | 1.3431 |
| E_softplus_phase3 | 0.3804 | 0.0022 | 0.0032 | T:0-50=0.153,50-200=0.156,200-800=0.088,>800=0.016; S:0-50=0.066,50-200=0.056,200-800=0.021,>800=0.002 | 0.0032 | 1.3431 |

## Plan decision rules (verbatim from PLAN §1-T1)

- If B and/or C cut the level violation rate by ≥ 5× vs A at ≤ 10% RMSE cost ⇒ Finding-1 mechanism confirmed; Phase 3 proceeds as planned.
- If C ≈ A (no improvement) ⇒ the violations are not basis-induced; escalate to human before Phase 3 (the representation chapter framing changes).
- D should show violation rate ≡ 0 by construction; record its RMSE cost — this is the "price of hard stability" headline number.

## Decision outcomes

- ESCALATE: B_joint_eof ≈ A — violations may not be basis-induced under N² level metric
- ESCALATE: C_density_spice_pca ≈ A — violations may not be basis-induced under N² level metric
- D monotone: N² level=0.0022 (σ₀ level=0.0032; pre-inv Δσ₀<0 count=11); vs A N² level=0.0091 (4.1×)
- GATE: B/C did not meet ≥5× level-violation cut under N² — Finding-1 still holds under historical σ₀ profile metric (see Reconciliation).
- Phase3 softplus E vs A T-RMSE: T[0-50] E/A=0.755, T[50-200] E/A=0.700, T[200-800] E/A=0.821, T[>800] E/A=0.970; PASS ≤10%

## Reconciliation (Finding-1 vs T1 N² numbers)

| row | σ₀ profile rate (tol=0.01) | interface rate |
|-----|----------------------------|----------------|
| RAW test | 0.0112 | 0.000013 |
| A PCA-16 | 0.2151 | 0.000821 |
| B joint EOF-32 | 0.2279 | 0.000804 |
| C density+spice | 0.2183 | 0.000748 |
| D monotone | 0.0048 | 0.000003 |
| E softplus Phase-3 | 0.0000 | 0.000000 |

- **(a_profile_vs_level)** N² profile rate ≫ level rate because violations are sparse per profile (A profile=0.7657, level=0.0091)
- **(b_n2_tol0_vs_1e8)** A level N² at tol=0: 0.009157; at 1e-8: 0.009116
- **(d_method)** Historical Finding-1 used readiness σ₀ Δσ₀<-0.01 profile rate (~1.12% raw → ~21.8% PCA-16), not N² level rate.
- **(mechanism_update)** SIGNED OFF 2026-07-16 (human): soft representation changes do not buy stability; only hard monotonicity does. B (joint EOF) and C (density/spice PCA) leave historical σ₀ profile rate ≈ A (~0.22); truncation itself — not T/S separateness — drives the violations. Softplus control-grid (E) zeros σ₀ violations by construction. Dissertation framing: 'no soft basis fixes stability; hard monotonicity does, at cost X.'
- **(hard_stability_win)** Headline: hard constraint improves upper-ocean T RMSE vs A while zeroing σ₀ violations (see Decision outcomes / Big win section). Deep-band cost was a softplus-clamp artefact on non-monotone linear-interp ctrl (~12% negative increments); fixed by isotonic projection before encode.
- **(f_backend)** headline backend=gsw (reference gsw)

## Big win — hard monotonicity (dissertation headline)

Human sign-off (2026-07-16): **no soft basis fixes stability; hard monotonicity does, at cost X.**

| band | A T-RMSE | E T-RMSE | E/A |
|------|----------|----------|-----|
| T[0-50] | 0.2025 | 0.1528 | 0.755 |
| T[50-200] | 0.2227 | 0.1560 | 0.700 |
| T[200-800] | 0.1065 | 0.0875 | 0.821 |
| T[>800] | 0.0162 | 0.0157 | 0.970 |

- Historical σ₀ profile rate: A=0.2151 → E=0.0000
- Phase-3 T-RMSE gate (E/A ≤ 1.10 all bands): PASS

Upper bands (0–800 m) improve under the hard constraint; deep-band regression (if any) is documented in `reports/e_deep_band_diagnostic.md` and fixed by isotonic-before-softplus encode.

## T3 — exclude top 15 m (N² level @ 1e-8)

| variant | full-column | exclude top 15 m |
|---------|-------------|------------------|
| A_separate_pca | 0.0091 | 0.0074 |
| B_joint_eof | 0.0092 | 0.0074 |
| C_density_spice_pca | 0.0089 | 0.0069 |
| D_monotone_density | 0.0022 | 0.0016 |
| E_softplus_phase3 | 0.0022 | 0.0016 |
