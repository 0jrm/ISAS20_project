# Hypotheses and experiment order

Drive cheapest CPU diagnostics before any TSIS cycle. A skipped arm with a named missing file is honest. A proxy from `eval_run.py` is not.

**"Beats xb at the Argo point" is neither necessary nor sufficient for DA value.** The xb table scores NeSPReSO as an estimator. Assimilation cares about it as an information source (error independence, structure, coverage). `estimator_blend_hits` and blend RMSE are estimator diagnostics. They do not start or skip an OSE.

## Why the table is stacked

NeSPReSO's inputs (SST, SSS, SSH) are somewhat already assimilated into HYCOM (since we're looking at a reanalysis). So a NeSPReSO cast is not a new measurement — it's a **nonlinear observation operator applied to data the system already saw**, plus a learned prior on vertical structure. The only new information is that prior. This has two consequences that shape everything below:

- NeSPReSO's errors and xb's errors are likely **correlated** (both cold-biased at 50–200 m in your table; both see the same altimetry). Standard DA assumes obs and background errors are independent. Violating this produces over-confident analyses — the ISOP/synthetic-profile literature explicitly warns that synthetic-profile assimilation can introduce representativeness errors, damp information from real profiles, and lead to undesirable model self-confirmation effects.
- The comparison in your table is **stacked against NeSPReSO**: xb at an Argo location is a forecast from an analysis that assimilated that same float 1–10 days earlier. xb is locally "pre-trained" on the verification data. NeSPReSO's value, if any, lives where no float has been recently.

## Steelmanned hypotheses

### Why a *worse* measurement can still be a *good* assimilation target

**P1 — Coverage beats accuracy.** Forecast error is dominated by unconstrained regions, not by error at float locations. Thousands of daily casts with 1.1 °C error may beat ~30 floats with 0 error if xb drifts to 1.5 °C between float visits. The 2024 subset (n=166) is the tell: **every NeSPReSO cell beats xb there** (0.67–0.82 vs 0.90). Something about those casts — likely a sparse-Argo period — made xb worse than the satellite-informed prior. Find out what.

**P2 — Complementary errors.** Two estimators with variances σ₁², σ₂² and error correlation ρ combine to something better than either if ρ is low. At 50–200 m, xb=0.946, A_CRPS_z32=1.139. If ρ ≈ 0.3, the optimal blend is ≈0.83 — a win *as an estimator*. If ρ ≈ 0.8, no blend helps *at those points*. Compute ρ from `profiles_nes.nc` + `profiles_xb.nc`. §5: ρ ≳ 0.7 everywhere means the information is not at the floats; ρ ≲ 0.4 in a regime is the paper-shaped complementarity. Neither result is a TSIS skill claim.

**P3 — Shape vs heave partition.** xb gets heave (isotherm depth) from altimetry but may get thermocline *shape* wrong; NeSPReSO may be the reverse (its D26 errors are ~xb's, but its T RMSE is worse). If the errors partition across shape/heave, a heave-aware or isopycnal-coordinate operator extracts the good part. Your `heave_vs_shape_split` is exactly this — run it on xb too.

**P4 — Bias is fixable, variance isn't.** Cold bias is coherent and can be subtracted; noise can't. Debiased A_CRPS_z32 at 50–200 m: √(1.139² − 0.338²) ≈ 1.09. Still loses to xb's debiased 0.87 pooled — but a per-region, per-depth, per-season bias table may close more of the gap. Check whether the 743 OOD casts and the LC box carry most of the bias.

**P5 — Lead-time value.** xb at cycle 0 is competitive; the question is day 3–7. Argo constraints decay, satellite-derived constraints refresh daily. The value may only appear at lead.

**P6 — Integral quantities are what matter.** For LC forecasting and hurricane intensity, D26/OHC/MLD dominate; pointwise T at 150 m doesn't. A cast that's 1 °C off at every level with the right integral can be a better constraint than one with the right levels and wrong integral. Assimilate the integrals directly (§3B).

### Why a *good* measurement can be a *bad* assimilation target

**C1 — Vertically correlated errors.** NeSPReSO errors are PCA-structured: 16–32 modes generate a 1800-level profile, so errors are nearly perfectly correlated in the vertical. Treating each level as an independent observation with diagonal R hands TSIS ~1800 pseudo-independent votes per cast. This is the mechanism that likely broke the OSSE, not the σ values per se.

**C2 — Smoothness kills stratification.** HeaveFast's N² fail rate (0.012 vs xb 0.0016) is the smoking gun. PCA-truncated profiles are smooth; assimilated into HYCOM's hybrid layers, smooth T/S increments at fixed z become layer-thickness changes that flatten the thermocline. The Navy hit exactly this: while minimizing T/S errors, vertical gradients tend to remain overly smooth and unrealistic, which changes acoustic and current predictions. NeSPReSO's low pooled RMSE is partly *because* it's smooth — the same property that makes it dangerous to assimilate.

**C3 — Double-counting the satellite.** TSIS already projects SSH to depth via its own covariance. Assimilating NeSPReSO adds a second projection of the same SSH innovation. Where the two agree, the analysis over-corrects; where they disagree, they fight. Neither is good.

**C4 — Uncalibrated, spatially clustered errors.** ENCE 0.24–0.80, cell-Spearman ≈0, and 2× worse error east of 82.67°W. Without a trustworthy R, DA weights bad casts most where they are worst. Representation error in profile DA is known to be much larger than instrument error and higher at depths of high vertical gradients — exactly the thermocline band where NeSPReSO already loses.

**C5 — Coherent bias injection.** A −0.4 °C bias on 30 Argo casts averages out spatially. A −0.4 °C bias on 5000 daily NeSPReSO casts is a coherent forcing that walks HYCOM's OHC downward every cycle. Density amplifies bias exactly as it dilutes noise.

**C6 — Dynamical imbalance.** T/S increments not consistent with the velocity/SSH field trigger adjustment; the forecast rejects or radiates the information as gravity waves. In the LC front region this is worst.

## Experiments (cheapest first)

### A. Offline diagnostics (CPU, extends compare_xb_argo.py)

Driveable now. See [error-blend.md](features/error-blend.md), [heave-shape.md](features/heave-shape.md), [xb-rescore.md](features/xb-rescore.md).

1. Error cross-correlation table. Driveable via `offline_asserts.py --xb --nes`. ρ and blend by band × slice (`all`, `2024`, `2025`, `in_bbox`, `ood`, `lc`). Read `complementarity_50_200` first: `all_high` means little complementary information *at floats*; `low_le_0p4` names paper-shaped slices, not a TSIS win. `estimator_blend_hits` (w>0.3 and blend below xb) is estimator-only.
2. Argo-staleness. TSIS obs log is still **skip**. In-file 100 km proxy is live JSON key `in_file_argo_age_km100` (not a substitute).
3. Autopsy of 2024. `offline_asserts.py --stats` dumps `n`, xb T RMSE, model T RMSE, and heave_vs_shape for slices `all`/`2024`/`ood`/`lc`. Live JSON also has `autopsy_2024`.
4. Vertical error covariance / effective DOF. Live JSON key `dof` (nes and xb per band). Rank hint for NES-thin / PC-space R. Not a CRPS-head R and not ingest of sample Σ at inov z.
5. Conditional skill map. `ood`/`lc`/`in_bbox` slices are in live `slices` and in `stats.json`. Front/MLD/MUR still **skip**. A trust mask is not "NES RMSE < xb RMSE".
6. Desroziers from `T_argo`/`T_xb`/`T_xa` in `profiles_xb.nc`. Live JSON key `desroziers`. Skip only NES-vs-TSIS innovation until cycle OmB files exist.

### B–D. OSEs, forecast, free-run

Not in this checkout as a user CLI. [pipeline-guards.md](features/pipeline-guards.md) lists the asserts that must exist before a cycle is called a result. Cast-column OSSE plumbing is `scripts/run_osse.py --selfcheck` only. It is not a TSIS OSE.

If an OSE runs: **NES-int** and **NES-R + mask + debias** first; **NES-full** diagonal R is the expected-fail control (C1+C2+C5). Verify vs CTRL, bootstrap over cycles. Forecast skill at lead *k* in regime *X* is the claim, not analysis T RMSE at floats.

## Pipeline asserts (separate science from plumbing)

- **Three-point sanity ladder.** (a) Feed the withheld Argo *as if* it were NeSPReSO → must reproduce the ARGO arm within tolerance. (b) Feed climatology as NeSPReSO → must degrade toward climatology. (c) NeSPReSO with R→∞ → must equal CTRL bitwise. If (a) fails, the operator/interp is broken; if (c) fails, something leaks.
- **Innovation consistency.** TSIS O−B for nes casts must match compare_xb_argo residual stats (mean, std, per band) within a few %. If not, the operator or QC differs between offline scoring and assimilation.
- **Increment physics.** Count unstable layers pre/post analysis; OHC of the increment field integrates to the sum of cast-level OHC innovations × gain (order of magnitude); no layer collapses below minimum thickness.
- **Leakage guard.** In OSE arms, no NeSPReSO cast may have used a withheld Argo as pair neighbor; assert dt>0 and neighbor ∉ withheld set. Offline pair construction already raises if `dt_days <= 0` (`build_xb_pairs`).
- **Blend check.** Offline optimal-blend RMSE (§A.1) should approximately predict the single-cycle analysis error of NES-R at the Argo points. Large disagreement means R or B is mis-specified.
- **Regression.** Rescore from profiles_nes.nc, hash stats.json; every new metric gets a 2-level toy assert as in `offline_asserts.py --selfcheck` and `compare_xb_argo.py --selfcheck`.
