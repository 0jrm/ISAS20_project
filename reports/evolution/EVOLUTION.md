# Project evolution — NeSPReSO v2 dissertation branch

**Branch:** `residual_cube` · **Generated:** 2026-07-20, from frozen artifacts only — no model was run, retrained, or re-scored to produce this report.
**HEAD:** `490de67738321368659180f8b173b42fcb22a9c8` · **evalphys manifest sha:** `7f9a043953361931db671903c5bbee2744b7b8fe` (`NeSPReSO2_onTemplate/evalphys/METRICS_MANIFEST.json`, frozen 2026-07-16, evalphys v1.1.0)

> **Warning — sha mismatch.** The evalphys manifest's recorded git sha differs from current HEAD. The manifest was frozen 2026-07-16; the repository has since completed the full Phase 5 ablation matrix and the Phase 6 OSSE (including the R_cal v2 promotion). The metric *definitions* in the manifest are unchanged and govern every number below — only the manifest's own sha pointer is stale. See `reports/evolution/PROVENANCE.json` (`meta.warnings`).
>
> **Warning — stale OSSE run.log.** `NeSPReSO2_onTemplate/saved/runs/phase6_osse/run.log` records E4=2.4186/E5=1.8906 for the cast-column run, which matches neither the v1 diag result (E4=0.5463) nor the current v2 full-localized result (E4=0.6160). The committed `reports/osse_results.md` and `cast_column_s42.json` agree with each other and with git history; the log is treated as a stale/unstable intermediate artifact, not a source for any number in this report. Full detail in `reports/evolution/PROVENANCE.json` (`contradictions[c2]`).

Full machine-readable provenance for every number and figure below: [`PROVENANCE.json`](PROVENANCE.json). Experiment graph: [`lineage.json`](lineage.json). Runnable check: [`check_provenance.py`](check_provenance.py).

---

## 1. Abstract

This branch built a frozen evaluation standard (`evalphys` v1.1.0), used it to run a pre-registered 3-representation × 3-head ablation matrix, and closed the loop with a toy observing-system-simulation-experiment (OSSE). The representation finding is real but narrow: a hard monotone-density constraint removes static-stability violations that no soft basis change touches (`reports/t1_basis_stability.md`). The matrix produced a mechanical default — representation A (separate T/S PCA-16) with a CRPS head — that wins deterministic-adjacent skill and clears the probabilistic calibration gate in PC space, but that same winner fails the calibration gate in physical temperature space overall (ENCE(T)=0.236, `reports/phase5_A_CRPS_physical_strata.md`) and is worst in the surface and thermocline bands. The dissertation's central data-assimilation question — do calibrated ML pseudo-observations improve an analysis — is answered in the negative in the cast-column Gulf-of-Mexico testbed: NeSPReSO casts do not beat an ISOP/MODAS-class synthetic-cast baseline (E3 vs E2: 0.5454 vs 0.5410, FAIL, `reports/osse_results.md`), and a structured (non-diagonal) calibrated observation-error covariance actively degrades the analysis relative to a diagonal one (0.6160 vs 0.5463, `HANDOFF.md`), with the degradation cleanly isolated to basis-induced cross-level correlation in the CRPS head rather than genuine observation-error structure. The datacube built in this branch is a data-quality and extraction-geometry fix, not a modeling win for the patch/residual branch it was meant to serve — that branch lost the matrix.

---

## 2. Timeline

| phase | question | change | verdict | source |
|-------|----------|--------|---------|--------|
| Phase 0 | What metric suite can every later phase trust without redefinition? | Freeze `evalphys` v1.1.0 (N², σ₀-monotonicity, CRPS/ENCE/PIT/spread-skill). | survivor | `NeSPReSO2_onTemplate/evalphys/METRICS_MANIFEST.json` |
| Phase 1 (T2) | Is the chronological test split contaminated by stale (time-constant) satellite inputs? | Stale-fraction audit by split and channel. | survivor (gate OPEN) | `reports/stale_by_split.md` |
| Phase 1 (T1) | Do soft representation changes (joint EOF B, density+spice PCA C) cut static-stability violations ≥5× vs baseline A? | Reconstruct truth through bases A/B/C/D; measure σ₀/N² violation rates. | killed (rule not met; escalated) | `reports/t1_basis_stability.md` |
| Phase 1 (T1→R1) | Does a hard monotone constraint (variant D) do what soft bases cannot? | softplus+cumsum control-grid density head. | survivor (human sign-off R1) | `reports/t1_basis_stability.md`, `PLAN-v2-recovery.md` |
| Phase 3 | Does the hard monotone head recover skill at acceptable cost? | σ₀ control-grid head + spice PCA-16 + Newton (SA,CT) inversion. | survivor | `PLAN-v2-recovery.md` |
| Phase 3 (erratum) | Was the chronological skill gate evaluated cleanly? | Discovered `argo16_scales` is a random-split checkpoint; all chronological evals of it (0.514 gate figure; densonly mse_σ 0.21) were leaked-optimistic. Corrected floor = clean-chrono raw × 1.10. | erratum | `reports/gate_floor_provenance.md` |
| Phase 3 | Does low-rank compression in physical σ₀ space (vs the nonlinear "a-space" preimage) preserve skill? | PCA-16 on `(σ₀-clim)` + isotonic-at-inference. a-space (R=16 recon RMSE 0.925) killed; σ₀-space (RMSE 0.026, T=0.562≤0.590) survived. | survivor (σ₀-space); killed (a-space) | `reports/finding_compress_physical_space.md` |
| Phase 4 | Does a two-stage CRPS/NLL head produce calibrated uncertainty? | Heteroscedastic head; v1 two-stage run ENCE 0.361 MISS; longer stage-2 (s2b) + val per-dim recalibration → ENCE 0.160 PASS. | survivor (s2b) | `reports/phase4_ence_recalib_s2b.md` |
| Phase 5 | Under a fair, locked, 3-seed protocol, which (representation, head) cell wins? | 3×3 matrix, protocol v1 (val-loss early stop) killed by a replicated calibration failure (ENCE 0.225); protocol v2 (val-ENCE early stop) scored all 9 cells. | survivor (v2 matrix) | `reports/ablation_summary.md`, `reports/phase5_C_CRPS.md` |
| Phase 5 | Does the Phase-3 single-run C×det admission pass (T=0.562) replicate at matrix scale? | C×det scored under locked protocol v2, 3 seeds. | killed (0.609±0.012 vs admission 0.562) | `reports/finding_C_det_gate_overfit.md` |
| Phase 5 | Does the latent PC-space ranking survive a like-for-like physical-space decode? | Rescore all 9 cells after decoding to (T,S). | superseded (physical table replaces latent table for cross-rep ranking) | `reports/ablation_summary.md` |
| Phase 5 | Which cell is the dissertation default? | Apply the pre-registered §3 decision rule to the physical table. | survivor: A×CRPS | `reports/ablation_summary.md` |
| Phase 5 | Does the winner's calibration hold across depth band and season? | Strata scoring of A×CRPS in physical T space. | killed (ENCE(T)=0.236 overall; worse in most strata) | `reports/phase5_A_CRPS_physical_strata.md` |
| Phase 6 | Pre-register the OSSE before the matrix winner identity biases the design. | E0–E5 table, R construction, QC rule locked in advance. | survivor | `reports/osse_preregistration.md` |
| Phase 6 | Do NeSPReSO casts beat an ISOP-class baseline, and does calibrated R help? | cast-column v1 OSSE (2021, n=1101), R_cal = diag(Σ_T). | killed (E3>E2 FAIL, E4≥E3 FAIL) | `reports/osse_results.md` (git history @ `28f623e`), `HANDOFF.md` |
| Phase 6 | Does Schur-localizing the full CRPS-head Σ_T restore OI stability while keeping useful structure? | Pre-register full localized Σ_T, `L_loc=150m` (commit `0422a51`). | survivor (prereg) | `reports/osse_preregistration.md` |
| Phase 6 | Does the full localized (structured) Σ_T beat the diagonal fallback? | Canonical run with pre-registered full R_cal (commit `490de67`). | killed (0.6160 vs 0.5463 diag-control) | `reports/osse_results.md`, `HANDOFF.md` |

---

## 3. Lineage narrative

The graph in [`lineage.json`](lineage.json) (rendered in [`figs/lineage_dag.svg`](figs/lineage_dag.svg)) is not a straight line — it is a sequence of hypotheses, several of them killed outright, one erratum, and one supersession, feeding a mechanical winner that itself fails a later gate.

**Phase 0–1: the ruler comes before the model.** `evalphys` v1.1.0 was frozen before any representation work, and the first substantive test (T1) was run to falsify, not confirm, a plan-level assumption: that a *soft* basis change (joint T/S EOF, or separate density+spice PCA) would fix the static-stability violations that truncated PCA bases manufacture (`reports/t1_basis_stability.md`). It didn't — B (22.63% profile violation rate) and C (21.83%) are statistically indistinguishable from the A baseline (21.51%). That result is recorded as **killed**, and the branch escalated to a human decision (R1) rather than being quietly reframed. Only a *hard* monotone constraint (variant D, a softplus+cumsum control-grid parameterization) cut the rate to 0.48% — a >40× reduction — and became Phase 3's representation C.

**Phase 3: an erratum, not a footnote.** The first attempt to pass the chronological skill gate with an isotonic-projection fallback failed (T=0.514 vs a published floor of 0.4574) and was **killed**. Chasing that failure surfaced a real erratum: the `argo16_scales` checkpoint used throughout has no `split_mode` key and defaults to a random split, so its published 0.4158 is a random-split number, and every chronological evaluation of that same checkpoint — including the 0.514 gate figure itself and an earlier density-diagnostic mse_σ of 0.21 — was leaked-optimistic (`reports/gate_floor_provenance.md`). The remedy was a same-day clean chrono retrain (`argo16_chrono_clean`, raw T=0.5367) and a restated gate: *within 10% of the argo16 baseline on the same split*, giving a corrected floor of 0.5903. The isotonic-projection candidate then passed cleanly against the corrected floor (T=0.5367, pre-inversion σ₀=0). Separately, an attempt to compress the density head's low-rank control vector in the softplus-inverse ("a-space") was **killed** (R=16 reconstruction RMSE 0.925, worse than climatology) and superseded by compressing in physical σ₀ space instead (RMSE 0.026) — a citable representation finding in its own right (`reports/finding_compress_physical_space.md`): *compress in physical space, constrain after*.

**Phase 5: a protocol failure, then a matrix, then a supersession.** The first attempt to run the probabilistic matrix cells under the originally-transcribed protocol (early-stop stage-2 on validation loss) was **killed**: it reproduced a known Phase-4 short-stage-2 calibration failure (ENCE 0.225 vs the 0.20 gate) independent of which cell "won." The protocol was corrected (early-stop on validation ENCE, patience 40, applied uniformly) and the full 3×3 matrix was scored under it. One cell's earlier single-run admission — C×det at T=0.562 — did **not** replicate under the fair 3-seed protocol (0.609±0.012, `reports/finding_C_det_gate_overfit.md`): the matrix caught its own over-fit gate pass. The latent-space judgment table itself was then **superseded**: PC-space CRPS is not comparable across representations that use different bases (A: separate PCA-16, B: joint EOF-32, C: density+spice PCA), so a physical-space decode-and-rescore pass (`reports/ablation_summary.md`) replaced it as the basis for cross-representation ranking.

**Phase 6: pre-register, then fail two pre-registered claims, then isolate why a third fix didn't help.** The OSSE experiment table was locked before the Phase-5 winner's identity was known, specifically so the E-table design could not be shaped by which cell won (`reports/osse_preregistration.md`). Both primary claims **failed** on the first run. A follow-up hypothesis — that Schur-localizing the full predicted covariance would restore numerical stability *and* buy skill from genuine cross-level structure — was itself pre-registered (commit `0422a51`, timestamped before the promotion run) and then **killed** by its own promotion run (commit `490de67`): the structured covariance is numerically stable, but scores worse than a diagonal covariance built from the identical head.

---

## 4. The matrix result, stated plainly

Physical-space table (`reports/ablation_summary.md`; skill floor 0.5903; ENCE gate <0.20):

| cell | T RMSE | physical CRPS(T+S) | physical ENCE | verdict |
|------|-------:|--------------------:|---------------:|---------|
| A×CRPS | 0.559±0.005 | 0.119±0.001 | 0.153±0.007 | **dissertation default** |
| A×NLL | 0.575±0.023 | 0.122±0.005 | 0.162±0.019 | ENCE survivor, not picked (higher CRPS) |
| A×det | 0.541±0.004 | — | — | det survivor |
| B×CRPS | 0.586±0.054 | 0.133±0.009 | 0.247±0.003 | ENCE FAIL |
| B×NLL | 0.563±0.011 | 0.128±0.002 | 0.299±0.013 | ENCE FAIL |
| B×det | **0.534±0.001** | — | — | **best det-only skill in the matrix** |
| C×CRPS | 0.618±0.103 | 0.139±0.022 | 0.384±0.010 | ENCE FAIL |
| C×NLL | 0.694±0.081 | 0.157±0.018 | 0.397±0.011 | ENCE FAIL |
| C×det | 0.609±0.012 | — | — | skill-floor FAIL |

**B won deterministic RMSE** (0.534, the lowest T RMSE anywhere in the matrix) but has no calibrated-uncertainty path and is not the DA-facing default for that reason, not because it lost on skill. **A won the probabilistic crown** by the pre-registered decision rule (lowest physical CRPS among ENCE<0.20 survivors: A×CRPS at 0.119 vs A×NLL at 0.122). **C (the monotone-density + spice representation) lost on both axes**: no C cell clears the physical ENCE gate, and C×det misses the skill floor (0.609 > 0.5903). The representation whose entire purpose was fixing static-stability violations does not win the matrix that was built to pick a headline model — its contribution is the T1 stability finding, not a Phase 5 win. See `reports/evolution/figs/matrix_gate_heatmap.svg` for the full latent-space judgment grid (hatched where PC-space CRPS is not cross-comparable) and `reports/evolution/figs/lineage_dag.svg` for how these cells relate.

---

## 5. Calibration reality

The A×CRPS winner clears the ENCE gate in the space it was judged in during the matrix (PC-space ENCE=0.053±0.004, `reports/phase5_A_CRPS.md`) but **fails** the same 0.20 gate once scored in physical temperature space: overall ENCE(T)=0.2362±0.0053 (`reports/phase5_A_CRPS_physical_strata.md`). This is not a marginal miss. Broken out by depth band × season (`reports/evolution/figs/depthband_season_ence.svg`), the worst calibration is in the **surface** band (0–50 m: 0.37–0.66 across seasons, worst in JJA at 0.6622) and the **thermocline** (50–200 m: 0.18–0.56, also worst in JJA), with the >800 m band also failing uniformly (0.32–0.55). Only the 200–800 m band is well calibrated (0.06–0.16). CRPS itself (`reports/evolution/figs/depthband_season_crps.svg`) peaks in the 50–200 m band (up to 0.375 in JJA) and is smallest below 800 m (as low as 0.028 in MAM) — the model is both least sharp and least calibrated in the upper ocean, which is also where dynamical variability is highest. The PC-space ENCE pass that admitted this cell to the matrix does not describe its behavior on the physical variable a downstream DA system would actually assimilate.

---

## 6. OSSE result, without spin

Canonical cast-column run, 2021, n=1101 casts (`reports/osse_results.md`, `NeSPReSO2_onTemplate/saved/runs/phase6_osse/cast_column_s42.json`):

| E | R construction | overall T RMSE |
|---|---|---:|
| E2 | ISOP/MODAS-class ridge baseline, R_fixed | 0.5410 |
| E3 | NeSPReSO casts, R_fixed | 0.5454 |
| E4 | NeSPReSO casts, R_cal (full localized Σ_T) | 0.6160 |
| E5 | E4 + QC (retain 0.444 of casts) | 1.4008 |

**NeSPReSO ties ISOP, and slightly loses**: E3 (0.5454) is not lower than E2 (0.5410) — the pre-registered claim **E3 > E2 is FAIL**. Calibrated R does not tie the diagonal fallback either, once localization is compared apples-to-apples (`HANDOFF.md`; commit `490de67` message):

| variant | overall T RMSE |
|---|---:|
| E3, R_fixed (no calibration) | 0.5454 |
| E4, `--rcal diag` (v1 fallback, diag(Σ_T) only) | 0.5463 |
| E4, `--rcal full` (v2, full localized Σ_T) | 0.6160 |

The diagonal calibrated covariance **ties** the fixed-R baseline within noise (0.5463 vs 0.5454). The full structured covariance is **worse than both** (0.6160). Because the Schur localization preserves diag(Σ) exactly, the entire 0.5463→0.6160 degradation is attributable to the CRPS head's off-diagonal (cross-level) terms alone (`reports/evolution/figs/diag_control_headline.svg`). Those terms are basis-induced — the same PCA/EOF basis `V` shared across levels produces them — not a learned representation of true observation-error correlation; the head was trained for per-dimension (marginal) CRPS and was never given a signal that would let it learn genuine cross-level structure. E5's QC rule retains a 0.444 fraction of casts (the best-calibrated half by predicted σ̄) and still scores 1.4008 — worse than any of E2–E4 — because the QC threshold was locked on validation data before the test run and was not retuned toward this outcome (`reports/osse_preregistration.md` §2.1).

---

## 7. What is genuinely established vs. what is not

**Established:**
- A hard monotone density constraint removes static-stability violations that no soft basis change reaches (21.51%→0.48% profile rate); this is a real, replicated, and cheap-to-verify finding (`reports/t1_basis_stability.md`).
- Compressing a monotone-constrained quantity linearly in its *physical* space, and applying the constraint only at inference, preserves low-rank skill; compressing in the constraint's nonlinear preimage does not (`reports/finding_compress_physical_space.md`).
- A published skill baseline can be silently random-split-trained; chronological evaluation of it will look worse than it should for reasons unrelated to the candidate being tested, and this can propagate into downstream branch decisions until traced (`reports/gate_floor_provenance.md`).
- A 3-seed, protocol-locked matrix catches at least two things a single run's admission pass hides: an under-trained calibration schedule that looks fine on one metric (`reports/phase5_C_CRPS.md`, protocol v1), and a skill number that does not replicate (`reports/finding_C_det_gate_overfit.md`).
- **The calibrated-ML-pseudo-observations-improve-DA thesis is empirically falsified in this testbed.** In the cast-column Gulf-of-Mexico proxy: NeSPReSO casts tie (and numerically trail) an ISOP/MODAS-class ridge-regression baseline on fixed R (E3 vs E2: 0.5454 vs 0.5410); a calibrated diagonal observation-error covariance ties the uncalibrated fixed-R baseline within noise (0.5463 vs 0.5454); and a calibrated *structured* covariance built from the same trained head actively degrades the analysis (0.6160), with the mechanism isolated by direct diag-vs-full control to basis-induced cross-level correlation in the CRPS head, not genuine observation-error structure. This is a closed question and a clean negative, not a tie and not "inconclusive."

**Not established (scope limits, stated honestly):**
- Everything in §6 is a **cast-column proxy**: truth is ARGO at cast locations, not a gridded ISAS20 field, and there is no horizontal localization (L_h) in the analysis update — the map-level version of this experiment has not been run (2021 ISAS grid is not on disk, `reports/osse_preregistration.md`).
- This is a **single region** (Gulf of Mexico) and a **single toy OI scheme** (column-wise, vertical-only B and R); nothing here bears on time-stepping DA or other basins.
- The negative covariance result is specific to a CRPS head trained for **marginal** per-dimension calibration; it does not test whether a head trained with an explicit joint-covariance objective would produce useful cross-level structure — that is an open architectural question, not one this branch answers.
- The Phase-5 winner's physical-space miscalibration (§5) means the *uncertainty* half of "NeSPReSO casts" (as opposed to the mean) is demonstrably unreliable in the upper ocean and deep water even before the OSSE finds no DA value in it.

**Recommended reframing.** Given the falsification in §6 and the scope limits above, the defensible contribution of this branch is not "calibrated NeSPReSO improves DA" — that objective should be formally retired. What is defensible and citable is the **package**: a frozen, audited evaluation standard (`evalphys` v1.1.0) that caught a real data leakage erratum and an over-fit gate pass; a pre-registered, protocol-locked ablation methodology that a single-run pipeline would not have caught either failure mode without; and a cleanly isolated, mechanism-explained negative result on structured observation-error covariance from ML uncertainty heads — itself a useful finding for anyone building pseudo-observation systems from marginally-calibrated probabilistic emulators. The datacube built alongside this work (`reports/evolution/figs/cube_dataflow.svg`) is part of that package as a data-quality and extraction-geometry fix (unified regional Zarr cube, on-demand bilinear sampling, v3 error-channel ingestion) — it is not evidence for the patch/residual branch, which lost the Phase 5 matrix on both skill and calibration axes.

---

## Datacube documentation strand

The regional cube (`NeSPReSO2_onTemplate/preproc/cube/`, `data/cube/gom_cube.zarr`) replaces a legacy per-station HDF5 patch-extraction pipeline (`utils/v2.json`/`v3.json`: `spatial_pad=20` grid cells, `temporal_pad=6` days, unchanged across the v2→v3 error-channel upgrade by explicit design note) with a single daily-resolution Zarr store over the GoM box (lat 18–31°N, lon −98..−81°W, 2015-01-01 to 2022-03-01, `NeSPReSO2_onTemplate/preproc/cube/cube_schema.py`) sampled on demand via bilinear interpolation weights (`NeSPReSO2_onTemplate/preproc/features/sampler.py`) rather than fixed-window patch files. Channels are the union of the v2/v3 product lists: SST (OSTIA), SSS (CMEMS), SSH (`adt`/`sla`/`ugos`/`vgos`, DUACS), bathymetry (GEBCO), plus the v3-only error channels `analysis_error`, `sos_error`, `err_sla` (`reports/phase2_2_error_channels.md`). Figures: [`figs/cube_schematic.svg`](figs/cube_schematic.svg), [`figs/cube_extraction_inset.svg`](figs/cube_extraction_inset.svg), [`figs/cube_dataflow.svg`](figs/cube_dataflow.svg), [`figs/cube_stale_fingerprint.svg`](figs/cube_stale_fingerprint.svg) (current cube-era stale-satellite fingerprint is 0.0% on all channels/splits, gate OPEN, `reports/stale_by_split.json`; no pre-cube "old" baseline is available in a committed artifact under that name — see `PROVENANCE.json:unsourced`).

**The cube's contribution, stated precisely:** it is a data-quality fix (closes the stale-satellite risk that a fixed-window legacy extraction could reintroduce) and a unified extraction geometry that enables the v3 error channels to be ingested consistently across point, patch, and residual models. **It is not evidence that patches beat points.** The dissertation-winning model (A×CRPS, §4 above) is a **point** model. The patch/residual branch that the cube was built to serve most directly lost the Phase 5 matrix on both the deterministic skill floor and the probabilistic calibration gate (§4); where older patch/L4 comparison artifacts are referenced in this report (`reports/evolution/PROVENANCE.json`, `reuse_*` entries) they document that branch's own pre-existing stale-satellite fix, not a claim that the branch won anything in the matrix that decided the dissertation default.
