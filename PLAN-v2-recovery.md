# PLAN-v2-recovery.md — NeSPReSO v2 scientific recovery plan

**Status:** authoritative plan for the representation + probabilistic + evaluation overhaul.
**Audience:** AI coding agent (Claude Code / Cursor) working in this repo. Read `CLAUDE.md`, `AGENTS.md`, `HANDOFF.md` first. Ponytail mode applies (`.cursor/rules/ponytail.mdc`): simplest thing that works, one runnable check per non-trivial unit. Numerical rules apply (`.cursor/skills/nespreso-numerical/SKILL.md`): explicit tolerances, seeds, no exact float equality.
**Environment:** conda env `nespreso`; GPU jobs via `srun --ntasks=1 --cpus-per-task=8 --gres=gpu:1`; CPU jobs drop `--gres`.

## Changelog

| Date | Change | Why |
|------|--------|-----|
| 2026-07-16 | **Inference-isotonic claim locked.** Low-rank δσ₀ is *not* head-monotone (45.6% pre-iso neg profiles). Isotonic mandatory at inference; PLAN §3.2 distinguishes claims. Reports: `finding_compress_physical_space.md`, `gate_floor_provenance.md`; eval hygiene + blame-split in spice_v3 report. Pass = Phase 5 matrix admission. | Hard-head vs inference-stable are different chapter claims; floor chain is dissertation material. |
| 2026-07-16 | **Low-rank δσ₀ PASS:** σ₀-space PCA R=16 + spice continue → chrono T **0.562** ≤ clean floor **0.590** (`phase3_lowrank_sigma0_spice_eval.md`). Pre-inv σ₀=0. a-space path retired. | In-head skill recovery clears corrected gate. |
| 2026-07-16 | **Low-rank erratum:** a-space PCA on `(a−a_clim)` is a dead end — R=16 σ₀ recon RMSE 0.93 > clim 0.72 despite 94% a-EVR (softplus⁻¹). v1 train T=0.830 FAIL. Retarget to **σ₀-space** PCA (`delta_sigma0_basis`); ceiling R=16 σ₀ RMSE 0.026. | Nonlinear control-space ≠ linear skill. |
| 2026-07-16 | **Low-rank δa (in-head #1):** `outputs.density_ctrl` = R scores (16); `io.n_ctrl` = K (64). Cache fits PCA on train `(a_true−a_clim)` → `delta_a_basis`; loss/eval decode `a = a_clim + scores @ basis` then softplus+cumsum. Prob+lowrank deferred. Smoke green. | Restore vertical coordination without abandoning monotone head. |
| 2026-07-16 | **Ruler repair + leakage erratum (human sign-off in-message).** (a) The §3.6 opt-2 gate floor 0.458 = published 0.4158 (random split) × 1.10 compared chronological candidates against a random-split baseline — a like-for-like split violation. Gate intent restated: *within 10% of the argo16 baseline on the same split*. Corrected floor = same-split argo16 raw × 1.10; both constants reported side by side in `eval_argo16_isotonic_gate.py`. (b) `argo16_scales/config.json` has **no `split_mode`** (loader default random) and its 0.416 matches the published random number ⇒ the checkpoint is random-trained; its training set overlaps the 2021–2022 chronological test era. All chrono evals of that checkpoint (0.514 gate figure; density-diag mse_σ 0.21 that drove `representation_plumbing`) are **leaked-optimistic**. Erratum added to `phase3_density_shift_diag.md`. Remedy executed same day: trained clean chrono argo16 (`argo16_chrono_clean`, early stop ep 814). **Results:** clean chrono raw T = **0.5367** (leaked figure was 0.514 — optimistic as predicted). Density diag re-run clean: argo16 mse_σ val 0.146 / test **0.234** vs densonly 0.913 → **`representation_plumbing` branch survives and strengthens** (`phase3_density_shift_diag_clean.md`). Corrected gate floor = 0.5367×1.10 = **0.5903**; argo16+isotonic → T 0.5367, pre-inv σ₀ = 0, proj cost 0.0014 °C → **gate PASS** (`phase3_argo16_isotonic_gate_clean.md`), committed as Phase 3 candidate. Both constants (0.4574 published-random floor, 0.5903 same-split floor) reported side by side. | Same error category as Finding 3, in the other direction; branch decision rested on a leaked control. |
| 2026-07-16 | **§3.6 opt-2 gate:** chronological argo16+isotonic T=0.514 **FAIL** vs published 0.458; pre-inv σ₀=0. Random-split same recipe **PASS** (T=0.416) — published 0.416 was random. Shrinkage recomputed in **σ₀ space** (~0.32). In-head priority: **low-rank-δa** then month-clim. No merge. | Projection opens stability, not chrono skill. |
| 2026-07-16 | **Density-shift diags (eval-only):** clim test/val 1.11; densonly 2.12; argo16 density mse_σ test 0.21 ≪ densonly 0.91. Branch = **representation_plumbing** (signal extractable). Keep v10 spice frozen; do not merge to main. | Close schedule/interference hypotheses before retrain. |
| 2026-07-16 | **densonly ablation (λ_τ=0):** pred σ₀+true τ = 0.547 ≈ v10 0.522 — multi-task interference **refuted**. Density chrono generalization remains the bottleneck. Next: sequential warm-start (freeze v10 spice) or EMA-normalized joint; §3.6 opt-2 still floor. | Close the interference hypothesis before more λ sweeps. |
| 2026-07-16 | **§3.6 fallback option 2 pre-registered:** if in-head monotone skill gate still fails after branch decoupling, keep best-skill μ and enforce stability at inference via isotonic σ₀ projection + re-inversion (T1 variant-D op; RMSE cost already measured small). Preferred path remains in-head. | Floor under skill gate so Phase 5 is not a cliff. |
| 2026-07-16 | **Phase 3/4 full runs:** density_spice + CRPS two-stage on full cache. Skill gate FAIL (T 0.72 vs argo16 0.42) with σ₀=0; CRPS Spearman 0.65 PASS / ENCE 0.33 MISS (informational). Diagnosis: spice-first λ + residual δa starved density gradients — structural decoupling next, not λ sweep. | Commit FAIL-state; recover mean before re-CRPS. |
| 2026-07-16 | **Phase 4 smoke cleared:** heteroscedastic/quantile `PatchConvMLP`, `DensitySpiceProbLoss` (`crps`/`nll`/`quantile`/`mse`), two-stage launcher, `dacov` Σ export, uncertainty decomposition script. Acceptance: all three modes train; twostage CRPS green; dacov PSD+MC. | Continue PLAN after Phase 3 gate. |
| 2026-07-16 | **Human sign-off:** framing = *"no soft basis fixes stability; hard monotonicity does, at cost X."* Phase-3 deep-band FAIL diagnosed (softplus clamp on ~12% negative ctrl increments) and fixed (isotonic before encode). T1-E gate **PASS** (E/A ≤ 1.0 all T bands). **Big win:** hard constraint *improves* upper-ocean T RMSE vs A while zeroing σ₀ profile rate (0.215→0.000). Phase 4 unblocked. | Human approved mechanism reframing; accepted diagnostic→fix→re-gate path. |
| 2026-07-16 | T1 escalate → R1: Phase 3.2 monotone head. Mechanism update: **truncation itself** (not T/S separateness) drives σ₀ violations — B joint EOF ≈ A under historical ruler; only hard constraint cuts 21.51%→0.48%. Residual post-inversion violations ⇒ Phase 3 tracks **inversion fidelity** as first-class. Phase 2.1 satisfied; R4 golden root-cause is a **Phase 5 prerequisite**. | Human accepted audit Decision R1; B/C historical rows complete the record. |
| 2026-07-16 | §3.2 / §0.1: hard constraint guarantees **σ₀ monotonicity**; residual **N²** violations are expected to be small and must be **reported**, not assumed zero. Additive `sigma0_monotonicity_violations` in evalphys v1.1.0. | Audit C.2: monotone σ₀ control-grid + PCHIP does not imply N²≡0 after (σ₀,τ)→(T,S) inversion (locally referenced N²). |
| 2026-07-16 | Headline metrics always use reference `gsw` via `evalphys.gsw_backend.get_gsw()`; `io.gsw_backend` / `--gsw-backend` select training / equivalence only. | Audit F: `diagnostics/readiness.py` aliases `gsw_torch as gsw`; evalphys must not silently substitute. |

---

## 0. Context and rationale (read once, do not skip)

Three session-established findings drive this plan:

1. **Truncation manufactures static-stability violations; soft basis changes do not fix them.** Reconstructing ground-truth ARGO profiles through truncated bases raises the historical σ₀-inversion profile rate from ~1.1% (raw) to ~22% (PCA-16). **Human sign-off 2026-07-16:** joint EOF (B) and density/spice PCA (C) leave that rate ≈ A — the load-bearing mechanism is truncation itself, not T/S basis separateness. Soft representation changes do not buy stability; only a hard monotone density constraint does (framing: *"no soft basis fixes stability; hard monotonicity does, at cost X"*). Cost X on GoM ARGO is **negative in the upper ocean** (E beats A on T RMSE in 0–800 m while zeroing σ₀ violations — see `reports/t1_basis_stability.md`).
2. **MSE + deterministic output ⇒ conditional-mean under-dispersion.** Anomaly PC1 amplitude shrinks to ~0.78×, ensemble-style spread to ~0.20×, and spread–|error| rank correlation is ~0.12. A deterministic conditional-mean emulator cannot provide the background/observation-error covariance a DA system needs.
3. **The binding constraint has been evaluation validity, not capacity.** Non-comparable val_loss across retrains, self-referential baselines, inference-mode artifacts. Therefore: metrics get frozen FIRST, before any new model runs.

Plan-level decisions already made (do not relitigate in-code):
- Hard-constrain static stability via a **monotone density parameterization** (Phase 3), not a penalty.
- Probabilistic head trained with **CRPS (preferred) or β-NLL**, two-stage schedule (Phase 4).
- Exploit **dataset-provided uncertainty fields** (`err_sla`, OSTIA `analysis_error`, SSS error, target adjusted errors) as inputs, augmentation noise, and loss attenuation (Phases 2 & 4).
- Collapse to **one backbone**; ablation = representation × head matrix (Phase 5).
- **Toy OSSE with an ISOP/MODAS-class synthetic-profile baseline**, not only a no-cast control (Phase 6).
- DUACS/OSTIA formal errors are smooth and under-dispersive: treat as **relative** quality indicators; the network learns the scaling. Never use them as absolute variances.

Phase dependency graph: `P0 → P1 → (P2 ∥ P3-design) → P3 → P4 → P5 → P6`. P1 outcomes gate P3 choices. P2 gates any headline metric on the chronological test split.

---

## Phase 0 — Freeze the rulers (physical + probabilistic metric suite)

**Objective:** a versioned, tested, immutable metrics package that every later phase imports. No later phase may redefine a metric.

**Create** `NeSPReSO2_onTemplate/evalphys/` with `__init__.py`, `metrics.py`, `calibration.py`, `manifest.py`, and `tests/test_evalphys.py`.

### 0.1 `metrics.py` — physical metrics

All functions take numpy arrays shaped `(n_profiles, n_levels)` for T [°C, in-situ], S [PSU, practical], plus `depth` (m, positive down, shape `(n_levels,)`), `lat`, `lon` `(n_profiles,)`. Use `gsw` (reference impl) — GSW-Torch is for training-time losses only, but add one test asserting gsw vs GSW-Torch agreement to `atol=1e-6` on σ₀ for 100 random profiles.

Conversions (do once, helper `to_teos10(T, S, depth, lat, lon)`):
```
p  = gsw.p_from_z(-depth, lat)          # dbar
SA = gsw.SA_from_SP(S, p, lon, lat)     # absolute salinity
CT = gsw.CT_from_t(SA, T, p)            # conservative temperature
```

**(a) Static-stability violation rate with tolerance floor.** Compute N² via `gsw.Nsquared(SA, CT, p, lat)` (returns mid-point values). A level is a violation iff:
```
N²_k < -N2_TOL,   N2_TOL = 1.0e-8  s^-2       # frozen default
```
Report: `violation_rate_profile` (fraction of profiles with ≥1 violation), `violation_rate_level` (fraction of all level-pairs), and both stratified by depth band `{0–50, 50–200, 200–800, >800} m`. Also report the sensitivity vector at `N2_TOL ∈ {0, 1e-9, 1e-8, 1e-7}` in every summary JSON — the headline number uses 1e-8, but the sweep is always attached so no one can cherry-pick.

**(b) Mixed-layer depth (de Boyer Montégut threshold).** MLD = shallowest depth where `σ₀(z) − σ₀(z_ref=10 m) > 0.03 kg/m³` (linear interpolation between levels). Report RMSE and bias of predicted vs true MLD.

**(c) Isotherm depths D20 and D26** (depth of 20 °C and 26 °C isotherm, linear interp; D26 is the hurricane-relevant one). Report RMSE/bias. Skip profiles where the isotherm doesn't exist; report coverage.

**(d) dρ/dz profile RMSE.** RMSE of `Δσ₀/Δz` between predicted and true, by depth band. This is the derivative-aware metric that catches smoothing that L2-on-T,S hides.

**(e) Steric-height consistency (diagnostic only until SSH targets exist).**
```
η_steric = -(1/ρ0) ∫_{z_bot}^{0} (ρ(z) − ρ_clim(z)) dz ,  ρ0 = 1025 kg/m³
```
Report RMS mismatch of predicted vs truth-derived η_steric (cm).

### 0.2 `calibration.py` — probabilistic metrics

Inputs: predicted mean `mu`, predicted std `sigma` (same shape as targets `y`), or an ensemble `(n_draws, n, d)`.

**(a) Gaussian CRPS (closed form)** — also used as a training loss in Phase 4, implement once here in numpy and mirror in torch:
```
z = (y − mu) / sigma
CRPS = sigma * ( z*(2*Φ(z) − 1) + 2*φ(z) − 1/sqrt(pi) )
```
Φ, φ = standard normal CDF/PDF. Report mean CRPS overall and per depth band.

**(b) PIT histogram.** `PIT = Φ((y − mu)/sigma)`; 20 bins; report histogram + deviation statistic `sup_bin |freq − 0.05|`.

**(c) Spread–skill.** Bin samples into 10 deciles of predicted σ; per bin compute RMSE; report (i) slope of RMSE vs mean σ (target 1.0), (ii) Spearman rank correlation of σ vs |error| (the RC-4 statistic; session baseline ≈ 0.12 — beat this).

**(d) ENCE** (Levi et al.): bins by σ, `ENCE = mean_b | RMSE_b − RMV_b | / RMV_b` where RMV = root mean predicted variance. Pre-registered prospectus threshold: **< 0.20**.

**(e) Stratification helper.** Every calibration metric callable with a `strata` dict: depth band × season (DJF/MAM/JJA/SON from JULD) × input-error tercile (Phase 2 provides the error channel; until then this stratum is skipped, not faked).

### 0.3 Freezing mechanics

1. `manifest.py` writes `evalphys/METRICS_MANIFEST.json`: `{version: "1.0.0", frozen_date, N2_TOL, thresholds: {ence_max: 0.20, rc1_note: "hard constraint ⇒ 0 by construction in Phase 3; report cost in RMSE/sharpness instead"}, git_sha}`.
2. Tests: synthetic stable profile ⇒ 0 violations; injected inversion of −0.05 kg/m³ over 5 m ⇒ detected; perfectly calibrated synthetic Gaussians ⇒ PIT uniform (chi-square p > 0.01), ENCE < 0.05, spread-skill slope ∈ [0.9, 1.1]; CRPS of point forecast (σ→σ_min) equals MAE within 1%.
3. `python -m pytest NeSPReSO2_onTemplate/evalphys/tests -q` must pass; then commit with tag `evalphys-v1.0.0`. **Rule: after the tag, edits only for bugs, each with a regression test, version bump, and a line in the manifest changelog.**

**Acceptance:** tests green; manifest committed; `selfcheck.py` extended to import evalphys and run the synthetic checks.

---

## Phase 1 — Decisive cheap tests (run before building anything)

**Objective:** validate/refute the two load-bearing mechanisms and clear the data audit, each in hours not days. Write results to `reports/phase1_decisive_tests.md`.

### T1 — Joint vs separate basis reconstruction test (the Finding-1 mechanism check)

Script: `NeSPReSO2_onTemplate/scripts/t1_basis_stability.py`.

1. Load the ARGO train cache (`train_ready_*.pkl`, tag `argo_*`); extract raw T, S profiles, depth, lat/lon, JULD; use **train split only** to fit bases (chronological split via `base.split_utils.build_split_indices`), evaluate on test split.
2. Fit four reconstructions of the *truth* (no model anywhere — this isolates the representation):
   - **A (current):** separate PCA, 16 T modes + 16 S modes, each standardized independently.
   - **B (joint EOF):** z-score T and S per level with train stats, concatenate to one `(n, 2·n_levels)` matrix, single PCA with 32 modes.
   - **C (density/spice PCA):** compute σ₀ and spice τ (`gsw.spiciness0(SA, CT)`); PCA with 16+16 modes on (σ₀, τ); invert back to (T, S) with the Phase 3.4 Newton inversion (use the gsw/scipy version, no torch needed here).
   - **D (monotone-density control-grid):** project σ₀ onto the Phase 3.2 monotone parameterization by isotonic regression (`sklearn.isotonic.IsotonicRegression`, increasing) at the 64-level control grid + PCHIP upsampling; spice as in C.
3. For each: reconstruct T, S on the test split; run frozen evalphys: violation rates (with tolerance sweep), T/S RMSE by depth band, dρ/dz RMSE, MLD RMSE.
4. **Decision rules (pre-registered):**
   - If B and/or C cut the level violation rate by ≥ 5× vs A at ≤ 10% RMSE cost ⇒ Finding-1 mechanism confirmed; Phase 3 proceeds as planned.
   - If C ≈ A (no improvement) ⇒ the violations are not basis-induced; escalate to human before Phase 3 (the representation chapter framing changes).
   - D should show violation rate ≡ 0 by construction; record its RMSE cost — this is the "price of hard stability" headline number.

### T2 — Stale-input audit of the chronological test split

Extend `diagnostics/stale_sat/split_vs_stale.py` to cover SST, SSH (adt+sla), **and SSS**, per split. Output `reports/stale_by_split.md` with stale fractions per variable per split.
**Gate:** if stale fraction in val or test > 5% for any variable ⇒ **all headline metrics are embargoed** until Phase 2.1 lands. Print this gate status in `selfcheck.py`.

### T3 — Violation-metric sensitivity (folds into T1)

Already produced by the tolerance sweep. One extra check: recompute rates excluding the top 15 m (where near-neutral N² makes even truth noisy). If exclusion changes conclusions, report both; headline stays full-column at N2_TOL=1e-8.

**Acceptance:** `reports/phase1_decisive_tests.md` exists with tables, decision-rule outcomes stated explicitly, and the T2 gate status.

---

## Phase 2 — Data: SSS fix, uncertainty-field ingestion, new predictors

**Objective:** clean test-split inputs; add product error fields and unused physical predictors to the patch pipeline and caches.

### 2.1 SSS gap fill (blocking)

1. Run `utils/download_SSS_range.py 2021-01-01 2022-02-28` (resumable; skips existing).
2. Identify affected stations (those whose 7-day SSS patch window intersects the gap); regenerate only those HDF5 batches via the existing resumable ARGO satellite generator; recombine; rebuild the train cache with `--force`.
3. Re-run T2. Gate lifts when stale fractions < 5% in val and test.

### 2.2 Uncertainty/error field ingestion

**Never hardcode variable names.** For each product, first open one file and `print(ds.data_vars)`; confirm against the list below; if a name differs, use what the file says and record it in `reports/data_census.md`.

Expected fields (verify per 2.2 rule):
- **SSH (DUACS L4, already downloaded):** `err_sla`, `err_ugosa`, `err_vgosa`.
- **SST (OSTIA):** `analysis_error` (may require re-download with the variable added to the subset request — extend `variables=[...]` in the download script).
- **SSS (CMEMS multiobs `..._my_multi_P1D`):** per-pixel error variable (inspect; commonly `sos_error` or a `*_uncertainty` name). If genuinely absent from the product, substitute the product's quality/pct-variance field and note it.
- **Winds (if/when used):** ERA5 EDA spread — optional, defer unless free.

Implementation:
1. Create `utils/v3.json` = `v2.json` + error variables under each product key. Keep `spatial_pad`/`temporal_pad` unchanged so patch geometry is identical.
2. Extend `retrieve_sat.py` product handling only if it filters variables (it should pass through whatever the config lists — verify with a 3-station smoke query like `check_ssh.py`).
3. Cache schema: add `inputs_err` array aligned with `inputs` (same station order), plus `input_error_channels: [names]` metadata. **Normalization for error channels (they are positive and right-skewed):**
```
e' = ( log(e + e0) − μ_train ) / s_train ,   e0 = 1e-6 (units of e)
```
μ, s from the train split only; store in `input_standardization` next to the existing stats. NaN error (e.g., land-adjacent pixels) → fill with the train-split 90th percentile of e before transform, and add a companion mask channel `err_missing ∈ {0,1}`.

### 2.3 New physical predictors (cheap, do with 2.2)

- Ensure `ugos`, `vgos` are in the model input set (they're downloaded in v2 config; confirm they reach the cache and the model input vector, not just the HDF5).
- Bathymetry gradient: from the existing elevation patch compute `|∇h|` (central differences on the 5×5 patch) as one scalar channel at center. Ponytail: one function + one assert-based check.

### 2.4 Target-side uncertainties

- ARGO-derived caches: if the upstream source retains `TEMP_ADJUSTED_ERROR`/`PSAL_ADJUSTED_ERROR`, carry per-level target errors into the cache as `target_err_T`, `target_err_S`. If unavailable, fall back to depth-dependent constants from the literature (T: 0.002 °C instrument floor is unrealistically small for representativeness — use test-era representativeness proxy: per-level std of ARGO minus ISAS20 colocated, computed once, stored). Document which path was taken.
- ISAS20 gridded targets (if/when used): ingest the error/percent-variance field at profile locations.

**Acceptance:** rebuilt cache passes `selfcheck.py`; `scripts/data_census.py` extended to report error-channel coverage and stats; T2 gate lifted; a 3-sample end-to-end fetch smoke test for v3 products exists under `tests/`.

---

## Phase 3 — Representation: monotone density + spice

**Objective:** replace the separate T/S PCA target with a stability-hard-constrained (σ₀, τ) representation, invertible back to (T, S).

### 3.1 Variables

Targets per profile: potential density anomaly `σ₀(z)` and spice `τ(z)` on the native depth grid, computed once at cache build:
```
σ₀ = gsw.sigma0(SA, CT)        τ = gsw.spiciness0(SA, CT)
```
Store alongside T, S in the cache (`targets_sigma0`, `targets_spice`), plus per-level train-split means/stds.

### 3.2 Monotone density head (hard RC-1 constraint)

Depth control grid: `K = 64` levels, log-spaced in depth over [0, z_max] (denser near surface), fixed and stored in config. The network's density output is a vector `a ∈ R^K`:
```
σ̂₀(z_1)     = a_1                                  (unconstrained surface value, standardized units)
σ̂₀(z_k)     = σ̂₀(z_1) + Σ_{j=2..k} softplus(a_j) · Δz̃_j     for k = 2..K
```
`Δz̃_j` = control-grid spacing normalized to mean 1 (keeps softplus outputs O(1)). Because `softplus > 0`, σ̂₀ is strictly increasing on the control grid ⇒ **zero σ₀-space inversions by construction** (see `evalphys.sigma0_monotonicity_violations`). Upsample control grid → native 1801 levels with **PCHIP** (`scipy.interpolate.PchipInterpolator`; monotone data ⇒ monotone interpolant, so the σ₀ guarantee survives upsampling — add a test asserting this on 1000 random draws). Torch-side, implement linear interpolation for the training loss (monotonicity also preserved by linear interp) and reserve PCHIP for eval/export.

**σ₀ vs N² (audit 2026-07-16):** The hard constraint guarantees **σ₀ monotonicity** (depth-increasing σ₀) **pre-inversion** on the control grid. The Phase-0 headline physical metric remains **N²** (`gsw.Nsquared`, §0.1). Residual violations after (σ₀,τ)→(T,S) Newton inversion (~0.3–0.5% σ₀ profile / ~0.2% N² level on T1-D) are expected and must be **reported** — they are dominated by **inversion round-trip error**, not control-grid non-monotonicity. Phase 3 therefore tracks **inversion fidelity** (round-trip |ΔT|, |ΔS|, recovered-σ₀ monotonicity, Newton fail rate) as a first-class metric alongside N². Do not delete or alter the N² metric.

Note: strict monotonicity is marginally stronger than the physical requirement (neutral layers allowed). Acceptable: softplus output can be arbitrarily close to 0. Do not add an ε relaxation.

**Truth-projection / cache targets (2026-07-16 deep-band fix):** linear interp of native σ₀ onto the ctrl grid leaves ~12% negative increments. Encoding those with softplus clamp injects a cumulative σ₀ bias that peaks below 800 m (T1-E FAIL). Always isotonic-project ctrl σ₀ before softplus encode (`project_monotone_sigma0_ctrl`); cache export applies the same projection to density targets so the loss floor matches what decode can represent.

**Low-rank δσ₀ path (2026-07-16 — different chapter claim):** When `outputs.density_ctrl = R < K` with `delta_sigma0_basis`, the network predicts R scores and reconstructs
```
σ̂₀_raw = σ₀_clim + scores @ V_σ ,   V_σ ∈ R^{R×K} from PCA on train (σ₀ − clim).
```
This is a **linear residual in physical σ₀ space** — it is **not** monotone by construction. Measured on the winning chrono test set: **45.6%** of profiles have ≥1 negative ctrl increment *before* projection. The required inference op is **isotonic projection** (`project_monotone_sigma0_ctrl`) then PCHIP+invert — exactly §3.6 option 2 / T1-D. Measured cost: σ₀ RMSE of projection ≈ 0.008; ΔT RMSE ≈ **−0.0002 °C** (slightly helps). **Dissertation claim for this candidate:** *"stable by construction at inference"*, **not** *"hard constraint in the head."* The full-rank softplus+cumsum head retains the stronger claim. Do not conflate the two in the representation chapter. Cov export (Phase 4.4) for this path is the clean `Σ_ρ = V diag(σ_z²) Vᵀ` (no softplus Jacobian).

### 3.3 Spice head

Standard PCA head: 16 modes fit on train-split standardized τ profiles; network predicts 16 scores; decode `τ̂ = z_τ V_τᵀ` then de-standardize. (Spice carries no monotonicity constraint.)

### 3.4 Inversion (σ₀, τ) → (SA, CT) → (T, S)

Per level, solve the 2×2 system `sigma0(SA, CT) = σ̂₀`, `spiciness0(SA, CT) = τ̂` by Newton iteration:
```
init: (SA, CT) from monthly climatology at (lat, lon, z)  [fallback: profile-set train mean]
repeat ≤ 12:
    F = [sigma0(SA,CT) − σ̂₀ ,  spiciness0(SA,CT) − τ̂]
    J = ∂F/∂(SA,CT)            # via GSW-Torch autograd (torch path) or finite differences 1e-4 (numpy path)
    (SA, CT) ← (SA, CT) − J⁻¹ F        # damped: step ← 0.5·step if ||F|| increases
    stop when ||F||_∞ < 1e-6
clamp: SA ∈ [30, 40] g/kg, CT ∈ [-2, 35] °C; log any clamp activation
```
Then `T = gsw.t_from_CT(SA, CT, p)`, `S = gsw.SP_from_SA(SA, p, lon, lat)`. Vectorize over (profile, level). **Validation test:** round-trip truth (T,S) → (σ₀,τ) → (T,S), assert max |ΔT| < 0.01 °C, |ΔS| < 0.01 PSU on 500 random test profiles; report convergence-failure count (must be < 0.1%, failures fall back to climatology + flag).

### 3.5 Training loss (deterministic version, upgraded in Phase 4)

```
L = λ_ρ · MSE(σ̂₀_ctrl, σ₀_ctrl) + λ_τ · MSE(ẑ_τ, z_τ) + λ_f · MSE_functional(T̂, Ŝ vs T, S on native grid)
```
σ₀_ctrl = truth interpolated to the control grid, standardized. Derive λ's with the existing `scripts/derive_loss_scales.py` pattern (equalize initial gradient magnitudes; freeze values in config). The functional term uses the differentiable linear-interp + GSW-Torch inversion path; if the in-graph Newton is unstable, drop λ_f to 0 for v1 and note it — the hard constraint does not depend on it.

### 3.6 Fallback

**Option 1 (inversion broken):** If 3.4 inversion fails validation and can't be fixed in ≤ 2 days: fall back to **joint T/S EOF (T1 variant B)** as the representation, keep the monotone-density evaluation as a diagnostic. The ablation matrix (Phase 5) still includes both. Note: T1 already showed joint EOF does **not** fix stability — this is an inversion-engineering escape hatch, not a stability fix.

**Option 2 (skill gate FAIL after decoupling — preferred vs retreat to B):** If the in-architecture monotone head cannot recover skill (overall T ≤ argo16×1.10) after the pre-registered structural fixes, do **not** abandon the best-skill representation. Instead: keep that μ for prediction, and enforce stability at **inference** by isotonic projection of predicted σ₀ onto the monotone control grid + re-inversion — exactly the **T1 variant-D** operation.

**2026-07-16 eval:** argo16+isotonic on **chronological** test → pre-inv σ₀=0, T≈0.514 (FAIL vs published 0.458 floor). Same recipe on **random** test → PASS (T≈0.416). Published argo16 bar was random-split; chrono bar must be re-established (chrono-trained baseline or in-head skill recovery) before opt-2 alone opens Phase 4/5.

**In-head priority (plumbing track, after densonly/shift diags):** (1) **low-rank δσ₀** — PCA on train `(σ₀ − clim)` (not a-space; softplus⁻¹ ceiling), predict ~16 scores, decode σ̂₀, isotonic at eval (PASS 2026-07-16: T 0.562 ≤ floor 0.590); (2) **month-resolved / harmonic clim** (JJA clim error peaks); (3) loss already in σ₀ space — do not move it to a-space. SSH→density ablation last. Keep **v10 spice frozen** / spice-continue after density.

**Process caution (Phase 5 fairness):** prefer procedures (EMA-normalized per-branch losses, sequential schedules, low-rank δa with shared PCA protocol) over representation-specific λ / weight magic numbers.

**Acceptance:** round-trip test green; T1-D style truth-projection through the full 3.2+3.3 parameterization reports its RMSE cost; one smoke training run (`config_argo_densityspice_smoke.json`, 2 epochs) completes; `selfcheck.py` extended with the round-trip check.

---

## Phase 4 — Probabilistic head and uncertainty-aware training

**Objective:** replace point outputs with calibrated distributions; wire the Phase 2 error fields into inputs, augmentation, and loss.

### 4.1 Head architecture

On the shared backbone trunk, two output branches per target block (density control vector, spice PCs):
- `mu`: as in Phase 3.
- `sigma = softplus(raw_sigma) + sigma_min`, `sigma_min = 1e-3` (standardized units). For the density block, σ is predicted on the **increments** `softplus(a_j)` domain? No — keep it simple (ponytail): predict σ per control-grid *level value*; the induced covariance handles structure (4.4).

### 4.2 Losses (implement all three behind a config switch `loss.prob_mode`)

- `"nll"` — Gaussian NLL with β-stabilization (Seitzer et al., β = 0.5):
```
L = mean( sg(σ^{2β}) · [ (y−μ)²/(2σ²) + ½ log σ² ] )      sg = stop-gradient
```
- `"crps"` (**default**) — closed-form Gaussian CRPS from §0.2(a), torch implementation, mean over dims.
- `"quantile"` — predict Q = 9 quantiles τ ∈ {0.05,…,0.95} via cumulative-softplus (non-crossing by construction: q_1 unconstrained, q_{i+1} = q_i + softplus(r_i)); pinball loss `mean_τ mean( max(τ(y−q), (τ−1)(y−q)) )`. Fit N(μ,σ) to the quantiles post hoc for covariance export (least squares on Φ⁻¹(τ)).

### 4.3 Two-stage schedule (avoids the NLL/CRPS variance-collapse pathology)

Stage 1: train μ only, σ frozen at per-level train-residual std of the Phase 3 deterministic run (or 1.0 if none), MSE loss, until early stop. Stage 2: unfreeze σ branch (μ branch LR × 0.1), switch to prob_mode loss, train until early stop. Both stages logged as separate run_ids sharing a parent tag.

### 4.4 Latent → profile covariance export (the DA deliverable)

For spice block (and any PCA block): predicted diagonal latent variance induces a full vertical covariance:
```
Σ_τ = V_τ diag(σ_z²) V_τᵀ  (de-standardized) + diag(floor)
```
For the density block, propagate through the (linearized) cumulative map: with `c_j = softplus(a_j)Δz̃_j` and Jacobian `G` of the cumulative sum + interpolation, `Σ_ρ ≈ G diag(σ_a²) Gᵀ`. Export per-profile `(Σ_T, Σ_S)` via the inversion Jacobian from 3.4 (delta method: `Σ_{T,S} = J⁻¹ Σ_{ρ,τ} J⁻ᵀ` per level; cross-level structure carried by Σ_ρ, Σ_τ). Provide `export_profile_covariance(model, batch) -> (mu_T, mu_S, Sigma_T, Sigma_S)` in `evalphys`-adjacent module `NeSPReSO2_onTemplate/dacov/`. Tests: PSD check (min eigenvalue > −1e-8 after floor), and the diagonal must reproduce the MC variance from 200 latent draws within 15%.

### 4.5 Input-error conditioning

Config flag `io.use_error_channels: true` appends the Phase 2.2 normalized error + missing-mask channels to the patch input. This is an ablation axis (Phase 5), default ON for prob heads.

### 4.6 Input-noise augmentation + aleatoric/epistemic split

Training-time augmentation (flag `augment.input_noise_alpha`, default 1.0; 0 disables):
```
x' = x + α · η ⊙ e_field ,   η ~ N(0, I)     # e_field in physical units, applied before standardization
```
Inference decomposition (script `scripts/uncertainty_decomposition.py`): draw M = 50 input-noise realizations;
`total_var = mean_m(σ_m²) + var_m(μ_m)`; the second term is the input-driven component. Report its fraction by region/season — this is a figure for the paper.

### 4.7 Target-error attenuation

Fold Phase 2.4 target errors into the predictive variance during loss evaluation:
```
σ_tot² = σ_pred² + σ_target²        (per level / per PC after projection: σ_z,target² = diag(Vᵀ Σ_target V))
```
Use σ_tot in NLL/CRPS. At eval time report both raw σ_pred calibration and σ_tot calibration; the DA export uses σ_pred (the target noise is not the product's uncertainty).

### 4.8 Evaluation

Frozen evalphys only. Required table: {CRPS, ENCE, PIT sup-dev, spread-skill slope, σ–|err| Spearman} × strata {depth band × season × input-error tercile}. Success anchors: ENCE < 0.20 (prospectus), Spearman ≫ 0.12 (session baseline), PIT visually/statistically uniform in the well-sampled strata.

**Acceptance:** all three loss modes train on the smoke config; two-stage schedule reproducible from one launcher script; covariance export tests green; decomposition script produces the fraction plot.

---

## Phase 5 — Consolidation and the ablation matrix

**Prerequisite (R4 / audit):** root-cause and either re-derive or formally waive the `test_combined_pca_loss_v2` combined/weighted_mse golden drift (fails identically on pre-Phase-0 `820e598`) **before launching the matrix**. PCA recon heads still match; the unexplained combination-term drift must not propagate as a silent question mark across every matrix cell.

**Objective:** one backbone, one pre-registered comparison, archive the sprawl.

### 5.1 Kill list

Keep: the point+patch backbone (the residual-model trunk minus the residual gating). Archive (move configs to `config/archive/`, keep checkpoints per `saved/README.md` retention policy): golden/anom/point_cube/residual/patch_l4/field variants. `eval_matched.py` remains the only cross-tag comparator.

### 5.2 Matrix (pre-registered; write `reports/ablation_preregistration.md` BEFORE launching)

```
representation ∈ { A: separate T/S PCA (baseline), B: joint T/S EOF, C: monotone-ρ + spice }
head           ∈ { det-MSE, CRPS (default prob), NLL-β }        [quantile: only on the winning representation]
error-channels ∈ { on, off }   — only for the winning (representation, head) cell, 2 extra runs
seeds: 3 per cell  ⇒  9 cells × 3 = 27 runs + 2 error-channel + 3 quantile ≈ 32 GPU runs
```
Identical data, split, backbone width, epochs/early-stop policy, and loss-scale derivation procedure across cells. Launcher: extend `scripts/launch_dual.sh` pattern to a matrix launcher writing one manifest.json. Results: `scripts/results_table.py` extended to emit the full evalphys table per cell, mean ± std over seeds.

Pre-registered readouts: (1) violation rate (A,B expected > 0; C ≡ 0 — report A,B rates and C's RMSE/sharpness cost); (2) T/S RMSE by depth band; (3) CRPS + ENCE + Spearman; (4) MLD & D26 RMSE; (5) dρ/dz RMSE. Decision rule for the dissertation's default model: best CRPS among cells with ENCE < 0.20, ties broken by dρ/dz RMSE.

**Acceptance:** preregistration file committed before first launch (check git timestamps); matrix manifest + results table committed; a one-page `reports/ablation_summary.md` interpreting outcomes against the pre-registered rules.

---

## Phase 6 — Toy OSSE + ISOP/MODAS-class baseline + error-structure deliverable

**Objective:** close the loop the dissertation exists to close, at toy scale: do calibrated NeSPReSO casts beat (i) nothing, (ii) climatology casts, (iii) an ISOP/MODAS-class synthetic-cast baseline, inside the same assimilation update — and does the calibrated R matter?

### 6.1 Truth, background, observations

- **Truth:** ISAS20 gridded (σ₀-consistent) monthly fields over the GoM box, held-out years (test era). Working grid: coarsen to 0.5°, native depth levels subsampled to ~60.
- **Background x_b:** monthly climatology (train-era mean per calendar month) — deliberately poor, so increments matter.
- **Observation locations:** real 2021 ARGO positions (from the cache), one analysis per month.
- **Casts y:** four sources at those positions: (i) none; (ii) climatology profile; (iii) MODAS/ISOP-class baseline (6.2); (iv) NeSPReSO-v2 (winning Phase-5 model) fed the real surface fields at those positions.

### 6.2 ISOP/MODAS-class baseline (fair, simple, citable)

Per-location regression in the spirit of Fox et al. (2002)/Carnes: predict the joint-EOF PC scores from `(SLA, SST_anom, month harmonics)` by ridge regression fit on the train era, decode to profiles. Depth-dependent observation error for its casts = its per-level test RMSE (the Dai et al. 2022 convention). ~200 lines; one self-check comparing its GoM RMSE order of magnitude to the published NeSPReSO-paper MLR baseline.

### 6.3 Analysis update (OI / one-step 3D-Var, no time stepping)

Column-wise (univariate in T and S after inversion; vertical-only B and R — ponytail v1; note the ceiling and the upgrade path = horizontal spreading):
```
x_a = x_b + B Hᵀ (H B Hᵀ + R)⁻¹ (y − H x_b)
B: vertical covariance = σ_clim(z) c(z,z') σ_clim(z'),  c = Gaussian, L_v = 150 m   (frozen)
H: interpolation truth-grid → cast levels
R: per experiment —
   R_fixed  = diag(depth-dependent test RMSE²)            (the Dai et al. convention)
   R_cal    = predicted Σ_T / Σ_S from Phase 4.4 (full vertical covariance, floored)
```
Analyses computed at cast columns, then spread to the grid with a fixed horizontal Gaussian (L_h = 100 km) for map-level scoring — same operator for every experiment, so it cancels in comparisons.

### 6.4 Experiment table (pre-register in `reports/osse_preregistration.md`)

```
E0 no casts | E1 climatology casts | E2 ISOP-class casts, R_fixed | E3 NeSPReSO casts, R_fixed
E4 NeSPReSO casts, R_cal | E5 NeSPReSO casts, R_cal, only casts whose predicted σ < threshold (QC use of calibration)
```
Score vs truth: subsurface T RMSE by depth band, MLD and D26 error, stratified by a Loop-Current-activity index (SSH-anomaly variance in the LC box; active vs quiescent months). Primary claims to test: E3 > E2 (beats the operational-class synthetic-cast paradigm), E4 ≥ E3 (calibration buys analysis skill), E5 behavior (calibrated σ as QC).

### 6.5 Error-structure characterization (standalone deliverable regardless of E-outcomes)

From the Phase-5 winning model on the test split: (a) representativeness proxy = per-level std of (model − colocated ISAS20) minus instrument error; (b) vertical error correlation matrices (empirical, from residuals) vs the model-predicted Σ (4.4) — plot both, report Frobenius agreement; (c) horizontal error correlation length by variogram of residuals at fixed depths; all stratified LC-active/quiescent and by season. Output: `reports/pseudoobs_error_structure.md` + NetCDF of the matrices. This is the "what DA centers need before ingesting ML casts" chapter.

**Acceptance:** preregistration precedes runs (git timestamps); all E-experiments reproducible from one script + config; report with the E-table and the 6.5 deliverable committed.

---

## Cross-cutting rules for the agent

1. **Never edit `evalphys` semantics after the v1.0.0 tag** (bug fixes only, with regression tests + changelog).
2. **Verify dataset variable names by opening files**; record actual names in `reports/data_census.md`. No hardcoding from this plan.
3. Every non-trivial numerical unit ships one runnable check (assert-based or a small pytest file). `selfcheck.py` stays green at every phase boundary; run it via `srun --ntasks=1 --cpus-per-task=8 python3 selfcheck.py`.
4. Seeds: 42 everywhere unless the matrix specifies {42, 43, 44}. Chronological splits only. Cross-tag comparisons only through `eval_matched.py`.
5. Formal product errors are relative indicators (learned scaling), never absolute variances — this caveat must appear in any doc/figure that uses them.
6. After each phase: update `HANDOFF.md` (status + next task) and append a dated entry to `reports/` — the reports are dissertation raw material.
7. If a decision gate fires (T1 mechanism refuted, T2 stale gate, 3.6 fallback), STOP and surface it to the human before proceeding.
