# Yearly handoff — calendar 2026

**For:** the next agent, and you.  
**Not:** a replacement for [`HANDOFF.md`](HANDOFF.md) (that file stays the short *session* pointer).  
**Window:** January–August 2026. Commits in this repo start mid-June. Claude Code history for this tree is **1–20 July** (plus 31 Aug UI only); per-session transcripts were cleaned 31 Aug 2026, so July method is reconstructed from dated `HANDOFF-2026-07-*.md`, `PLAN-*.md`, and Claude `memory/` notes. August lives in Cursor sessions and the reports in §8.  
**Code home:** [`NeSPReSO2_onTemplate/`](NeSPReSO2_onTemplate/). Conda: `nespreso`.

This note is the 2026 story: **what the project is for**, **how it is allowed to claim anything**, and **which later choices that method forced**. Numbers appear only when they are the claim. Paths, checkpoint hashes, and figure-by-figure recaps live in the reports linked at the end.

---

## 1. Objective

**NeSPReSO** (Neural Sparse Profiling of the Subsurface Ocean) maps satellite surface fields at a location and time to a full-depth **temperature and salinity profile**. The 2026 dissertation branch is Gulf of Mexico only. **ARGO/CORA profiles are the truth.** Gridded L4 products (ISAS, OSTIA, DUACS, …) are allowed as *inputs*, *baselines*, or *augmentation* — not as a silent substitute for the float.

Two nested questions drove the year:

1. **Reconstruction.** Given only surface maps (and calendar/location encodings), how well can a small network recover T(z) and S(z) at a float, under a split that does not leak the future into training?
2. **Pseudo-observations.** If those profiles (and a stated uncertainty) are treated as extra casts in a simple analysis — later, in TSIS/HYCOM — does the analysis get closer to real floats than a classical satellite-to-profile method (ISOP/MODAS-class)? And does *learned* uncertainty help the blend?

By late July, question 2 had a **clean negative** in the toy (cast-column) testbed. August repeated it in live HYCOM–TSIS cycles: ordinary satellite+Argo DA beats a null run; NeSPReSO-as-obs does not. The rest of the year is the method catching up: stop selling pooled RMSE as a DA trophy, stop writing CRPS-head covariances into observation error, and score what an ocean model actually sees (thermocline, layer-mean after HYCOM’s vertical remap, stability on every 1 m, independent-era Argo vs the DA background).

The defensible 2026 contribution is therefore not “calibrated ML improves DA.” It is the **evaluation standard + the bake-off + the explained negatives**, plus the August DA-facing hydrography work that those negatives forced.

---

## 2. Method axioms (why later results are even comparable)

A result here is only usable if **split, target, surface realism, missingness, and scoring** are defensible. Architecture experiments were explicitly *not* the first job.

**Truth and grids.** Train and headline-eval against ARGO on the native 1 m (0–1800 m) grid. Do not put raw `eval_run` RMSE from an ISAS (187-level) run next to an ARGO run. Pair every checkpoint with the cache it was trained on; mixing PCA bases across caches is a silent wrong model.

**Time.** Random train/val/test mixing is legacy/ablation only. Dissertation default is **chronological**. The GoM export starts in **2015**, not 2002, so the prospectus-style 2002–2015/2016–17/2018–20 split is empty. Default B is a 70/15/15 date-ordered cut (~2901 / 621 / 623 profiles). High-density 2020 and sparse 2015–2018 exist as named stress subsets; the 2022 tail is thin.

**Ruler before model.** Metrics (`evalphys`) were frozen before representation work. The skill gate is “within 10% of the same-split argo16 baseline,” not a published random-split number. A leaked chrono eval of a random-trained checkpoint was caught and the floor restated (clean chrono raw ≈ 0.54 °C; operative floor ≈ 0.59 °C). That erratum is first-class: several branch decisions would have been wrong against the old ruler.

**Pre-register, then fail honestly.** The 3×3 representation×head matrix and the OSSE E-table were locked before looking at test. Stage-2 of a probabilistic head **stops on validation calibration (ENCE)**, not on loss — stopping on loss under-trains spreads. After decode, rank cells in **physical T/S**; CRPS in different PCA/EOF bases is not the same currency.

**What “calibrated” means.** ENCE asks whether stated spreads are the right *size* (gate: &lt; 0.20). Spearman asks whether large σ ranks hard casts. CRPS scores the whole distribution. Passing those in **PC space** is not passing them in **°C**, and passing a pooled T+S number is not passing **temperature in the thermocline**.

**What DA is allowed to ingest.** After the OSSE, the product is a profile plus a **diagonal** observation-error story. Dense vertical R from this CRPS head is not used. CRPS-head σ is research-grade until per-band ENCE(T) clears 0.20. Operational ingest uses **Dai-style σ_o**: chronological held-out RMSE after HYCOM’s layer operator H, floored at Argo analysis limits, not 1 m pooled RMSE and not a full Σ.

---

## 3. How a sample becomes a profile

At a float (lat, lon, time):

1. Read surface fields. The July default is still a **point** model: six Fourier encodings of time/lat/lon plus three L4 scalars (SSS, SST, SSH) — nine numbers. Heave-family cells splice **ONI/RONI** after the harmonics (eleven numbers). Ablations may add cube operators, a 3×3 patch, bathymetry, or wind; those are not the frozen ingest cell.
2. **SatEncoder** (`PatchConvMLP`): linear embed of encodings + satellite block (point linear, or a small conv if a patch is configured), MLP to a latent vector.
3. A **head** predicts a short vector of latent scores. Decode back to T(z), S(z). Probabilistic heads also emit a per-score spread σ.
4. Training truth is always the ARGO profile.

The scientific bets in 2026 were almost entirely **target representation** and **loss**, not a new backbone.

| Representation | What is compressed | Decode |
| --- | --- | --- |
| **A** | Separate PCA on T and on S | Inverse PCA |
| **B** | One joint EOF on concatenated [T; S] | Inverse EOF |
| **C** | PCA on density (σ₀) and spice | TEOS-10 inversion; isotonic density at inference |
| **Heave** | Warp landmarks (MLD / thermocline depth) + residual PCs on a canonical z-grid | Warp climatology, then unwarp the residual onto physical z |

Heads: **det** (MSE on scores), **CRPS**, **NLL**. Two-stage probabilistic training: fit μ, then σ. Validation-only rescaling of σ is allowed; test-set tuning is not.

**Datacube.** A regional daily Zarr store (SST, SSS, SSH, bathymetry over the GoM box) replaced per-cast HDF5 patches. The trigger was not “patches need a nicer store.” Early July L4-patch chrono comparisons were invalid: satellite archives ended ~January 2021, and nearest-file clamp filled val/test with **stale** maps (test ~100%). After a real rebuild through the test window, patch still did not beat point. The cube’s job is to make missing days stay empty instead of silently clamping, and to let feature recipes (gradients, tendency, geostrophy) change without NetCDF recuts. Sampler goldens must carry a cube `data_revision`; they rot silently across rebuilds. It is **plumbing**. The July winner is a point model; the cube is not evidence that patches beat points.

A **point-anchored residual** was designed so the patch is a strict input superset of the point encoder and cannot underperform it “in principle.” After the sat fix it added **no** test skill. The residual’s frozen point block must see the **same z-scored inputs** as the point cache; same-input weight tests that skip that contract are vacuous.

L3/masked-input rasterization (value, mask, age, uncertainty, count around each float) was built as the dissertation *data* path. SST L3U and SMAP SSS rasterization stayed deferred. The locked 2026 scoreboard still uses L4 point scalars, not the L3 tensor.

---

## 4. What the year actually did

Two movements. [`PLAN.md`](PLAN.md) is the original data-foundation charter (census → chrono split → ARGO-first → L3 masks → later physics/ensembles). **Do not treat its Phase 4–10 checklist as current status.** July executed a different recovery plan ([`PLAN-v2-recovery.md`](PLAN-v2-recovery.md)): freeze eval, fix representation, run a fair matrix, then an OSSE. August is DA-facing work after that OSSE.

### June–July — lock the experiment, then close two questions

Port v2 into this template (dual ISAS/ARGO path, same backbone). Census the actual GoM ARGO years. Make chronological split the ARGO default. Build L3 scaffolding and L4-mask augmentation with source flags, so L4 cannot pretend to be L3. Diagnose L4-patch collapse as **stale satellites + unstandardized inputs**, not a missing deeper net. Do not trust patch-vs-point numbers from before the archive/clamp fix.

Then the recovery:

- Truncated PCA **manufactures** static-stability inversions (~1% on raw ARGO → ~22% on the PCA-16 *target*). Soft basis changes (joint EOF, density/spice) do not cut that. A hard monotone density constraint does. A mislabeled “nature 25% inversion” rate was the PCA-16 reconstruction, not the ocean — so a physics *loss* on already-over-smoothed models was retired (PLAN Phase 8 **closed**).
- Compress density in **physical σ₀ space**, then constrain; compressing in the nonlinear preimage (“a-space”) reconstructs worse than climatology.
- A density/spice recipe that looked good in a **single run** missed the skill floor under a locked 3-seed protocol. Nonlinear profile autoencoders improved compression and did **not** transfer to surface→profile skill; PCA stayed.
- Fair 3×3: **B×det** is the best thermometer; **A×CRPS** is the probabilistic default (best physical CRPS among cells that clear pooled ENCE); **C lost both** skill and physical calibration. PC-space calibration of A×CRPS is fine; **ENCE(T) fails** in the surface and thermocline, especially summer.
- Cast-column OSSE (2021 floats, climatology background, vertical-only OI): NeSPReSO **ties** ISOP on fixed diagonal R and does not beat it. A diagonal learned R ties the fixed-R blend. **Full localized R from the CRPS head is worse.** The extra error is the off-diagonals — induced by sharing the PCA basis across depths, while the head was trained for *marginal* per-score CRPS, not joint observation-error structure. Dropping “uncertain” casts by mean σ sends those columns back toward climatology.

MC dropout on the deterministic point models confirmed over-confidence. Pooled spread–error rank correlation is a **depth confound** (both spread and error are large near the surface); within depth, dropout barely ranks casts. Do not write dropout spread into R. A separate inference gotcha: anomaly models in **eval mode** under-predict PC1; the dropout ensemble mean recovers a chunk of T RMSE that eval-mode hid. Before GSW on anomaly caches, reconstruct absolute profiles — `true_profiles` there are not physical T/S.

Readiness: models are **over-smoothed**, not unstable. Steric-vs-SLA is **saturated** (the network matches or beats true profiles at reproducing observed SLA because smoothing removes structure that does not project onto SSH). That diagnostic is not a search objective. Retuning anomaly `profile_scales` closed only a small slice of the point-vs-anomaly gap; **stop blaming the T:S loss knob.** Do not compare `val_loss` across scale retunes.

### August — after the negative, ask what DA actually needs

SLA/Cooper–Haines already owns large heave. Leftover skill is **shape**: spice, double thermocline, MLD independent of SSH. Insertion that argues in **z** while the rest of the system updates **heave** can look fine on pooled T and still wreck 50–200 m. The next OSSE (if run) is allowed to headline 50–200 m T, D20/D26, max-N² depth, and “analysis not worse than xb” — not depth-pooled T RMSE.

That produced a second model family: **heave + residual PCs**. Same backbone, different output geometry (warp in metres + residual scores). `HeaveResidualFast` is the same science as the Python-loop heave, cheaper per epoch so patience actually ran. It is a slightly better *hydrography* than frozen A×CRPS on native T and D26, and a worse *OI background*: level N² about 3× higher, and σ on thermocline depth is not a usable R (ENCE far above 0.20). Warp-of-climatology is a **worse** representation of truth than z-PCA-16; the leftover error is shape, which A already outputs.

Ablations that changed the method:

- A 3×3 satellite patch **loses** to point heave. Global pooling can erase the center pixel; fixing the stencil and dropping AdaptiveAvgPool still does not beat point. Spatial conv is not DA-ready.
- Nineteen cube operators (gradients, Laplacian, 7-day tendency, geostrophy) are the only extras that contest D26. They must be computed at **serve** time from the satellite archive, not assumed to live in the training zarr.
- Real GEBCO bathymetry **hurt** T and D26 (and blew loss until z-scored).
- RONI + ops + heave as one “x-tudo” cell: ops **alone** is the best of that family on T and 50–200 m ENCE; the combo barely moves pooled T, **loses to ops**, and thermocline ENCE gets worse. Do not ingest the combo.
- Native-z heads with **no PCA**: T can tie; S collapses; essentially every cast is statically unstable. Compression is still required for physical plausibility.

**Latent** (learned decoder) and **Direct** (full-depth T/S + smoother) can look good on column T. They are not DA-ready: no σ, and ~45% of *levels* violate N². Profile-violation at a hair-trigger tolerance is the wrong statistic; HYCOM integrates every 1 m.

**Physical CRPS (A_CRPS_z32).** Train the probabilistic loss on decoded T/S (with a cheap PC-space regularizer), equal T/S, band-equal CRPS, stop stage-2 on val ENCE(T). Pooled ENCE(T) can pass while 0–50 m and 50–200 m still miss 0.20 — the wrong metric for Score B. That cell is the **skill/calibration baseline**. Frozen **A×CRPS** (PCA-16, PC-space CRPS, 9-d) stays the OSSE/ingest pedigree cell until a new prereg says otherwise. Serve path: **μ only**; R is Dai σ_o, not CRPS σ. Do not replace the Ozavala `/v1_profile` SAT cell.

**H, not 1 m.** TSIS remaps 1 m profiles through hybrid-layer interfaces. A sharp thermocline on the 1 m grid can average down in a thick layer. Ingest σ_o is RMSE of H(T̂)−H(T_Argo) on 41 layers (reference-H from a drifted GOMb archive in the reports; a live cycle must use live `thknss`). Deep layers sit on the Argo floor.

**Live cycles (HYCOM + TSIS), not just column OI.** Three machines: NeSPReSO as the observation path, TSIS as the analysis, HYCOM as the dynamics. Campaigns used so far: twins near initial condition, and a drifted Gulf (short free run, then assimilate, then a week of forecast). Members: null, ordinary satellite+Argo (`ops`), NeSPReSO at floats, NeSPReSO along thinned tracks. Ordinary `ops` beats null. **NeSPReSO argo/tracks are worse than null at analysis** — the apply path is not the blamed failure; the observation is. Tracks that are a wet map rather than the float hull, and hybrid gap-fill of failed tracks, were rejected. Score B next shot, if any: **floats only**, with a HeaveFast gate, before another forecast week. `incflg=-1` is required on this ingest path.

**Independent era.** Chrono-test RMSE (2021–22 holdout, training satellites) is not the 2024–25 TSIS Argo set. On that later set, **none** of the NeSPReSO cells beat the DA background **xb** on the 2025 majority; they can beat xb on a small 2024 slice and still lose to **xa**. Do not put the chrono-test ~0.56 °C next to those RMSEs. Ingest gate: inserting these profiles as obs on 2025 cycles would pull the analysis *away* from the float unless R is huge.

---

## 5. Dead ends that explain the method (only these)

| Tried | What it taught | What we do instead |
| --- | --- | --- |
| Random-split published skill; chrono eval of that checkpoint | Leakage, not a better model | Same-split chronological ruler |
| Soft basis change to “fix” inversions | Truncation, not T/S separateness, manufactures inversions | Hard monotone constraint if stability is the claim; do not expect C to win skill |
| Physics loss on σ₀ | Models already over-smoothed; “nature 25%” was the PCA target; `point_cube` failures were a cube-SSS/halocline defect | Phase 8 closed; fix features, do not penalize smoothness |
| Nonlinear profile AE | Better compressor ≠ better surface→profile map | Stay on PCA/EOF for the locked pipeline |
| C×det single-run admission | One seed is not a matrix | 3-seed protocol; C is a stability finding, not the default |
| Deeper L4 patch nets; patch vs point on the old chrono split | Archives ended before test; nearest-file clamp → stale maps; after fix, patch still loses | Date-diff guard; cube (no silent clamp); point default |
| Point-anchored residual (“cannot underperform point”) | After sat fix, gated patch residual adds no test skill; z-score mismatch makes S0 tests lie | Point A×CRPS default; cube still serves operators/heave extras |
| Anomaly loss-scale retune as the parity fix | ~18% of the gap; remaining is anomaly framing / eval-mode inference | Stop searching T:S scales; reconstruct physical profiles before GSW |
| Stop σ training on loss | Spreads under-trained | Stop on val ENCE |
| Full Σ_T from marginal CRPS as R | Numerically localizable, scientifically worse than diag | Diagonal R; Dai σ_o for ingest |
| QC-by-σ / MC dropout as R | Upper-ocean ENCE fails; dropout rank is depth-confounded | Do not keep/drop casts by σ; do not ingest dropout |
| Conv patch vs point heave | Global pool can drop the center pixel; even a fixed stencil still loses | Point extras (ops) if anything; not conv as first cell |
| Bathy as a free extra | Hurt skill | Dropped as a win path |
| Latent / Direct as DA cells | No σ; level N² catastrophic | Hydrography toys, not ingest |
| Native-z head, no PCA | T may tie; S collapses; ~all casts unstable | Keep PCA/EOF (or heave+residual PCs) |
| RONI + ops + heave in one cell | Combo loses to ops alone; thermocline ENCE worse | Factor one at a time; do not ingest the combo |
| HeaveFast as drop-in OSSE replacement | Better T/D26, worse stability and σ_D26 | Frozen A×CRPS remains first DA cell; HeaveFast is a named challenger with the same σ_o product |
| Pooled 0–1800 m T as the DA metric | SLA already owns heave; NES-in-z vs SLA-heave wrecks 50–200 m | Thermocline / D26 / layer RMSE after H |
| Writing CRPS-σ into TSIS `err` | Physical ENCE(T) still fails 0–200 m | Dai table; promote CRPS-σ only if per-band ENCE clears |
| NeSPReSO tracks / hybrid gap-fill in a live cycle | Wet-map tracks ≠ float hull; tracks and argo members worse than **null** | Ordinary `ops` is the working DA; next NeSPReSO shot floats-only if at all |

Omitted on purpose: L3 SST download homework, agentic Track B (sampler search), datacube speed micro-opts after the cube existed, and a custom bilinear regrid meant to replace scipy interpolators (coastal NaN/bracket behavior is not safe to hand-replicate).

---

## 6. Roles as of 31 Aug 2026

| Role | Choice | Why |
| --- | --- | --- |
| Frozen DA / next OSSE xb | **A×CRPS** (9-d, PCA-16, PC-space CRPS) | Pedigree, σ head, low level N², E3 cell |
| Skill / calibration baseline for new heads | **A_CRPS_z32** (physical CRPS after decode, 32+32) | Loss lives in °C; pooled ENCE(T) can pass; 0–200 m still does not |
| Hydrography challenger | **HeaveFast** | Same Dai σ_o product, diag R; promote only if thermocline/D26 beat E3 *and* analysis is not worse than xb |
| Ingest R | **Dai σ_o after H**, diagonal, Argo floors | Not CRPS-head Σ, not 1 m RMSE |
| Not ingest | Latent, Direct, conv3, bathy, z-native (no PCA), RONI+ops+heave combo, Score B tracks, Ozavala v1 on the 2024–25 set | No σ / N² / lost to ops or to xb / worse than null in cycle |
| Cube | Regional store + operators | Data quality; not a modeling win |

**Do not compete with SLA on heave. Do not insert TSIS as training truth.**

---

## 7. Still open

- Map-level OSSE with gridded truth and horizontal scales (the July run is vertical-only, one region, toy OI).
- A head trained for **joint** covariance, if anyone still wants off-diagonal R. Marginal CRPS cannot be asked to invent that structure.
- Surface and thermocline ENCE(T), especially JJA — the interesting gate for any new cell ([`reports/NEXT_A_CRPS_z32_roni_ops_heave.md`](reports/NEXT_A_CRPS_z32_roni_ops_heave.md)).
- Live `thknss` H vs the reference-H used in reports.
- L3 SST/SMAP as model inputs (scaffolded, not on the scoreboard).
- Whether any v2 cell belongs in a 2025 TSIS cycle given the xb comparison **and** the January-style cycles where NeSPReSO lost to null. Floats-only Score B with a HeaveFast gate is the named next shot, not another track experiment.

---

## 8. Where to read next

| If you need | Open |
| --- | --- |
| Tonight’s session pointer | [`HANDOFF.md`](HANDOFF.md) (July 20 OSSE freeze — stale for August roles) |
| Locked July science (matrix + OSSE + cube, no spin) | [`reports/july_briefing/BRIEFING.md`](reports/july_briefing/BRIEFING.md), [`reports/evolution/EVOLUTION.md`](reports/evolution/EVOLUTION.md) |
| Original data-foundation charter | [`PLAN.md`](PLAN.md), [`PLAN-dissertation-data-foundation.md`](PLAN-dissertation-data-foundation.md) |
| What July actually executed | [`PLAN-v2-recovery.md`](PLAN-v2-recovery.md) |
| DA-facing August read | [`reports/v2_da_candidate.md`](reports/v2_da_candidate.md), [`reports/heave_da_compare.md`](reports/heave_da_compare.md), [`reports/osse_claim_language.md`](reports/osse_claim_language.md) |
| Independent-era Argo vs xb | [`reports/xb_argo_compare/eval_xb_argo.md`](reports/xb_argo_compare/eval_xb_argo.md) |
| Architecture slides | [`reports/maps_v2_handoff/notes.md`](reports/maps_v2_handoff/notes.md) |
| Live-cycle / Score B campaigns | [`reports/maps_v2_handoff/da_update_notes.md`](reports/maps_v2_handoff/da_update_notes.md) |
| Split / census | [`reports/split_design.md`](reports/split_design.md), [`reports/data_census.md`](reports/data_census.md) |

Dated `HANDOFF-2026-07-*.md` files are session fossils (cube, stale-sat, agentic RC tracks). Use them only to reconstruct a July week, not as current status.
