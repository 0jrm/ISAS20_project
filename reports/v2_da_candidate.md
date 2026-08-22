# Early v2 DA candidate

Chrono test n=623, cache `train_ready_3adcff404b0b.pkl`, evalphys 1.2.0.
Native 1 m vs Argo truth. LC steric-vs-ADT is **not evaluable** (cache has no `ssh_obs_sla`).

A×CRPS is the frozen Phase 6 cell. HeaveFast is a v2 DA challenger in the same workstream. Both get Dai σ_o after H, diag R, and Argo floors. Latent and Direct are not DA-ready.

## What DA needs that RMSE does not

Cast-column OSSE (frozen): E3 A×CRPS **0.545** vs E2 ISOP **0.541**; dense R hurt (E4 0.616 vs diag 0.546). Next OSSE scores **50–200 m T**, D20/D26, max-N² depth, and “analysis not worse than xb”. Diagonal R from a CRPS head. SLA already owns heave.

That implies four gates on the *background*:

1. A **σ** the OI can use as diag(R) or diag(P).
2. Columns that are not inverted on a large fraction of **levels** (HYCOM will see every 1 m).
3. Competitive 50–200 m T and D26 vs nature.
4. Not a new architecture the OSSE has never seen, unless the skill gap is large enough to pay for a new frozen cell.

## Results

| | T 0–1800 | S 0–1800 | T 50–200 | D26 m | MLD m | N² profile | N² **level** | σ head |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| **A×CRPS** (9-d, z-PCA-16) | 0.562 | **0.091** | 1.215 | 19.05 | 36.8 | **0.39** | **0.0029** | yes |
| Heave s42d (loop warp) | 0.577 | 0.098 | 1.205 | 19.85 | 33.4 | 0.99 | 0.0099 | yes |
| **HeaveFast** s42 | **0.550** | 0.092 | 1.174 | **18.46** | **33.3** | 0.99 | 0.0082 | yes |
| Latent (learned 32-d decode) | 0.546 | 0.142 | **1.170** | 19.50 | 42.1 | 1.00 | 0.47 | no |
| Direct (1801-d + FIR) | 0.541 | 0.333 | 1.189 | 19.03 | 48.1 | 1.00 | 0.43 | no |

HeaveFast MLD on test is 6–90 m (mean 34 m), not the old 10 m floor. ENCE(σ_D26)=0.52 (gate 0.20); JJA Spearman undefined. A×CRPS physical ENCE(T)=0.236 already failed 0.20 in Phase 5; Fast D26 is worse.

T1 ceilings unchanged: PCA-16 of *truth* 0.116 °C / D26 8.3 m in 50–200 m. Warp-clim with *true* landmarks 3.44 °C / 61 m. The leftover error is shape, which z-PCA already represents.

## HeaveFast vs Heave s42d

Same 11-d inputs, same exp-MLD decode, same CRPS-in-metres loss. `HeaveResidualFast` is an empty weight clone; `HeaveResidualFastLoss` batches the warp `searchsorted`.

| | epochs | wall | s/epoch | val best | test T |
|---|---:|---:|---:|---:|---:|
| s42d (Python loop) | 785 | 123 min | 9.41 | 3.909 | 0.577 |
| Fast s42 | 2091 | 131 min | **3.74** | **3.853** | **0.550** |

Epoch-1 loss matched s42d exactly. Fast is not a new scientific model. It is ~2.5× cheaper per epoch, so patience-500 ran longer and found a better val. **Drop s42d; keep Fast as the heave checkpoint.** Do not treat Fast vs s42d as an architecture bake-off.

## Why not Latent or Direct

Both beat A×CRPS on full-column T (0.546 / 0.541 vs 0.562) and Latent ties Fast on 50–200 m.

They fail DA for two independent reasons:

- **No σ.** DA here is diagonal R from a CRPS head. A deterministic 1801-d T/S vector is an xb, not an error model. Adding a hetero head later is a new train, not a free upgrade.
- **Level N² ~45%.** Profile-violation at 1e-8 is a hair-trigger (one unstable point flags the cast). Level rate is what HYCOM integrates: Latent 0.47, Direct 0.43, vs A 0.0029 and HeaveFast 0.008. Direct S RMSE 0.333 (vs 0.091) is a third disqualifier.

max-N²-depth RMSE: Latent 524 m, Direct 110 m, A 47 m, HeaveFast 44 m.

## Why A×CRPS over HeaveFast for the first DA cell

HeaveFast is the better *hydrography* of the two CRPS models: T 0.550 vs 0.562, 50–200 m 1.174 vs 1.215, D26 18.46 vs 19.05, MLD 33 vs 37, S a tie.

It is the worse *background for OI*:

1. **Stability.** Profile N² 99% vs 39% is mostly the hair-trigger. Level rate is still ~3× A (0.82% vs 0.29%). Frozen OSSE already showed that a slightly better xb does not automatically beat ISOP (E3 0.545 vs E2 0.541). Unstable 1 m structure is the thing that made dense R and naive insertion fail before.
2. **σ_D26 is not a usable R.** ENCE 0.52; σ over-dispersed (RMV 15–87 m vs RMSE 6–32 m). Phase 6 wanted **diag R**, and off-diagonals already hurt. A miscalibrated σ_η is a bad diag R.
3. **Pedigree.** A×CRPS is the frozen Phase 5/6 head. A new OSSE cell on HeaveFast is a new prereg, not a continuation.
4. **T1.** Warp-of-climatology is a *worse* representation of truth than z-PCA-16. Fast’s D26 win is real but small (0.6 m). SLA/CH already owns the large heave. The DA leftover is shape, which is A’s native output.

Fairness note: Fast has ONI/RONI (11-d); A×CRPS is 9-d. That confound does not overturn N² or ENCE.

## σ_o ingest

The DA ingest file is the 41-layer Dai table in [`reports/sigma_o_hycom.md`](sigma_o_hycom.md). Session write-up (architecture, inputs, DA recs) is [`reports/heave_da_compare.md`](heave_da_compare.md). It is not 1 m RMSE and not dense Σ.

H is reference-H from the 2024-01-05 18Z drifted GOMb0.04 interfaces (9 GDAC columns plus mean p_ifc), not live thknss. Floors are 0.05 °C and 0.02 psu. CRPS-head σ is not this product. Use it once ENCE is below 0.20 by band.

A×CRPS stays the frozen Phase 6 xb. HeaveFast is the challenger candidate with the same σ_o product, same diag R, and same floors. conv3, ops, bathy, and bathy_wind are in that table as heave-family ablations (own cache, same H and floors), not ingest, unless they beat HeaveFast on thermocline σ_T. The N², ENCE, and pedigree caveats above still apply. Promote the challenger only if 50–200 m T and D26 beat E3 and the analysis is not worse than xb. Do not insert TSIS. Do not compete with SLA on heave.

## Decision

| Role | Choice | Why |
|---|---|---|
| **DA / next OSSE xb** | **A×CRPS** `p5_A_CRPS_v2_s42_s2/model_best.pth` | σ, level N², frozen E3 cell |
| **v2 DA challenger** | HeaveFast `heave_fast_s42/model_best.pth` | same 41-layer Dai σ_o after H, diag R, floors |
| Heave-family checkpoint | HeaveFast (same file) | same science as s42d, better skill, 3.7 s/epoch |
| Not this DA | Latent, Direct | no σ; level N² ~45%; Direct S broken |
| Not this DA | Heave s42d | superseded by Fast |

Steric 2 cm LC gate: still **not evaluable** on this cache.
