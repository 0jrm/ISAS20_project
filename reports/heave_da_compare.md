# Heave family vs A×CRPS for TSIS ingest

Chronological test, n=623, GoM ARGO. This note aggregates the native-z ablation in [`heave_ablation_compare.md`](heave_ablation_compare.md) with the 41-layer Dai σ_o after TSIS’s H in [`sigma_o_hycom.md`](sigma_o_hycom.md). It is the DA-facing read of that work.

**Headline.** TSIS needs a per-layer scalar σ_o(k) on HYCOM-layer T and S, written into `err` as diagonal R. It does not ingest a vertical covariance. Depth-pooled 1 m RMSE is the wrong statistic. The usable product is chronological held-out RMSE of H(T̂)−H(T_Argo) on 41 layers, floored at 0.05 °C / 0.02 psu. A×CRPS is the frozen ingest cell. HeaveFast is the heave-family challenger with the same product. ops is the only ablation that contests HeaveFast on Loop Current thermocline σ_o and D26. conv3, bathy, and bathy+wind do not.

---

## What TSIS actually uses

TSIS remaps 1 m profiles with `layer_sample` in the GOMb0.04 ingest path. Interfaces `p_ifc` come from the background `thknss` column (pressure thickness / ONEM=9806). That H is not a fixed z table. A 3 °C jump on the 1 m grid can average down across a thick hybrid layer. That is why 50–200 m died when a sharp thermocline met a thick layer.

This session scored **reference-H**, not live `thknss`. Interfaces are the 2024-01-05 18Z drifted GOMb0.04 archive at 9 GDAC sites, plus a mean `p_ifc` for casts that do not match a site within 0.05° ([`NeSPReSO2_onTemplate/data/hycom/H_OPERATOR.md`](../NeSPReSO2_onTemplate/data/hycom/H_OPERATOR.md)). Label is `h_kind=reference`. A live TSIS cycle must re-extract `thknss` at the cast.

σ_o(k) = max(RMSE_k after H, floor). Floors are the Argo analysis limits already used for real profiles (0.05 °C, 0.02 psu). Deep k=32 sits on the T floor for every model here. It is not the random-split v1 file (σ_o(1800 m)=0.013 °C). Layers k≥33 are empty because H stops at ZMAX=1800 m.

A full localized R already lost the column OSSE (E4 0.616 vs diag 0.546). Do not rebuild it. CRPS-head σ is not this product until per-band ENCE < 0.20. A×CRPS physical ENCE(T)=0.236. HeaveFast ENCE(σ_D26)=0.52.

---

## Shared backbone

Every model in this note is a `PatchConvMLP`. Encoder scalars go through `Linear(n_enc, 128)`. Satellite extras go through a point linear map, or a small Conv2d when `patch_shape` is set. The two 128-d streams concatenate into an MLP with head layers `[1024, 1024]`, dropout 0.2. Probabilistic heads emit μ and σ (`softplus`, σ_min=0.001).

Adam, lr=0.001, batch 512, chronological 70/15/15, early_stop=500, seed 42 unless noted. GoM ARGO from the v2 pickle. Pair each checkpoint with the cache it was trained on.

The fork is the **output geometry** and the **surface extras**, not a new network class.

---

## A×CRPS (frozen Phase 5/6 cell)

**Role.** Next OSSE xb and the first TSIS ingest file.

**Architecture.** `PatchConvMLP` in point mode. Outputs 32 PCA scores (16 T + 16 S) on the native 1 m grid. Decode is the cache PCA inverse. No warp. Heteroscedastic CRPS in PC space (`loss_config.mode=combined`, `prob_mode=crps`). Protocol v2 stage-2 stops on val ENCE, not val loss. Three seeds (42, 43, 44). Stage-2 checkpoints `p5_A_CRPS_v2_s{42,43,44}_s2`.

**Inputs (9-d).** Cyclic time, lat, lon (6) plus point SSS, SST, SSH (3). No ONI/RONI. Cache hash `train_ready_3adcff404b0b.pkl` once the empty `cache_path` resolves.

**Design choice.** z-PCA-16 is the T1 representation of *shape*. Frozen OSSE E3 used this cell. Pedigree is the reason it stays the first ingest even when HeaveFast wins native T and D26. Physical ENCE(T) fails 0.20, so ingest R is Dai RMSE after H, not the CRPS head.

**Native test (s42).** T 0.562, S **0.091**, 50–200 T 1.215, D26 19.05 m, MLD 36.8 m, N² profile 0.39, N² level 0.0029.

---

## Heave family (shared science)

**Architecture.** `HeaveResidualFast` is an empty weight clone of `HeaveResidual`. The backbone is still `PatchConvMLP`. Output layout is 35-d: warp (3) + residual T PCs (16) + residual S PCs (16). Residual PCs live on a canonical z-grid, not native-z PCA. Decode is `decode_warp` (exp MLD/D26, raw=0 → 50 m / 120 m) then unwarp of the residual onto physical z. Loss is `heave_residual_fast`. CRPS on warp in metres (`heave_geom_scale=10`), residual PCs, and a weak dT/dz term (`heave_dtdz_scale=0.1`).

`HeaveResidualFastLoss` batches the warp `searchsorted`. Same numerics as the Python-loop s42d run, ~2.5× cheaper per epoch. Drop s42d. Do not treat Fast vs s42d as an architecture bake-off.

**Shared heave inputs.** Cyclic time/lat/lon plus ONI/RONI (8 encoder dims when spliced) plus point SSS/SST/SSH. Ablations pin `n_enc=11` so those three local sat scalars also sit in the encoder half. HeaveFast itself uses `n_enc=8`, `n_sat=3`, `input_dim=11`.

**Shared heave failure mode for OI.** Profile N² ~0.97–0.99 (hair-trigger at 1e-8). Level N² ~0.008–0.011, still ~3× A×CRPS (0.0029). Warp-of-climatology is a worse T1 than z-PCA-16 (truth-through-warp D26 61 m vs PCA-16 8.3 m in 50–200 m). SLA already owns large heave. The leftover is shape.

Flavors below only change **what is concatenated on the surface**. Same warp head, same residual PCA, same loss, seed 42, patience 500.

### HeaveFast (challenger)

Point HeaveResidualFast. Cache `train_ready_3adcff404b0b.pkl`. 2091 epochs, 3.75 s/epoch.

Native. T **0.550**, S 0.092, 50–200 T 1.174, D26 18.46 m, MLD 33.3 m.

### conv3 (ablation)

3×3 spatial at 1°, T=t−2…t0, channels SSS/SST/SSH. Flattened sat block is 3×3×3×3=81. `patch_shape=[3,3,3,3]`, `n_enc=11` (local sat in the encoder), `input_dim=92`. Conv2d trunk on the patch. Cache `train_ready_heave_conv_3x3.pkl`. 836 epochs, 1.50 s/epoch.

The design question was whether a local advisor patch beats point sat. It does not. Native T 0.571, 50–200 T 1.214, D26 18.64 m. Pooling still costs.

### ops (ablation, LC contest)

19 point operators from the cube cache, stacked on the 11-d encoder. Gradients of SST/SSS/SSH at local and 1° scales, SSH Laplacian at 1°, SST/SSH 7-day tendency, geostrophic u/v at local and 1°. `n_sat=19`, `input_dim=30`, no conv. Cache `train_ready_heave_ops.pkl`. 714 epochs, 1.51 s/epoch.

Native. T 0.549 (noise vs HeaveFast), D26 **18.27 m**, 50–200 T 1.184.

### bathy (ablation)

GEBCO depth at the cast, `max(0, −elevation)` from the ARGO HDF5 center pixel. `n_sat=1`, `input_dim=12`. Cache `train_ready_heave_bathy.pkl`. 704 epochs.

Native. T 0.584, D26 20.80 m. Real bathy hurt.

### bathy+wind (ablation)

Bathy plus NBS daily `u_wind`, `v_wind`, `windspeed` bilinear at t0. `n_sat=4`, `input_dim=15`. Cache `train_ready_heave_bathy_wind.pkl`. 698 epochs, 2.15 s/epoch.

Native. T 0.561, D26 19.84 m, MLD **32.7 m**. Wind recovered some T and the best MLD. It did not recover D26.

---

## Native hydrography (not R)

From [`heave_ablation_compare.md`](heave_ablation_compare.md). 1 m RMSE vs Argo. Secondary for DA.

| run | T | S | T 50–200 | D26 m | MLD m | N² level |
|---|---:|---:|---:|---:|---:|---:|
| A×CRPS s42 | 0.562 | **0.091** | 1.215 | 19.05 | 36.8 | **0.0029** |
| HeaveFast | 0.550 | 0.092 | 1.174 | 18.46 | 33.3 | 0.0082 |
| conv3 | 0.571 | 0.093 | 1.214 | 18.64 | 33.2 | 0.0108 |
| ops | **0.549** | 0.092 | 1.184 | **18.27** | 33.4 | 0.0099 |
| bathy | 0.584 | 0.096 | 1.251 | 20.80 | 33.3 | 0.0097 |
| bathy+wind | 0.561 | 0.093 | 1.203 | 19.84 | **32.7** | 0.0093 |

A×CRPS 3-seed native 50–200 T is 1.215 / 1.189 / 1.216 (D26 19.05 / 18.91 / 18.77 m). Heave flavor T deltas of 0.001 sit inside that seed spread. D26 and 50–200 T are the ranking numbers, not column RMSE.

---

## σ_o after H (the ingest statistic)

Dai RMSE of H(pred)−H(Argo) on 41 layers, floor 0.05 °C. A×CRPS is the 3-seed mean. Heave flavors are s42. Full vectors in [`sigma_o_hycom.md`](sigma_o_hycom.md).

**Thermocline layers (zmid 50–200 m), σ_T °C**

| k | zmid | A×CRPS | HeaveFast | conv3 | ops | bathy | bathy+wind |
|---|---:|---:|---:|---:|---:|---:|---:|
| 10 | 56 | 1.309 | 1.255 | 1.258 | **1.208** | 1.238 | 1.231 |
| 13 | 80 | 1.335 | 1.268 | 1.319 | 1.269 | 1.350 | 1.315 |
| 18 | 124 | 1.210 | **1.178** | 1.218 | 1.200 | 1.276 | 1.230 |
| 21 | 199 | 1.014 | **1.004** | 1.054 | 1.022 | 1.098 | 1.022 |

Basin-wide thermocline mean σ_T is 1.25 (A), 1.20 (HeaveFast), 1.21 (ops). HeaveFast still wins the column after H. ops wins the top of the thermocline (k=10–11) and the Loop Current box.

**Loop Current (24–28°N, 88–84°W, n=240) vs complement (n=383), thermocline mean σ_T**

| run | LC | complement |
|---|---:|---:|
| A×CRPS s42 | 1.072 | 1.347 |
| HeaveFast | 1.051 | 1.277 |
| conv3 | 1.109 | 1.315 |
| ops | **1.024** | 1.304 |
| bathy | 1.200 | 1.317 |
| bathy+wind | 1.127 | 1.291 |

Complement is ~1.28–1.35 °C for every model. The skill gap is in the LC box. ops is the only flavor that beats HeaveFast there. bathy is the worst LC table.

Deep layers hit the 0.05 °C floor at k=32 for every run. σ_S after H is in the csv. Thermocline σ_S is ~0.14–0.19 psu, then the 0.02 psu floor from ~860 m.

---

## Recommendations for DA experiments

**1. Ingest file for the first TSIS cycle.** A×CRPS Dai σ_o after H, diagonal, floors 0.05 / 0.02. Use the 3-seed mean table in `reports/sigma_o_hycom.csv` (model `A_CRPS`, regime `all`). This is the continuation of frozen E3. Do not feed 1 m RMSE. Do not feed the random-split v1 file. Do not feed dense Σ.

**2. Frozen OSSE xb stays A×CRPS.** Level N² 0.0029 and the existing E-table pedigree outweigh HeaveFast’s 0.012 °C native T win. A new heave xb is a new prereg, not a continuation.

**3. Challenger cell (optional, separate prereg).** HeaveFast xb, **Dai σ_o after H** as diag R, same 2021 cast-column protocol. Do not use ENCE-failing σ_D26 as R. Promote only if 50–200 m T and D26 beat E3 **and** the analysis is not worse than xb. Do not compete with SLA on heave.

**4. Second challenger only if you split regimes.** ops is not a better basin ingest than HeaveFast (thermocline σ_T 1.21 vs 1.20). It is the better LC table (1.024 vs 1.051) and the better D26 (18.27 vs 18.46 m). If TSIS can take two `err` profiles, ship HeaveFast (or A) for complement and ops for the LC box. If it can take only one, keep HeaveFast as the heave ingest and ignore ops. Do not promote conv3, bathy, or bathy+wind.

**5. H at run time.** Scorecard reference-H is honest for this report. A TSIS experiment must call the same `layer_sample` on the cycle’s `thknss` column. Using `scorecard_reference_p_ifc` as climatology-H for a Loop Current OSSE will not predict the 4 °C thermocline failure.

**6. Two σ_o tables if cheap, not a covariance.** Carry LC vs complement as different `σ_o(k)` vectors. That is the regime split this data supports. Shelf is missing (`bottom_depth` not in the cache). A low-rank thermocline model σ_T(z)≈|T_z|σ_η+σ_res is the next table after Dai, fitted once on this split. Ship it only if it is larger in 50–200 m and not smaller at 1800 m.

**7. Do not ingest CRPS σ yet.** Recalibrate until ENCE < 0.20 by depth band, with inflation in 50–200 m where residual_cube / A physical σ was under-dispersed. Then compare that vector to Dai. Until that gate passes, Dai after H is R.

**8. Drop from the DA queue.** Latent and Direct (no σ, level N² ~45%). Heave s42d (superseded by Fast). conv3 (pooling tax). bathy and bathy+wind (GEBCO hurt D26). Full localized R.

---

## How to reproduce

```bash
cd NeSPReSO2_onTemplate
python3 selfcheck.py test_h_operator_layer_sample test_h_operator_rejects_blkdat test_sigma_o_floor
python3 scripts/export_sigma_o.py
```

Registry is `CANDIDATES` in `scripts/export_sigma_o.py`. Each heave flavor is paired with its own config and cache. A×CRPS uses the Phase 5 matrix config and stage-2 checkpoints.
