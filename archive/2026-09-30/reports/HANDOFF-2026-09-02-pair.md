Historical. Not instructions.

# Full session handoff, stochastic EOF emulator through pair-mix heads

Session dates: 1–3 Sep 2026. This document covers the complete experiment path, not only the final pair experiment.

**For** the next person who has to decide what to train.  
**Split** chronological 70/15/15 on the target float date. Test n = 623 Gulf of Mexico Argo casts, cache `train_ready_4ee013852d33.pkl`, 32+32 PCA, 0–1800 m.  
**Truth** is always the Argo profile at the target.  
**Headline error** is raw T/S RMSE. Headline uncertainty mismatch is raw ENCE(T). Pass bar 0.20. Do not mix these numbers with the 2024–2025 HYCOM/xb table.

---

## Executive summary

The best typical profile error came from giving the network a real, earlier Argo profile. The best uncertainty estimate came from the satellite-only stochastic EOF emulator. A 50/50 training mix of real and model-made profiles, marked with a source flag, made the pair network safe to use with either source, but it did not beat the real-Argo pair model.

The recommended split is therefore operational, not one universal model.

1. Use the satellite-only stochastic EOF emulator when no earlier float is available.
2. Use the Argo pair model when a legal earlier float is available.
3. Use the flagged 50/50 pair-mix model only when the input pipeline may provide either a real or model-made profile.
4. Do not use the fake-only pair model or the jointly trained unfrozen encoder as the default.

The experiment also found that “make the distance variables Gaussian” is the wrong fix. The training sampler was biased toward very close and recent neighbors. The useful correction was to sample across distance and time bins while keeping the physically valid limits.

## The starting point: the stochastic EOF emulator

The original map-only model predicts a full temperature and salinity profile from nine values at the target location and time. The nine values are six sine and cosine calendar/location encodings plus satellite SSS, SST, and SSH.

The output is 64 numbers. The first 32 describe temperature and the next 32 describe salinity. They are scores in a learned basis. Decoding those scores produces values on the native 1 m grid from 0 to 1800 m.

The stochastic EOF version trains against the decoded profiles rather than only against the 64 scores. Its loss has two parts:

\[
L = L_{\mathrm{profile\ CRPS}} + 0.1 L_{\mathrm{whitened\ PC}}.
\]

The first term asks whether the predicted distribution is close to the observed Argo profile at each depth. The second keeps the short output vector tied to the score-space statistics. The network also predicts a spread \(\sigma\) for every output score.

Training has two stages. Stage 1 learns the mean profile while ignoring the spread. Stage 2 learns the spread with the CRPS loss and stops on validation ENCE. ENCE measures whether the reported error bars match the actual errors.

The map-only test result was:

| Model | T RMSE | S RMSE | raw ENCE(T) |
|---|---:|---:|---:|
| Stochastic EOF emulator | 0.545 °C | 0.089 | **0.151** |

This is the calibration reference. It is the only tested cell in this session below the 0.20 ENCE(T) bar.

## Why add a profile input

The next question was whether a nearby earlier float could supply information that satellites cannot see. For a target cast \(i\), an earlier source cast \(j\) must satisfy:

\[
3 \le t_i-t_j \le 45\ {\rm days},
\]
\[
\operatorname{distance}(i,j) \le 250\ {\rm km},
\]
\[
t_j < t_i.
\]

The target label remains the Argo profile at \(i\). The source profile is only an input. This distinction prevents the model from being rewarded for copying the source.

The pair input contains the target's nine map values, the source's 64 profile scores, and:

\[
(\Delta{\rm lat},\Delta{\rm lon},\Delta t/30).
\]

The first pair model used a 76-number input. It selected up to four legal earlier neighbors for each training target and one for each validation or test target.

Copying the earlier profile directly gave 0.685 °C temperature RMSE on average. The learned pair model gave 0.531 °C. The gain is therefore not a persistence shortcut. The model uses the earlier profile together with today's satellite values and the displacement information.

## The first pair tests

Four source arrangements separated the useful effect from a source-format problem.

| Training source | Test source | T RMSE | S RMSE |
|---|---|---:|---:|
| Real Argo | Real Argo | **0.531** | **0.085** |
| Real Argo | Frozen model-made profile | 0.536 | 0.088 |
| Frozen model-made profile | Frozen model-made profile | 0.554 | 0.090 |
| Frozen model-made profile | Real Argo | 0.866 | 0.159 |

The second row is encouraging but is not enough to justify fake-source training. It says a pair model trained on real floats can tolerate a model-made source at test time. The third and fourth rows show the limitation. A model trained only on synthetic profiles learns their numerical shape and does not learn a general “use either kind of profile” rule.

## The unfrozen encoder experiment

The side experiment kept the nine-value encoder live instead of freezing it. A 21-number input held the target nine values, source nine values, and three displacement values. The source nine values passed through a trainable encoder to make the 64 source scores.

The loss was:

\[
L_{\mathrm{total}} =
L_{\mathrm{pair}} +
L_{\mathrm{direct,target}}.
\]

The direct term forced the encoder's target branch to remain a profile predictor. This did not work as intended. The pair product scored 0.587 °C and the direct encoder path scored 0.576 °C. Both were worse than the frozen map-only reference of 0.545 °C. The two tasks pulled the shared encoder in different directions.

The lesson is specific. “Let the loss act on the first encoder too” is not a free improvement. If the pair head and the profile emulator need different representations, they need separate training or a stronger design than this small shared wrapper.

## Why the distance sampler changed

The legal source pool contains 109,977 train candidates. The old nearest-four rule retained 11,349 pairs. Its average distance was 57 km and its average time gap was 12 days. The full legal pool averaged 141 km and 24 days.

The old rule over-represented close and recent pairs. It was not a Gaussian problem. The full pool already had nearly symmetric latitude and longitude differences. Transforming those variables toward a Gaussian would have removed real spatial cases. The actual problem was selection bias in the neighbor picker.

The selected training sampler uses four distance bins and three time bins:

\[
{\rm distance}\in[0,50),[50,100),[100,175),[175,250]\ {\rm km},
\]
\[
\Delta t\in[3,10),[10,20),[20,45]\ {\rm days}.
\]

It walks through the bins in round-robin order and selects up to eight neighbors per target. This is not a claim that the ocean variables are Gaussian. It is a simple way to avoid letting the nearest bin dominate.

The census found 87.1% of train targets had at least eight legal candidates. Twenty-five targets had none and remain skipped. The census and sampler are in `scripts/pair_mix_census.py` and `preproc/pair_table.py`.

## The 50/50 pair-mix design

For each selected \((i,j)\), the loader creates two rows:

\[
(x_{ij}^{\rm Argo},f=+1,y_i),\qquad
(x_{ij}^{\rm synth},f=-1,y_i).
\]

Both rows use the same target \(i\), source location \(j\), and displacement. Only the source profile and flag change. The label is always the new Argo target profile \(y_i\).

The model input width is 77. It is the original 76 pair values plus the source flag. The frozen model-made profile comes from the stochastic EOF emulator evaluated at the source location and time. Its average score-vector length is about 0.87 times the Argo score-vector length, which explains why the flag and balanced training matter.

Training used eight stratified neighbors and the two source rows. Validation and test retained one nearest legal neighbor per target and included both source types. This keeps the test geography comparable while measuring each source kind separately.

The mix model's test results were:

| Test source | T RMSE | S RMSE |
|---|---:|---:|
| Argo, \(f=+1\) | 0.539 | 0.087 |
| Model-made, \(f=-1\) | 0.547 | 0.089 |
| 50/50 combined | 0.543 | 0.088 |

The mix model is slightly worse than the real-Argo pair model, but it no longer collapses when the source kind changes. This is the cleanest answer to the mixed-input question.

## Metrics and how to read them

Use these measures together.

**Temperature and salinity RMSE.**

\[
{\rm RMSE}(v)=
\sqrt{\frac{1}{N}\sum_{n=1}^{N}(v_n-\hat v_n)^2}.
\]

Compute it on the decoded native-depth profiles. Report T and S separately. Salinity is not interchangeable with temperature because their scales and scientific uses differ.

**CRPS.** CRPS evaluates both the predicted center and spread. It is the training objective for the stochastic EOF cells. Report it by variable and, when possible, by depth band.

**ENCE.** ENCE checks whether predicted spread matches observed error. Use raw test ENCE as the headline. A scale fitted on validation may be reported as a secondary diagnostic, but it must not replace the raw result.

**Persistence.** Copy the source profile to the target and score it. This is the minimum baseline for a profile-input experiment.

**Source-swap tests.** Train on one source type and test on the other. A large swap gap means the network learned a source-specific numerical format rather than the intended physical relationship.

**Distance and time coverage.** Report counts and histograms by distance and lag. A single mean hides the nearest-neighbor pileup. Use fixed bins and report empty-target counts.

**Chronology and leakage checks.** Assert \(t_j<t_i\) for every training source. Assert that the target index, not the pair-row index, selects the label. Keep the PCA basis and cache paired with every checkpoint.

## Reproduction

Use the `nespreso` environment. Keep CPU scope capped at eight threads. The key commands are:

```bash
cd NeSPReSO2_onTemplate
python3 scripts/pair_mix_census.py
python3 selfcheck.py test_pair_table_causal test_pair_dataset_index_is_target test_pair_mix_and_stratified
```

The completed mix run used:

```bash
scripts/tmux_stoch_eof_pair_mix.sh
```

Its configuration is `config/argo/config_argo_stoch_eof_pair_mix.json`. Its three source-head scores are in `reports/eval_pair_mix_heads.json`. The full census is in `reports/pair_mix_census.json`.

Do not compare raw `eval_run.py` results from this Argo cache with the 2024–2025 `profiles_xb.nc` results. They use different eras, grids, and truth records.

## What "nearby" means

A neighbor \(j\) of target \(i\) is allowed only if

\[
3 \le t_i - t_j \le 45\ \text{days}, \qquad \mathrm{haversine}(i,j) \le 250\ \text{km}, \qquad t_j < t_i.
\]

No future sources. Score used only by the old picker,

\[
s_{ij} = \left(\frac{\mathrm{km}}{50}\right)^2 + \left(\frac{\Delta t}{10}\right)^2.
\]

Inputs to a pair net are today's 9 map numbers at \(i\), 64 profile scores of \(j\), \(\Delta\)lat, \(\Delta\)lon, \(\Delta t/30\), and for the mix net a flag \(f \in \{+1,-1\}\).

---

## How we chose the mix and the extra pairs

Census of the 2901 train targets, script `scripts/pair_mix_census.py`, file `reports/pair_mix_census.json`.

| | Closest-4 (old train) | All legal neighbors |
|--|----------------------:|--------------------:|
| Mean km | 57 | 141 |
| Mean lag, days | 12 | 24 |
| Std of Δlat, deg | 0.44 | 0.94 |
| Train pairs | 11349 | 109977 pool |

Δlat and Δlon in the full pool are already near mean 0 with small skew. A Gaussian target on those three numbers would pull mass back toward zero, which is the opposite of the hole we have. The hole is km and lag. Closest-4 puts 3582 of 11349 pairs in the 0–50 km and 3–10 day bin. The pool has more mass at 175–250 km and 20–45 days.

Choice that we trained.

- Train. Round-robin across 4 km bins × 3 lag bins, \(k=8\) neighbors per target. About 87% of train targets have at least 8 legal neighbors. Then each \((i,j)\) is written twice, \(f=+1\) with Argo scores at \(j\), \(f=-1\) with frozen sat-model scores at \(j\). Labels stay Argo at \(i\). That is about 4× the old pair train set.
- Val and test geography. Still the single nearest legal neighbor, so the 623 test pairs match the old table. Mix only duplicates the source kind. `eval_run` on the mix config therefore sees 1246 rows. Compare skill on the Argo-flag slice, n = 623.

Metrics. T/S RMSE on native z. ENCE(T) raw. Persistence RMSE of copying \(j\). Three test heads for mix. Argo flag, synth flag, 50/50 pool.

---

## Results, chrono test

| Cell | T RMSE | S RMSE | ENCE(T) raw | n |
|------|-------:|-------:|------------:|--:|
| Maps only, `stoch_eof` | 0.545 | 0.089 | **0.151** | 623 |
| Pair, train Argo, test Argo | **0.531** | **0.085** | 0.279 | 623 |
| Pair, train Argo, test fake | 0.536 | 0.088 | — | 623 |
| Pair, train fake, test fake | 0.554 | 0.090 | 0.270 | 623 |
| Pair, train fake, test Argo | 0.866 | 0.159 | — | 623 |
| Pair live, unfrozen encoder | 0.587 | 0.096 | 0.418 | 623 |
| Mix, test Argo \(f=+1\) | 0.539 | 0.087 | — | 623 |
| Mix, test fake \(f=-1\) | 0.547 | 0.089 | — | 623 |
| Mix, 50/50 pool | 0.543 | 0.088 | 0.265 | 1246 |
| Copy the earlier Argo | 0.685 mean | 0.105 | — | 623 |
| Extra maps, `A_CRPS_z32` ops | 0.535 | 0.091 | 0.671 | 623 |

---

## Top 3, what they won, and how they work

Ranked by test temperature RMSE among cells we actually trained and scored on this split.

### 1. Pair trained on Argo. Won temperature and salinity error.

T 0.531 °C, S 0.085. That is −0.014 °C vs maps only. ENCE(T) 0.279, fails the 0.20 bar.

**Architecture.** Same `PatchConvMLP` as maps only. Point mode, d_model 128, two 1024-wide ReLU layers, dropout 0.2, 64-D mean and 64-D spread heads. Input width 76. Six calendar/location harmonics go through `enc_proj`. The other 70 numbers, three maps plus 64 neighbor scores plus three deltas, go through `sat_proj`.

**Data.** Chrono split on the target date. Train uses up to 4 nearest legal earlier Argo casts per target, 11349 pairs. Val and test use 1. Neighbor scores are the cache PCA of the real float at \(j\). Maps are at target time \(i\).

**Loss.** Stochastic EOF CRPS on decoded T(z) and S(z) vs raw Argo, plus a small whitened PC term. Two-stage. Stage 1 fits the mean with spread ignored, stop on val profile RMSE. Stage 2 fits spread, stop on val ENCE. A later val-only stretch of σ hurt test ENCE on this cell. Headline is unstretched.

### 2. Extra maps, `A_CRPS_z32` ops. Won among map-only recipes that add derived fields.

T 0.535 °C, S 0.091. ENCE(T) 0.671, badly overconfident.

**Architecture.** Same MLP. Input width 30. Eleven numbers in `enc_proj` (harmonics plus SST, SSS, SSH, ONI, RONI). Nineteen cube operators in `sat_proj`.

**Data.** Same 4145-cast cache family, ops/heave pickle, chrono split, no neighbor profile.

**Loss.** Physical CRPS in z after 32+32 PCA decode, equal T/S and band means. Same two-stage protocol. SSH geostrophy is the operator that actually moves T. Space harmonics are unused. Input ablation is in `reports/eval_input_ablation_z32_ops.json`.

### 3. Mix pair, Argo flag at test. Won "does not explode on fakes" while still beating maps only on real neighbors.

T 0.539 °C, S 0.087 on real neighbors. T 0.547 °C on fakes. Pool 0.543 °C. ENCE(T) 0.265 on the 1246-row mix test.

**Architecture.** Same MLP. Input width 77. Last channel is \(f=+1\) Argo source, \(f=-1\) frozen `stoch_eof` source.

**Data.** Train. Stratified \(k=8\) then 50/50 source mix. Val and test. Same nearest-1 pairs as cell 1, each duplicated with both flags. Frozen synth PCs live in `../data/cache/synth_pcs_stoch_eof_s42.npy`, L2 about 0.87× Argo PCs.

**Loss.** Identical stochastic EOF CRPS and two-stage stops. Val is mixed, so early stop is not the old Argo-only val. Stage 1 stopped at 556, best val profile RMSE 0.249. Stage 2 stopped at 97, best val ENCE 0.116.

Maps only, `stoch_eof`, is fourth on T RMSE and first on ENCE(T) 0.151. If the product is a profile plus honest error bars and there is no extra float, that cell still wins.

---

## Recommendations

**Use.** Maps-only `stoch_eof` when you have only SST, SSS, SSH. Pair-Argo when a real earlier float exists inside the gate. Mix if operations will feed satellite-made neighbors and you need the net not to blow up.

**Do not use.** Synth-only train. Live unfrozen encoder+pair. Val-α stretch as a headline. Reverse-time as the trained product.

**Next, if any.** Mix already answers the 50/50+flag question. A smaller follow-up is whether dropping stratified \(k=8\) back to nearest-4, keeping only the flag mix, keeps 0.539. That would tell you if the extra far pairs helped. I would not train another unfrozen encoder.

---

## Files

Census. `reports/pair_mix_census.json`, `NeSPReSO2_onTemplate/scripts/pair_mix_census.py`  
Mix eval. `reports/eval_pair_mix_heads.json`, `eval_stoch_eof_pair_mix_s42.json`, `eval_stoch_eof_pair_mix_cal.json`  
Earlier pair. `reports/eval_stoch_eof_pair_synth_live.md`  
Trail. `.audit/pair-mix.tsv`

Checkpoints under `NeSPReSO2_onTemplate/saved/stoch_eof_pair*/`. Conda env `nespreso`. No commit in this session.
