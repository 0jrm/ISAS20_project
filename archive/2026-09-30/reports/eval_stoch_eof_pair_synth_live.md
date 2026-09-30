Historical. Not instructions.

# Pair sources: real Argo vs frozen synthetic vs live encoder

Test: chronological 70/15/15, n=623, cache `train_ready_4ee013852d33.pkl` (32+32 PCA). Gate 3–45 days, ≤250 km, K=4 train / 1 val–test. Source is **strictly earlier** than the target. Labels are always raw Argo at the target. Headline ENCE is **raw** (val-α picker not used unless noted). Finished 2 Sep 2026.

Do not mix these numbers with the 2024–2025 HYCOM/xb table.

## What we compared

| Cell | Neighbor the net sees | Encoder |
|------|------------------------|---------|
| Sat-only `stoch_eof` | none (9-d maps only) | trained once, frozen |
| Pair + Argo | 64 PCs of a real earlier float | not in the graph |
| Pair + frozen synth | 64 PCs from frozen `stoch_eof` at the neighbor | frozen; dumped to `synth_pcs_stoch_eof_s42.npy` |
| Pair live | live 9-d encoder at the neighbor, then pair head | unfrozen; `L_pair + L_direct` on the same target Argo |

Same pair table for all pair cells. Stage-1 stop `val_profile_rmse`; stage-2 `val_ence`.

## Test-set scores

| Cell | T RMSE | S RMSE | ENCE(T) raw | vs sat-only T |
|------|-------:|-------:|------------:|---------------|
| Sat-only | 0.545 | 0.089 | **0.151** | — |
| Pair + Argo (trained on Argo) | **0.531** | **0.085** | 0.279 | −0.014 |
| Pair + Argo, swap in synth (no retrain) | 0.536 | 0.088 | — | −0.009 |
| Pair trained on synth, test synth | 0.554 | 0.090 | 0.270 | **+0.010** |
| Pair trained on synth, test Argo | 0.866 | 0.159 | — | +0.321 |
| Pair live, pair product | 0.587 | 0.096 | 0.418 | +0.042 |
| Pair live, 9-d encoder path | 0.576 | 0.099 | — | +0.031 |
| Copy the earlier Argo (persistence) | 0.685 mean / 0.456 median | 0.105 | — | worse |

Reverse-time probes (later neighbor, not a trained product):

| Probe | T RMSE |
|-------|-------:|
| Argo-trained pair + later Argo | 0.533 |
| Argo-trained pair + later synth | 0.537 |
| Synth-trained pair + later synth | 0.551 |
| Synth-trained pair + later Argo | 0.849 |

## Training

| Cell | Stage-1 stop | Best val profile RMSE | Stage-2 stop |
|------|-------------:|----------------------:|-------------:|
| Pair + Argo | 557 | ~0.28 (train ~0.16) | 110 |
| Pair + frozen synth | 568 | 0.253 (epoch ~67) | 111 |
| Pair live | 556 | 0.251 (epoch ~55) | 101 |

Val-α picker: Argo-pair **hurt** test ENCE (0.279 → 0.357). Synth and live pickers chose `none` (no stretch).

## Reading

The only skill win is **a real earlier float**. Training on frozen synthetics does **not** teach the pair net to use them: test T 0.554 is worse than ignoring the neighbor (0.545). Feeding that same net a real Argo source at test (0.866) shows it did not learn a source-agnostic “profile + Δ” map — it overfit the synthetic PC scale.

The unfrozen joint cell is worse still. The pair product (0.587) and the 9-d path (0.576) both lose to the frozen sat-only start (0.545). Letting profile loss act on the encoder **and** the pair head did not keep the encoder honest; it moved it off the sat-only solution.

Calibration: sat-only remains the only cell near the 0.20 ENCE(T) bar. Pair cells pay in stated-uncertainty honesty for any neighbor skill they get (and the two new cells get none).

## In plain language

We asked: if there is already a nearby float from a few weeks ago, can the model use it? And if we do not have a real float there, can we fake one from satellites and still get the benefit?

**A real nearby float helps a little.** Temperature error drops from 0.545 °C to 0.531 °C. That is a small but clean gain. Just copying the old float is worse (about 0.69 °C), so the model is not merely pasting; it is combining the old profile with today’s maps.

**A fake nearby profile does not help**, whether we freeze the fake-maker or keep training it.

- If we *train* the combiner on real floats, then at test time swap in a satellite-made fake, we keep most of the gain (0.536 °C). That was a lucky test: the combiner still expects “real-float-shaped” numbers.
- If we *train* the combiner on fakes, it gets **worse** than using today’s maps alone (0.554 °C). Show it a real float after that and it falls apart (0.866 °C). It learned the look of the fakes, not “use whatever profile is next door.”
- If we let the fake-maker keep changing while we train the combiner, both pieces get worse (0.587 °C for the combo, 0.576 °C for the map-only path that used to be 0.545 °C). They pull on each other and neither job is done well.

**Uncertainty statements get less honest** whenever we add a neighbor, even in the one cell that improves the typical error.

**Bottom line:** keep the sat-only model as the map-only product. The neighbor trick is worth keeping only when the neighbor is a **real earlier Argo**. Do not train on satellite-made stand-ins, and do not unfreeze the map model into the same loss as the combiner.

## Files

- Argo-source pair: `eval_stoch_eof_pair_s42.json`, `eval_stoch_eof_pair_cal.json`, `eval_stoch_eof_pair_persist.json`
- Frozen-synth train: `eval_stoch_eof_pair_synth_s42.json`, `eval_stoch_eof_pair_synth_cal.json`, `eval_pair_synth_trained_swap.json`
- Swap on the *Argo-trained* pair net (no retrain): `eval_pair_synth_swap.json`
- Live joint: `eval_stoch_eof_pair_live_s42.json`, `eval_stoch_eof_pair_live_cal.json`, `eval_stoch_eof_pair_live_direct.json`
- Sat-only: `eval_stoch_eof_s42.json`, `eval_stoch_eof_cal.json`

Checkpoints: `saved/stoch_eof_pair/.../stoch_eof_pair_s42_s2/model_best.pth`, `saved/stoch_eof_pair_synth/.../stoch_eof_pair_synth_s42_s2/model_best.pth`, `saved/stoch_eof_pair_live/.../stoch_eof_pair_live_s42_s2/model_best.pth`.
