# PC-routing ablation

Chronological Argo test, n=623. Short ids (`pc_skip`, `ops`) are for scripts.
The names below are what the row actually is.

Frozen sat-only top 3: A_CRPS_z32+ops, A_CRPS_z32+RONI, stochastic EOF.
Pair MLP and persistence are reference rows. They are not mixed into sat-only training.

Win vs A_CRPS_z32+ops: T RMSE ≤ 0.525, or raw ENCE(T) ≤ 0.20 with T no worse than 0.545,
or a clear 50–200 m T drop.

T and S are physical RMSE (°C, psu). ENCE(T) is raw test calibration.
50–200 m is T RMSE for linear probes and ENCE(T) in that band for trained cells.

## Frozen comparators (not retrained)

| model | T RMSE | S RMSE | ENCE(T) | 50–200 m | n |
|-------|-------:|-------:|--------:|---------:|--:|
| A_CRPS_z32 + 19 cube operators (frozen sat-skill leader) | 0.535 | 0.091 | 0.671 | — | 623 |
| A_CRPS_z32 + ONI/RONI, no operators (frozen) | 0.538 | 0.091 | 0.461 | — | 623 |
| Stochastic EOF, 9-d SSS/SST/SSH (frozen ENCE bar) | 0.545 | 0.089 | 0.151 | — | 623 |
| Pair MLP: satellites + earlier Argo PCs (frozen; needs a neighbor) | 0.531 | 0.085 | 0.279 | — | 623 |
| Persistence: copy earlier Argo profile, no network (frozen) | 0.685 | 0.105 | — | — | 623 |

## Linear probes (no neural net)

| model | T RMSE | S RMSE | ENCE(T) | 50–200 m | n |
|-------|-------:|-------:|--------:|---------:|--:|
| Climatology: decode all 64 PCs as 0 | 1.736 | 0.216 | — | 3.519 | 623 |
| Linear GEM: ridge SSH/SST/SSS/season/lon → T PC1–4 and S PC1–2; other PCs climatology | 0.585 | 0.098 | — | 1.366 | 623 |
| Linear ridge: same 6 inputs → all 64 T/S PCs | 0.584 | 0.097 | — | 1.368 | 623 |
| Linear ridge on 64 PCs, then zero T/S PC 17–32 | 0.584 | 0.097 | — | 1.368 | 623 |

## Trained cells

| model | T RMSE | S RMSE | ENCE(T) | 50–200 m | n |
|-------|-------:|-------:|--------:|---------:|--:|
| Skip-easy MLP: frozen linear GEM on T PC1–4 / S PC1–2, network residual on the rest | 0.575 | 0.095 | 0.520 | 0.769 | 623 |
| Aux stem: 19 operators added only onto hard PCs (not T1–2 / S1–2) | 0.534 | 0.091 | 0.371 | 0.527 | 623 |
| PC CRPS reweighted by unexplained variance (1−r²) | 0.538 | 0.089 | 0.454 | 0.525 | 623 |

## Diagnostics, not bake-off

| model | T RMSE | S RMSE | ENCE(T) | 50–200 m | n |
|-------|-------:|-------:|--------:|---------:|--:|
| Skip-easy MLP, 2+2 epoch smoke (not a bake-off) | 0.586 | 0.099 | — | — | 623 |

## Train-only linear structure

SSH vs T_PC1 r = 0.971 (train).
Easy PCs [0, 1, 2, 3, 32, 33] from ['timecos', 'timesin', 'loncos', 'sss', 'sst', 'ssh'].
max |r| T_PC1=0.971 T_PC5=0.299 T_PC16=0.070 T_PC32=0.063.

## Kill gates

Cell0 T=0.585 vs ops 0.535. Skip still worth training. Do not put aux on PC1. Trunc16 ΔT=0.0000 vs full-ridge. High PCs add almost nothing linearly. Do not train a new 16-PC sat net. Pair keeps 32.
