# HeaveFast ablations vs A×CRPS and HeaveFast s42

Chronological test, n=623, native ARGO z. Each ablation paired with its own cache.
Conv is 3×3 at 1°, T=t−2…t0, local SST/SSS/SSH in `n_enc`. Point extras are operators, GEBCO bathy, NBS wind.

## Train

| run | epochs | wall | s/epoch | best val |
|---|---:|---:|---:|---:|
| HeaveFast s42 | 2091 | 130.5 min | 3.745 | — |
| conv3 | 836 | 20.9 min | 1.501 | 3.762 |
| ops | 714 | 18.0 min | 1.514 | 3.678 |
| bathy | 704 | 16.8 min | 1.431 | 3.865 |
| bathy+wind | 698 | 25.0 min | 2.148 | 3.717 |

## Test RMSE (`eval_run` native z)

| run | T RMSE | S RMSE |
|---|---:|---:|
| A_CRPS | 0.562 | **0.091** |
| HeaveFast | 0.550 | 0.092 |
| conv 3×3@1° | 0.571 | 0.093 |
| current+ops | **0.549** | 0.092 |
| current+bathy | 0.584 | 0.096 |
| current+bathy+wind | 0.561 | 0.093 |

## Thermocline

- **A_CRPS**: D26 19.05 m; MLD 36.8 m; T 0–50/50–200/200–800 1.147/1.215/0.658; N² profile 0.39, level 0.0029
- **HeaveFast**: D26 18.46 m; MLD 33.3 m; T 0–50/50–200/200–800 1.122/1.174/0.653; N² profile 0.99, level 0.0082
- **conv 3×3@1°**: D26 18.64 m; MLD 33.2 m; T 0–50/50–200/200–800 1.145/1.214/0.685; N² profile 0.97, level 0.0108
- **current+ops**: D26 18.27 m; MLD 33.4 m; T 0–50/50–200/200–800 1.123/1.184/0.649; N² profile 0.97, level 0.0099
- **current+bathy**: D26 20.80 m; MLD 33.3 m; T 0–50/50–200/200–800 1.118/1.251/0.703; N² profile 0.99, level 0.0097
- **current+bathy+wind**: D26 19.84 m; MLD 32.7 m; T 0–50/50–200/200–800 1.154/1.203/0.663; N² profile 0.99, level 0.0093

## Takeaway

The advisor conv cell is in the game (T 0.571 vs skill-floor context ~0.59) but it loses to point HeaveFast on T, S, D26, and 50–200 m T. Pooling still costs even with local sat in `n_enc`.

**current+ops** is the only ablation that beats HeaveFast on T (0.549 vs 0.550) and D26 (18.27 vs 18.46 m). The T delta is noise-scale. D26 is the clearer move.

Bathy from real GEBCO hurt T and D26. Adding NBS wind recovered some T (0.561) and the best MLD (32.7 m) but not D26.

S still belongs to A×CRPS (0.091). All HeaveFast-family S sit at 0.092–0.096.
