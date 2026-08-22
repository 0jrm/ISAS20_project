# Heave / latent / direct / Fast vs A×CRPS (chrono test, n=623)

Same cache `data/cache/train_ready_3adcff404b0b.pkl`. evalphys 1.2.0.

Heave family: exp MLD/D26, CRPS σ in metres, geom_scale=10. Fast = batched warp, same objective as s42d.

## Training time

| run | epochs | wall | s/epoch | state |
|---|---:|---:|---:|---|
| Heave s42d | 785 | 123.1 min | 9.410 | done/early_stop |
| HeaveFast s42 | 2091 | 130.5 min | 3.745 | done/early_stop |
| Latent | 4684 | 134.8 min | 1.727 | done/early_stop |
| Direct | 2872 | 82.4 min | 1.721 | done/early_stop |

## Test RMSE (eval_run native z)

| run | T RMSE | S RMSE |
|---|---:|---:|
| A_CRPS | 0.562 | **0.091** |
| Heave s42d | 0.577 | 0.098 |
| HeaveFast | 0.550 | 0.092 |
| Latent | 0.546 | 0.142 |
| Direct | **0.541** | 0.333 |

## Thermocline scorecard

- **A_CRPS**: D26 19.05 m; MLD 36.8 m; T 0–50/50–200/200–800 1.147/1.215/0.658; N² profile 0.39, level 0.0029
- **Heave s42d**: D26 19.85 m; MLD 33.4 m; T 1.134/1.205/0.704; N² profile 0.99, level 0.0099
- **HeaveFast**: D26 **18.46** m; MLD **33.3** m; T 1.122/1.174/0.653; N² profile 0.99, level 0.0082; ENCE(σ_D26)=0.52
- **Latent**: D26 19.50 m; MLD 42.1 m; T **1.070/1.170**/0.652; N² profile 1.00, level 0.47
- **Direct**: D26 19.03 m; MLD 48.1 m; T 1.111/1.189/0.617; N² profile 1.00, level 0.43

DA recommendation: [`v2_da_candidate.md`](v2_da_candidate.md).
