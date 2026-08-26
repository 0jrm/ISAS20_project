# A_CRPS_z32 ablation (RONI / ops / heave)

Seed 42, chronological test n=623. Baseline pins from `reports/A_CRPS_z32.json`.

A cell wins if any gate holds. S RMSE is reported, not a win. None of these is an ingest product. 0–50 m and 50–200 m ENCE(T) after val-α still miss 0.20 except where noted.

| cell | T RMSE | S RMSE | ENCE(T) raw | ENCE(T) val-α | 50–200 ENCE(T) val-α | gates |
|------|-------:|-------:|------------:|--------------:|---------------------:|-------|
| A_CRPS_z32 | 0.560 | 0.095 | 0.164 | 0.135 | 0.415 | baseline |
| RONI/ONI | **0.538** | 0.091 | 0.461 | 0.180 | **0.411** | T RMSE. 50–200 ENCE barely. Pooled ENCE misses 0.135. |
| ops (19 operators, PatchConvMLP, 32-PC paired) | **0.535** | 0.091 | 0.671 | 0.371 | **0.247** | T RMSE. 50–200 ENCE. Pooled ENCE fails. |
| heave 16+16, physical T/S CRPS, val ENCE stop | 0.570 | 0.099 | 0.353 | 0.257 | **0.381** | 50–200 ENCE only. T RMSE worse than pin. |

Headline heave checkpoint is stage-2 `model_best` after resume (`0825_140921`, `val_ence` 0.099, recipe `global_var`). A tmux eval of an earlier s2 `model_best` printed T RMSE 0.533 before that file went away. Protocol is ENCE stop, so 0.570 is the cell.

RONI 0–50 m ENCE(T) after α is 0.452. Ops 0.469. Heave 0.444. Stock HeaveFast conv3 1° was T RMSE 0.741 on the same split. Heave N² / D26 vs A_CRPS_z32 was not scored here.

## Checkpoints

- RONI. `NeSPReSO2_onTemplate/saved/acrps_z32_roni/models/NeSPReSO2_ARGO_GoM_A_CRPS_z32_roni_acrps_z32_roni_s42_s2/acrps_z32_roni_s42_s2/model_best.pth`. Cache `train_ready_4ee013852d33.pkl`.
- Ops. `.../acrps_z32_ops/.../acrps_z32_ops_s42_s2/model_best.pth`. Cache `train_ready_heave_ops_pca32.pkl` (ops inputs, 32-PC copied from `4ee013852d33`).
- Heave. `.../acrps_z32_heave_s42_s2/0825_140921/model_best.pth`. Cache `train_ready_3adcff404b0b.pkl` for inputs only. Residual PCA is fit in the loss.

JSON. `eval_A_CRPS_z32_{roni,ops,heave}_s42.json` and `*_cal.json`.

Heave stage 2 hit a TensorBoard `bins="auto"` OOM at epoch 459. Resume used fixed 50-bin histograms.

Optional ops+heave cell was allowed (ops and RONI already won gates) and was not run.
