# Input permutation importance

Checkpoint `saved/pc_aux/models/NeSPReSO2_ARGO_GoM_pc_aux_pc_aux_s2/pc_aux_s2/model_best.pth`. Chronological test n=623.
Joint row-shuffle of the named columns. ΔT > 0 means the trained net used that group.
Not a retrain leave-one-out.

Baseline T RMSE = 0.532 (0–50 m 1.108). S RMSE = 0.090.

| group | n | ΔT | T | ΔT 0–50 m |
|-------|--:|---:|--:|----------:|
| sat | 3 | +1.721 | 2.253 | +1.504 |
| ssh | 1 | +1.654 | 2.187 | +0.349 |
| sst | 1 | +0.122 | 0.655 | +1.189 |
| roni | 1 | +0.031 | 0.564 | +0.015 |
| time | 2 | +0.024 | 0.556 | +0.076 |
| sss | 1 | +0.013 | 0.545 | +0.043 |
| oni | 1 | +0.007 | 0.540 | -0.006 |
| enso | 2 | +0.002 | 0.534 | +0.003 |
| ops | 19 | +0.001 | 0.533 | +0.000 |
| ops_geo | 4 | +0.001 | 0.533 | -0.001 |
| ops_sst_grad | 4 | +0.000 | 0.533 | -0.000 |
| ops_tendency | 2 | +0.000 | 0.532 | +0.000 |
| ops_sss_grad | 4 | +0.000 | 0.532 | +0.003 |
| lon | 2 | +0.000 | 0.532 | +0.000 |
| ops_ssh_grad | 4 | +0.000 | 0.532 | -0.000 |
| ops_ssh_lap | 1 | +0.000 | 0.532 | +0.000 |
| space | 4 | -0.000 | 0.532 | +0.000 |
| lat | 2 | -0.001 | 0.531 | +0.000 |

## vs `eval_input_ablation_z32_ops.json`

| group | ΔT this | ΔT other | this − other |
|-------|--------:|---------:|-------------:|
| sat | +1.721 | +1.704 | +0.017 |
| ssh | +1.654 | +1.670 | -0.016 |
| sst | +0.122 | +0.091 | +0.032 |
| sss | +0.013 | +0.009 | +0.004 |
| ops | +0.001 | +0.043 | -0.042 |
| ops_geo | +0.001 | +0.022 | -0.021 |
| roni | +0.031 | +0.046 | -0.015 |
| oni | +0.007 | +0.028 | -0.020 |
| enso | +0.002 | +0.001 | +0.000 |
| time | +0.024 | +0.032 | -0.008 |
| space | -0.000 | +0.004 | -0.005 |
| lon | +0.000 | +0.004 | -0.004 |
| lat | -0.001 | -0.000 | -0.001 |
