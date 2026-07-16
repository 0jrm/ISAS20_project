# Phase 1 decisive tests — T1 basis stability

Cache: `/unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project/data/cache/train_ready_4411c65ee518.pkl`  |  train n=2901  test n=623

| variant | viol_rate_profile | viol_rate_level | mean T/S RMSE | dρ/dz RMSE | MLD RMSE |
|---------|-------------------|-----------------|---------------|------------|----------|
| A_separate_pca | 0.7657 | 0.0091 | 0.0893 | 0.0067 | 36.6068 |
| B_joint_eof | 0.8860 | 0.0092 | 0.0886 | 0.0066 | 38.6180 |
| C_density_spice_pca | 0.7271 | 0.0089 | 0.0854 | 0.0068 | 39.4356 |
| D_monotone_density | 0.3868 | 0.0022 | 0.0696 | 0.0032 | 1.3431 |

## Decision rules
- ESCALATE: B_joint_eof ≈ A — violations may not be basis-induced
- ESCALATE: C_density_spice_pca ≈ A — violations may not be basis-induced
