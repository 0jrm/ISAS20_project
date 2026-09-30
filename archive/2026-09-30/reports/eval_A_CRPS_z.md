Historical. Not instructions.

# Depth-by-depth models vs the two existing recipes

Test set: 623 Gulf of Mexico Argo profiles, time-ordered split, seed 42.

Two new models predict temperature and salinity at every depth instead of through a compressed summary. Both finished on 27 Aug 2026.

## What we compared

| Model | Inputs | How the profile is built |
|-------|--------|--------------------------|
| Usual three maps, compressed | SST, SSS, SSH | Short summary, then rebuild T/S |
| Usual three maps, 32-mode summary, depth-space training | SST, SSS, SSH | Richer summary; trained to match real T/S with depth |
| Usual three maps, every depth | SST, SSS, SSH | Direct T/S at each depth; no summary |
| Extra maps + Pacific climate index, every depth | SST, SSS, SSH, 19 derived map features, RONI | Same, with more inputs |

## Test-set scores

| Model | Temperature error (°C) | Salinity error | Casts with unstable layering | Share of depths that are unstable | How jagged the *average* temperature profile is |
|-------|-----------------------:|---------------:|-----------------------------:|----------------------------------:|------------------------------------------------:|
| Usual three maps, compressed | 0.562 | 0.091 | 39% | 0.29% | 0.00046 |
| Usual three maps, 32-mode, depth-space | 0.560 | 0.095 | 77% | 0.72% | 0.0010 |
| Usual three maps, every depth | **0.551** | **0.392** | **100%** | **34%** | **0.0082** |
| Extra maps + RONI, every depth | 0.563 | 0.125 | **100%** | **30%** | **0.0026** |

Unstable layering means density increases toward the surface somewhere on that cast (water that would want to overturn). The real Argo mean profile is stable. Both compressed recipes keep a stable *average* profile; both every-depth models do not — even the average of 623 casts is unstable.

## Reading

Temperature typical error is about the same (the every-depth three-map model is 0.01 °C better). Salinity is the failure: 0.39 vs 0.09 for the usual compressed recipe. Extra map features and the Pacific climate index pull salinity back to 0.13, still worse than 0.09.

The smoothness gap is larger than the temperature gap. The compressed recipe is smooth by construction. Training at every depth, with no summary that has to look like real profiles, produces noisy T/S. About one in three depth points is unstably layered, versus well under 1% for the existing models. Extra inputs reduce the jaggedness of the average temperature profile (0.0026 vs 0.0082) but every cast is still unstably layered.

## How well the stated uncertainty matches the errors

After a validation-only width adjustment (same recipe as the 32-mode model):

| Model | Temperature uncertainty mismatch, raw | After width adjustment | Passes the 0.20 bar after adjustment? |
|-------|--------------------------------------:|-----------------------:|---------------------------------------|
| Usual three maps, 32-mode | 0.164 | 0.135 | yes (pooled; surface and 50–200 m still miss) |
| Usual three maps, every depth | 0.291 | 0.091 | yes (pooled) |
| Extra maps + RONI, every depth | 0.521 | 0.178 | yes (pooled; 0–50 m still misses) |

Width adjustment can make the *error bars* look honest. It does not fix salinity or the unstable profiles.

## Files

- Three-map every-depth: `eval_A_CRPS_z_s42.json`, `eval_A_CRPS_z_cal.json`
- Extra maps + RONI: `eval_A_CRPS_z_roni_ops_s42.json`, `eval_A_CRPS_z_roni_ops_cal.json`
- Smoothness numbers: `eval_A_CRPS_z_n2.json`

Checkpoints: `saved/acrps_z/.../acrps_z_s42_s2/model_best.pth` and `saved/acrps_z_roni_ops/.../acrps_z_roni_ops_s42_s2/model_best.pth`.
