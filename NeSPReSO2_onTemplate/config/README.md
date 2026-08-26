# Training configs

All configs are run from `NeSPReSO2_onTemplate/`:

```bash
python3 train.py -c config/argo/config_argo.json
```

Paths in JSON (`data_path`, `cache_dir`, etc.) are relative to the template root unless absolute.

## ARGO (`argo/`)

| File | Purpose | Input dim | Cache tag | Status |
|------|---------|----------:|-----------|--------|
| `config_argo.json` | Point L4 scalars, 16+16 PCA, chronological (matrix **A**) | 9 | `argo_v2` | production |
| `config_argo_A_CRPS_z32.json` | **A_CRPS_z32** (A×CRPS-z): physical CRPS, PCA 32+32, equal T/S, band means, ENCE(T) stop | 9 | `argo_v2` | research |
| `config_argo_A_CRPS_z32_roni.json` | A_CRPS_z32 + CPC ONI/RONI splice (same 32-PC cache) | 11 | `argo_v2` | research |
| `config_argo_joint_eof.json` | Joint T/S EOF-32 (matrix **B**) | 9 | `argo_v2` | production |
| `config_argo_densityspice_lowrank_crps.json` | Low-rank δσ₀ + spice CRPS (matrix **C**) | 9 | `argo_v2` | production |
| `config_argo_chrono_dates.json` | Same as A with explicit date split ranges | 9 | `argo_v2` | ablation |
| `config_argo_smoke.json` | 2-epoch ARGO point smoke | 9 | `argo_v2` | smoke |
| `config_argo_l3_smoke.json` | L3 mask-native patch smoke | 46881 | `argo_v2` | smoke |
| `config_argo_l3_l4_smoke.json` | L3 + L4 mask-augment smoke | 46881 | `argo_v2` | smoke |
| `config_argo_densityspice*.json` | Phase 3/4 density_spice family | 9 | `argo_v2` | research |

Archived (§5.1 kill list, **before** matrix results): [`archive/`](archive/) — anom / point_cube / residual / patch_l4 / field variants.

## ISAS (`isas/`)

| File | Purpose | Input dim | Cache tag | Status |
|------|---------|----------:|-----------|--------|
| `config_isas.json` | Point MLP, 15+15 PCA | 9 | `isas20` | production |
| `config_isas_patch.json` | Patch conv, 16+16 PCA, auto batch | 306 | `isas20` | production |
| `config_isas_patch_decoder.json` | Patch + decoder profile loss (dim 16 AE) | 306 | `isas20` | ablation |
| `config_isas_patch_decoder_dim32.json` | Patch + dim-32 AE decoder | 306 | `isas20` | ablation |
| `config_isas_patch_decoder_dim32_res.json` | Residual dim-32 decoder | 306 | `isas20` | ablation |
| `config_isas_patch_decoder_dim32_satres.json` | Residual decoder + satellite resolution | 306 | `isas20` | ablation |

## Shared smoke (`smoke/`)

| File | Purpose | Input dim | Cache tag | Status |
|------|---------|----------:|-----------|--------|
| `config_smoke.json` | Template PatchConvMLP smoke (ISAS paths) | 306 | `isas20` | smoke |

## Encoding compare (`compare/`)

Pinned random-split configs for PCA vs AE ablation (6 models). See [`../scripts/run_encoding_compare_train.sh`](../scripts/run_encoding_compare_train.sh) and [`../../PLAN-encoding-compare.md`](../../PLAN-encoding-compare.md).

| File | Purpose | Input dim | Cache tag | Status |
|------|---------|----------:|-----------|--------|
| `isas_point_pca16.json` | ISAS point PCA-16 | 9 | `isas20` | ablation |
| `isas_patch_pca16.json` | ISAS patch PCA-16 | 306 | `isas20` | ablation |
| `isas_point_ae128.json` | ISAS point AE-128 latents | 9 | `isas20` | ablation |
| `isas_patch_ae128.json` | ISAS patch AE-128 latents | 306 | `isas20` | ablation |
| `argo_point_pca15.json` | ARGO point PCA-15 | 9 | `argo_v2` | ablation |
| `argo_point_pca16.json` | ARGO point PCA-16 | 9 | `argo_v2` | ablation |

## Removed

`config.json` at template root was a duplicate of `config/isas/config_isas.json` (victoresque template default) and has been deleted.
