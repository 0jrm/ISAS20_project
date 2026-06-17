# NeSPReSO v2 port — source mapping

Source of truth: `/unity/g2/jmiranda/v2-nespreso` (branch `main`).

| v2 module | Target (this repo) | Notes |
|-----------|-------------------|-------|
| `src/nespreso/models/mlp.py` `PredictionModel` | `NeSPReSO2_onTemplate/model/model.py` | Replaces buggy `FFNN`; config-driven dims |
| `src/nespreso/losses.py` | `NeSPReSO2_onTemplate/model/loss.py` | PCA helpers inlined; N-output `outputs` map |
| `src/nespreso/data/pca.py` | `NeSPReSO2_onTemplate/model/loss.py` | Inlined (only consumer) |
| `src/nespreso/models/density.py` | `NeSPReSO2_onTemplate/model/density.py` | Verbatim; opt-in via `density.enabled` |
| `src/nespreso/data/features.py` `prepare_inputs` | `NeSPReSO2_onTemplate/preproc/preproc_isas_sat.py` | Vectorized; offset=0 center-pixel path |
| `src/nespreso/data/dataset.py` PCA fit / weights | `NeSPReSO2_onTemplate/preproc/preproc_isas_sat.py` | Pickle is the cache |
| `src/nespreso/determinism.py` | `NeSPReSO2_onTemplate/train.py` + `playground/util.py` | Reuse template seed + `prepare_device` |
| `src/nespreso/runner.py` criterion build | `NeSPReSO2_onTemplate/train.py` | `CombinedPCALoss` built after dataset load |
| `src/nespreso/inference.py` inverse | `NeSPReSO2_onTemplate/model/metric.py` | Per-variable offsets from `outputs` |

Not ported (ponytail YAGNI): typed `AppConfig`, separate caching subsystem, VAE/DIRESA encoders, determinism module, separate `pca.py`.
