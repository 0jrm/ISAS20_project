# Run artifacts (`saved/`)

Training outputs land here. The directory is **gitignored** except for small JSON stubs checked in elsewhere; treat this as local disk cache.

## Layout

| Path | Contents |
|------|----------|
| `models/<name>/<run_id>/` | Per-training run: `config.json`, `model_best.pth`, `status.json`, checkpoints |
| `log/<name>/<run_id>/` | Text logs (`info.log`); TensorBoard events when enabled |
| `compare_runs/<key>/` | Encoding-compare trains (see `manifest.json` for canonical checkpoints) |
| `decoders/<tag>/` | Frozen profile autoencoder weights exported for AE-surface models |
| `notebook_runs/` | Short notebook smoke trains (2 epochs) |
| `benchmarks/` | Batch-size and ML-opt sweep JSON |
| `gom_diagnostics/`, `phase6/` | Aggregated eval tables and reports |

## Retention policy

**Keep** for any run you may resume or cite:

- `model_best.pth`
- `config.json` (copy of training config at run time)
- `status.json` (monitor state)

**Safe to delete** once a run is finished and eval is archived:

- `checkpoint-epoch*.pth` (non-best checkpoints)
- Duplicate run folders (same config, superseded timestamp)
- `*.train.log` under `compare_runs/` after manifest is updated
- Old smoke trees under `smoke_*` after validation

When pruning compare runs, update [`compare_runs/manifest.json`](compare_runs/manifest.json) so notebooks point at surviving `model_best.pth` paths.

## Regenerating

```bash
# encoding-compare manifest + trains
bash scripts/run_encoding_compare_train.sh

# compare notebook (source of truth: notebooks/build_notebook.py)
python3 notebooks/build_notebook.py
jupyter notebook notebooks/compare_v2_vs_template.ipynb
```
