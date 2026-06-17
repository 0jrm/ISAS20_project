# Port NeSPReSO (v2) into ISAS20_project — ponytail mode

**Plan only** (this document). Implementation follows after approval. Governing style:
**ponytail (lazy senior dev)** — reuse stdlib/template/existing deps first, fewest files,
deletion over addition, no unrequested abstractions, mark shortcuts with `ponytail:`
comments, and leave exactly **one runnable check** behind for non-trivial logic. Still
not lazy about: trust-boundary validation, numerical reproducibility, no `/unity`
hardcoding in code (paths live in config JSON only).

## Source and target

- Source of truth: `v2-nespreso` — `PredictionModel`, `RhoMLP`/`DensityConstraint`,
  `CombinedPCALoss`, PCA helpers, determinism.
- Target (this repo): `NeSPReSO2_onTemplate` — victoresque template with placeholder
  MNIST model/loss, an existing `preproc/preproc_isas_sat.py`, and gridded-satellite
  HDF5 under `data/`.

## Port phases (Phases 0–7) — **DONE**

See git history / `SOURCES.md`. `selfcheck.py` pins v2 forward/loss equivalence.

---

## Dual-dataset fair comparison (ARGO+COAPS vs ISAS20+newsat) — **IMPLEMENTED**

### YAGNI gate

- **Build?** Yes — user wants two trained models on two data regimes with identical
  hyperparams; not achievable by reusing the compare notebook alone.
- **Not building:** ISAS ingest inside v2 `TemperatureSalinityDataset`, cross-dataset
  profile alignment, unified experiment framework, new deps.

### Goal

Train twice on the **same** `PredictionModel` + `CombinedPCALoss` + split seed, differing
only in data source. Evaluate each checkpoint on **its own** cache PCA + raw profiles.

### Data paths

| Tag | Source | Cache builder |
|-----|--------|---------------|
| `isas20` | HDF5 `NeSPReSO_v2_GoM_sat` | `build_train_cache()` |
| `argo_v2` | v2 `config_dataset_full.pkl` | `export_v2_cache.build_argo_cache()` |

`io.dataset_tag` is in `config_hash` → separate `train_ready_<hash>.pkl` files.

ponytail: ~5 station overlap ARGO↔ISAS — comparing **regimes**, not matched profiles.

### Cache schema (extended)

Same pickle as before, plus:

- `profiles`: `{temperature, salinity}` depth-major `(n_z, N)` for eval
- `dataset_tag`: `isas20` | `argo_v2`
- `JULD`: time coordinate (ISAS `JULD` or v2 `TIME`)

### Shared training config (`config_isas.json` / `config_argo.json`)

Matches v2 `default.yaml`: `seed=42`, `batch=512`, `epochs=8000`, `early_stop=500`,
`70/15/15` split via `torch.random_split(..., generator=manual_seed(42))`.

Density penalty **off** on both (v2 default is on — document as intentional simplification
unless user enables paths in both configs).

### Files (minimal)

| File | Role |
|------|------|
| `preproc/export_v2_cache.py` | ARGO pickle → cache |
| `preproc/preproc_isas_sat.py` | +`profiles`, `dataset_tag`, `write_train_cache` |
| `data_loader/data_loaders.py` | aligned split; no `BaseDataLoader` for NeSPReSO |
| `config_isas.json`, `config_argo.json` | twin configs |
| `train.py` | `ensure_cache` dispatches on `dataset_tag` |
| `eval_run.py` | test-split raw profile RMSE + loss |
| `selfcheck.py` | +split seed pin, cache schema if cache exists |

### Commands

```bash
cd NeSPReSO2_onTemplate

# Build caches (paths in config JSON)
srun --ntasks=1 --cpus-per-task=8 python3 -c "
from playground import read_json
from preproc.preproc_isas_sat import build_train_cache
from preproc.export_v2_cache import build_argo_cache
build_train_cache(read_json('config_isas.json'), force=True)
build_argo_cache(read_json('config_argo.json'), force=True)
"

# Train both (same hyperparams, different data)
srun --ntasks=1 --cpus-per-task=8 python3 train.py -c config_isas.json
srun --ntasks=1 --cpus-per-task=8 python3 train.py -c config_argo.json

# Eval test split (raw profile RMSE)
srun --ntasks=1 --cpus-per-task=8 python3 eval_run.py -c config_isas.json -r saved/.../checkpoint.pth
srun --ntasks=1 --cpus-per-task=8 python3 eval_run.py -c config_argo.json -r saved/.../checkpoint.pth
```

### Eval rule (trust boundary)

Never mix checkpoint PCA across caches. `eval_run.py` uses checkpoint PCA + same-cache
`profiles` as ground truth.

### Needs human review

- Loss scales still GoM temp/sal defaults.
- Vertical grids differ (187 ISAS levels vs 1801 m ARGO) — raw RMSE not comparable across
  tags; compare within-tag only, or add ISAS regridding later.
- `config_argo.json` `v2_pickle` / `v2_src` are machine paths — edit per host.

## Ponytail deltas (port — historical)

- **No parallel typed-config/dataclass layer.** The template already has `config.json`
  + `parse_config.ConfigParser`. Extend the JSON + one tiny `validate()`.
- **No new caching subsystem.** `preproc/preproc_isas_sat.py` already reads HDF5 once
  and pickles. Extend it to emit the train-ready pickle; reuse it as the cache.
- **No new determinism module.** `train.py` already seeds; reuse it + a `prepare_device`
  helper.
- **Fewest files.** PCA reconstruction helpers go inside `model/loss.py` (its only
  consumer), not a separate `pca.py`.
- **Density penalty defaults OFF** in the first port (pulls an external `rhoMLP_*.pt`
  and adds a sizeable code path). Wired but opt-in; flagged.

---

## Agent train monitor (planned)

See **[PLAN-agent-train-monitor.md](PLAN-agent-train-monitor.md)** — autonomous dual-run
launch, JSON `status.json` telemetry, shell sentinels, and post-train eval for the agent.
