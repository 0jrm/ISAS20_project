# ISAS20 Project

NeSPReSO v2 GoM dissertation branch: ARGO-first subsurface targets, mask-native L3 surface inputs, chronological evaluation.

## Documentation

| Doc | Purpose |
|-----|---------|
| [`HANDOFF.md`](HANDOFF.md) | Session status and next tasks (**start here**) |
| [`PLAN.md`](PLAN.md) | Full dissertation roadmap (Phases 0–10) |
| [`PLAN-dissertation-data-foundation.md`](PLAN-dissertation-data-foundation.md) | Implementation notes, commands, split rationale |
| [`NeSPReSO2_onTemplate/README.md`](NeSPReSO2_onTemplate/README.md) | Training ops |
| [`context.txt`](context.txt) | L3 satellite product IDs and download code |

## Quick start

```bash
cd NeSPReSO2_onTemplate
srun --ntasks=1 --cpus-per-task=8 python3 selfcheck.py
srun --ntasks=1 --cpus-per-task=8 python3 scripts/data_census.py -c config/argo/config_argo.json
srun --ntasks=1 --cpus-per-task=8 --gres=gpu:1 python3 train.py -c config/argo/config_argo_smoke.json
```

Reports land in [`reports/`](reports/).

## Data layout

```
data/
├── raw/          # unchanged downloaded products
├── processed/    # rasterized L3 patches (future)
├── cache/        # train-ready pickles
└── manifests/    # download_manifest.jsonl
```

## Module structure

```
ISAS20_project/
├── NeSPReSO2_onTemplate/   # training code
├── utils/retrieve_sat.py     # legacy L4 satellite retrieval
├── reports/                  # census + split design
└── data/                     # caches and raw downloads
```

## Legacy satellite retrieval

See [`utils/retrieve_sat.py`](utils/retrieve_sat.py) for L4 gridded product interpolation. New dissertation work uses L3 observational products via [`NeSPReSO2_onTemplate/scripts/download_l3_products.py`](NeSPReSO2_onTemplate/scripts/download_l3_products.py).
