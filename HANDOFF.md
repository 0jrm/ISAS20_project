# Session handoff — dissertation data foundation

**Branch focus:** GoM NeSPReSO dissertation branch — ARGO-first targets, mask-native L3 inputs, chronological splits.

**Read first:** [`PLAN.md`](PLAN.md) (full roadmap), [`PLAN-dissertation-data-foundation.md`](PLAN-dissertation-data-foundation.md) (what changed + how to run).

## Status (Jun 2026)

| Phase | Status | Notes |
|-------|--------|-------|
| 0 Data census | **Done** | `scripts/data_census.py` → `reports/data_census.*`, `reports/split_design.*` |
| 1 Chronological split | **Done** | `base/split_utils.py`; `split_mode: chronological` in ARGO configs |
| 2 ARGO-first path | **Done** | Existing v2 export + eval; smoke config updated |
| 3 L3 pipeline | **Done** | Rasterization, L3 train cache, batch loading, PatchConvMLP forward (SSH + ERA5 wind MVP) |
| 4 L4 augmentation | **Scaffolded** | `preproc/l4_augment.py`; config `io.l4` (disabled by default) |
| 5 Model L3 channels | **Done** | `preproc/l3_input.py`; `train.py` L3 cache path; 15-ch patch mode |
| 6–9 | **Pending** | Stratified eval, diagnostics, physics/ensemble hooks |

## Key finding

GoM ARGO export spans **2015–2022 only** (4145 profiles), not 2002–2020. Candidate A split (2002–2015 train) is **empty**. Default dissertation split: **chronological 70/15/15** (`split_mode: chronological` without `split_config`). Alternative: explicit dates in `config_argo_chrono_dates.json`.

## Commands

```bash
cd NeSPReSO2_onTemplate

# Self-check (includes chronological split test)
srun --ntasks=1 --cpus-per-task=8 python3 selfcheck.py

# Data census + split design reports
srun --ntasks=1 --cpus-per-task=8 python3 scripts/data_census.py -c config_argo.json

# ARGO smoke (2 epochs, chronological split)
srun --ntasks=1 --cpus-per-task=8 --gres=gpu:1 python3 train.py -c config_argo_smoke.json

# L3 download (requires credentials)
python3 scripts/download_l3_products.py --product all_scaffold
python3 scripts/download_l3_products.py --product ssh_l3_historical --date 2020-01-15
python3 scripts/download_l3_products.py --product era5_wind --year 2020 --month 1

# L3 train cache + batch smoke
srun --ntasks=1 --cpus-per-task=8 python3 scripts/build_l3_samples.py -c config_argo_l3_smoke.json --max-samples 20 --export-train-cache --force

# L3 smoke train (2 epochs, mask-native patches)
srun --ntasks=1 --cpus-per-task=8 --gres=gpu:1 python3 train.py -c config_argo_l3_smoke.json
```

## Eval rules (unchanged)

1. Pair checkpoint with its cache.
2. Cross-tag: `eval_matched.py` only.
3. Dissertation results: use **chronological** split, not random.
4. L4 is augmentation/baseline — ARGO/CORA remains primary target.

## Next coding tasks

1. Stratified eval reports by L3 coverage regime (Phase 6).
2. Readiness diagnostics module (`diagnostics/readiness.py`).
3. Full L4 masked-augmentation pipeline (apply real masks to L4 fields).

## Known limitations

- Current ARGO cache still uses **L4 gridded** SST/SSH/SSS (v2 COAPS path).
- L3 downloaders + rasterization scaffolded; processed L3 samples via `build_l3_samples.py` (SSH + ERA5 wind MVP).
- ISAS configs still use random split (legacy parity); switch when ISAS temporal census is done.
