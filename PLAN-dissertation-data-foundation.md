# Dissertation data foundation — implementation notes

Companion to [`PLAN.md`](PLAN.md). Documents what changed in the dissertation branch and how to run it.

## What changed

1. **Chronological splitting** replaces random split as the dissertation default for ARGO configs.
2. **Data census** scripts inspect actual profile coverage before choosing splits.
3. **L3 download scaffolding** for mask-native surface inputs (Copernicus L3 SSH, VIIRS L3U SST, ERA5 wind, optional SMAP/SWOT).
4. **Split design reports** under `reports/` justify temporal holdouts for thesis methods.

Random split (`split_mode: random`) remains for backward compatibility and ablations.

## Why random split is no longer the dissertation default

Random splitting mixes years across train/val/test, which:

- Leaks future observing-system regimes into training.
- Inflates metrics when adjacent-year profiles are correlated.
- Cannot support defensible temporal generalization claims.

Chronological splitting assigns samples by date (explicit ranges or sorted 70/15/15 blocks).

## How the temporal split was selected

Run the census:

```bash
cd NeSPReSO2_onTemplate
srun --ntasks=1 --cpus-per-task=8 python3 scripts/data_census.py -c config_argo.json
```

Inspect `reports/split_design.md`. For the current GoM ARGO export:

| Candidate | Viable? | Notes |
|-----------|---------|-------|
| A (2002–2015 / 2016–17 / 2018–20) | **No** | Data start in 2015 |
| B (chronological 70/15/15) | **Yes** | **Default** |
| C (test = 2020 high-density year) | Yes | High-observation subset |
| D (test = 2015–2018 sparse era) | Yes | Low-observation stress test |
| E (2015–19 / 2020 / 2021) | Yes | Common-overlap explicit dates |

**Recommended default:** Candidate B (`config_argo.json` with `split_mode: chronological`).

**Explicit date alternative:** `config_argo_chrono_dates.json` (Candidate E).

## High/low data-regime subsets

Defined in `reports/split_design.json` → `recommendation.evaluation_subsets`:

- **high_observation:** year 2020 (1515 profiles — peak density).
- **low_observation:** 2015–2018 (691 profiles — sparse early era).
- **common_overlap:** 2015–2021 (exclude sparse 2022 tail).

Use these for stratified evaluation once L3 coverage metrics exist.

## How to run

| Task | Command |
|------|---------|
| Data census | `python3 scripts/data_census.py -c config_argo.json` |
| ARGO smoke train | `python3 train.py -c config_argo_smoke.json` |
| ARGO baseline train | `python3 train.py -c config_argo.json` |
| ARGO eval | `python3 eval_run.py -c config_argo.json -r <checkpoint> --out saved/eval.json` |
| L3 download list | `python3 scripts/download_l3_products.py --product all_scaffold` |
| L3 SSH sample day | `python3 scripts/download_l3_products.py --product ssh_l3_historical --date 2020-01-15` |
| L3 rasterization smoke | `python3 scripts/build_l3_samples.py -c config_argo_l3_smoke.json --max-samples 20` |

L3 sample generation rasterizes Copernicus L3 SSH + ERA5 wind around ARGO targets into mask-native bundles (`value`, `mask`, `age`, `uncertainty`, `count`). SST/SMAP deferred.

## Config fields

```json
{
  "data_loader": {
    "args": {
      "split_mode": "chronological",
      "train_frac": 0.7,
      "val_frac": 0.15,
      "test_frac": 0.15
    }
  }
}
```

Explicit date ranges:

```json
{
  "split_mode": "chronological",
  "split_config": {
    "train": {"start": "2015-01-01", "end": "2019-12-31"},
    "val": {"start": "2020-01-01", "end": "2020-12-31"},
    "test": {"start": "2021-01-01", "end": "2021-12-31"}
  },
  "unassigned": "exclude"
}
```

## L4 augmentation

Not yet implemented. When added, it must:

- Label synthetic vs real observations with source-flag channels.
- Never silently replace L3 missingness with L4 fills.
- Record augmentation settings in result metadata.

## Known limitations

- ARGO cache inputs are still L4 gridded (v2 COAPS); L3 pipeline is scaffold-only.
- ISAS configs retain random split until ISAS temporal census is run.
- Readiness diagnostics, physics-loss hook, and ensemble aggregation are pending (Phases 7–9).

## Next steps

1. Wire L3 mask-native channels into PatchConvMLP (Phase 5).
2. Extend model input channels for value/mask/age/uncertainty/count bundles.
3. Stratified baseline evaluation reports.
4. Static-stability readiness diagnostic on saved predictions.
