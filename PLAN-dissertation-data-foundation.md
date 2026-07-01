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
srun --ntasks=1 --cpus-per-task=8 python3 scripts/data_census.py -c config/argo/config_argo.json
```

Inspect `reports/split_design.md`. For the current GoM ARGO export:

| Candidate | Viable? | Notes |
|-----------|---------|-------|
| A (2002–2015 / 2016–17 / 2018–20) | **No** | Data start in 2015 |
| B (chronological 70/15/15) | **Yes** | **Default** |
| C (test = 2020 high-density year) | Yes | High-observation subset |
| D (test = 2015–2018 sparse era) | Yes | Low-observation stress test |
| E (2015–19 / 2020 / 2021) | Yes | Common-overlap explicit dates |

**Recommended default:** Candidate B (`config/argo/config_argo.json` with `split_mode: chronological`).

**Explicit date alternative:** `config/argo/config_argo_chrono_dates.json` (Candidate E).

## High/low data-regime subsets

Defined in `reports/split_design.json` → `recommendation.evaluation_subsets`:

- **high_observation:** year 2020 (1515 profiles — peak density).
- **low_observation:** 2015–2018 (691 profiles — sparse early era).
- **common_overlap:** 2015–2021 (exclude sparse 2022 tail).

Use these for stratified evaluation once L3 coverage metrics exist.

## How to run

| Task | Command |
|------|---------|
| Data census | `python3 scripts/data_census.py -c config/argo/config_argo.json` |
| ARGO smoke train | `python3 train.py -c config/argo/config_argo_smoke.json` |
| ARGO baseline train | `python3 train.py -c config/argo/config_argo.json` |
| ARGO eval | `python3 eval_run.py -c config/argo/config_argo.json -r <checkpoint> --out saved/eval.json` |
| L3 download list | `python3 scripts/download_l3_products.py --product all_scaffold` |
| L3 SSH sample day | `python3 scripts/download_l3_products.py --product ssh_l3_historical --date 2020-01-15` |
| L3 rasterization smoke | `python3 scripts/build_l3_samples.py -c config/argo/config_argo_l3_smoke.json --max-samples 20` |

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

Enabled via `io.l4` in L3 configs (`config/argo/config_argo_l3_l4_smoke.json`). Applies real L3 mask geometry to L4 DUACS SSH fields with source-flag metadata. See [`HANDOFF.md`](HANDOFF.md).

## L3 model inputs (Phase 5)

Mask-native bundles flatten to PatchConvMLP with shape `(C,T,H,W) = (15, 5, 25, 25)` for the default three variables × five features. Config fields:

- `io.l3.variables` — which surface fields (ssh, wind_u, wind_v)
- `io.l3.features` — which channels per variable (`value`, `mask`, `age`, `uncertainty`, `count`)

`sync_arch_with_io()` aligns `arch.args` with the L3 layout; `verify_l3_cache_layout()` catches channel-order mismatches at train/eval time.

## Known limitations

- SST (VIIRS L3U) and SMAP SSS rasterization **deferred**.
- L4 augment: SSH only; wind/SST L4 deferred.
- Value-channel train-split z-score normalization **deferred** (raw values today).
- ISAS configs retain random split until ISAS temporal census is run.
- Physics-loss hook and ensemble aggregation are pending (Phases 8–9).

## Next steps

1. Download real L3/L4/ERA5 for 2020; rebuild caches; verify non-zero coverage.
2. Stratified baseline comparison: L4 point vs L3 vs L4-mask-augment.
3. Steric SSH (RC-2) and uncertainty calibration (RC-4) in readiness diagnostics.
4. Physics-loss hook (Phase 8) and ensemble aggregation (Phase 9).
