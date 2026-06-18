# Resume: Phase 3 L3 Rasterization + Mask-Native Cache

## Where the last session stopped

The session in transcript `226fab9d` (Jun 18) completed through **PLAN commit order step 7**:

| Step | Status | Key artifacts |
|------|--------|---------------|
| 0–2 Data census, chronological split, ARGO path | **Done** | `base/split_utils.py`, `scripts/data_census.py`, ARGO configs with `split_mode: chronological` |
| 7 L3 downloader scaffolding | **Done** | `scripts/download_l3_products.py` |
| 8–9 L3 rasterization + masked batch loading | **Done** | `preproc/l3_rasterize.py`, `preproc/export_l3_cache.py`, `scripts/build_l3_samples.py`, `config_argo_l3_smoke.json` |

**Next per HANDOFF.md and PLAN.md MVP:** L3 SSH + ERA5 wind mask-native pipeline (SST/SMAP deferred).

## Design constraints

- **Target stays ARGO** — rasterize surface obs around each profile `(lat, lon, time)` from cache JULD/LAT/LON.
- **No fake complete maps** — grid cells are sparse bins; empty cells get `mask=0`.
- **Per-variable feature bundle:** `value`, `mask`, `age` (hours), `uncertainty`, `count`.
- **Reuse patch geometry** from `preproc_isas_sat.py` via `spatial_pad` / `temporal_pad`.
- **Do not change PatchConvMLP yet** (Phase 5).

## Implementation plan

### 1. Config block (`io.l3` + `config_argo_l3_smoke.json`)

### 2. Core rasterization (`preproc/l3_rasterize.py`)

### 3. Sample builder (`preproc/export_l3_cache.py` + `scripts/build_l3_samples.py`)

### 4. Download real MVP data (SSH 2020-01-15 + ERA5 2020-01)

### 5. Batch-loading smoke in `selfcheck.py`

### 6. Tests + data_census L3 coverage

### 7. Update HANDOFF.md and PLAN-dissertation-data-foundation.md

## Verification

```bash
cd NeSPReSO2_onTemplate
srun --ntasks=1 --cpus-per-task=8 python3 selfcheck.py
python3 scripts/download_l3_products.py --product ssh_l3_historical --date 2020-01-15
python3 scripts/download_l3_products.py --product era5_wind --year 2020 --month 1
srun --ntasks=1 --cpus-per-task=8 python3 scripts/build_l3_samples.py -c config_argo_l3_smoke.json --max-samples 20
```
