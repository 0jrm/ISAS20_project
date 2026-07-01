# L3 data download — user handoff & homework

**Session:** 2026-06-18  
**Branch:** `phase3-l3-rasterization`  
**Repo:** `/unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project-phase3-commit`  
**Conda env:** `nespreso`

This file is the **user-facing** checklist for finishing L3 surface downloads and rebuilding the mask-native ARGO cache. Agent/implementation context stays in [`HANDOFF.md`](HANDOFF.md).

---

## What this session did

1. Created shared raw data root at **`~/SubsurfaceFields/Data/raw/`** (not inside the git repo).
2. Fixed three pipeline bugs that blocked real SSH coverage:
   - `download_l3_products.py`: removed `no_directories` (incompatible with `skip_existing` in `copernicusmarine` 2.1.2).
   - `preproc/l3_rasterize.py`: recursive file discovery (`rglob`) for Copernicus nested subdirs.
   - `preproc/l3_rasterize.py`: longitude normalization (altimetry 0–360° vs ARGO −180–180°).
3. Pointed `config/argo/config_argo_l3_smoke.json` → `io.l3.raw_root: /unity/g2/jmiranda/SubsurfaceFields/Data/raw`.
4. Installed `cdsapi` in `nespreso` (ERA5 still blocked until you add credentials).
5. Started a **background 2020 daily SSH download** loop.
6. Verified non-zero SSH rasterization on five `2020-01-15` profiles (~10–17% patch coverage; nearest track 14–74 km).

**Not done yet:** ERA5 wind download, full 2020 SSH completion, full ARGO-span download (2015–2022), full L3 train cache rebuild.

---

## Current state (check when you return)

Run this snapshot first:

```bash
# SSH download progress
rg '^=== ' ~/SubsurfaceFields/Data/raw/ssh_l3_2020_download.log | tail -3
find ~/SubsurfaceFields/Data/raw/altimetry_l3 -name '*.nc' | wc -l
du -sh ~/SubsurfaceFields/Data/raw/altimetry_l3

# Is the background job still running?
pgrep -af 'download_l3_products.*ssh_l3' || echo "no SSH download process"

# ERA5 credentials
test -f ~/.cdsapirc && echo "CDS ready" || echo "CDS missing — see Homework step 1"
ls ~/SubsurfaceFields/Data/raw/era5/
```

**At session handoff (2026-06-18 ~20:05 UTC):**

| Item | Status |
|------|--------|
| Copernicus Marine login | **OK** (`~/.copernicusmarine/.copernicusmarine-credentials`) |
| CDS API (`~/.cdsapirc`) | **Missing** — blocks ERA5 |
| SSH smoke day `2020-01-15` | **Done** |
| SSH full 2020 daily loop | **Running** (~day 11/366, ~68 files, ~30 MB) |
| ERA5 2020 monthly | **Not started** |
| L3 samples with real SSH | **Verified** on 5 profiles (`l3_samples_b0d638f3f31e.pkl`) |
| Wind channels in L3 cache | **Still zero** (no ERA5 files) |

---

## Data layout

```
~/SubsurfaceFields/Data/raw/
├── altimetry_l3/          # Copernicus L3 along-track SSH
│   └── SEALEVEL_GLO_PHY_L3_MY_008_062/.../*.nc   # nested subdirs (expected)
├── era5/                  # ERA5 monthly GoM u10/v10 (empty until CDS setup)
│   └── era5_u10_v10_YYYYMM_gom.nc                # flat names (pipeline expects this)
└── ssh_l3_2020_download.log
```

**Manifest (repo):** `data/manifests/download_manifest.jsonl` — append-only log of download calls.

**Processed L3 batches (repo):** `data/processed/l3_samples_<l3_hash>.pkl`  
**Train caches (repo):** `data/cache/train_ready_l3_<config_hash>_<l3_hash>.pkl`

**Important:** Raw files live on shared FS under `~/SubsurfaceFields/Data/raw/`. Repo `data/raw/` is **not** used.

---

## Uncommitted code changes (review before commit)

These fixes are in your working tree but not committed:

| File | Change |
|------|--------|
| `NeSPReSO2_onTemplate/scripts/download_l3_products.py` | Copernicus API compat |
| `NeSPReSO2_onTemplate/preproc/l3_rasterize.py` | `rglob` + lon normalization |
| `NeSPReSO2_onTemplate/config/argo/config_argo_l3_smoke.json` | absolute `raw_root` |
| `NeSPReSO2_onTemplate/selfcheck.py` | `test_l3_lon_normalization_gom_bbox` |
| (+ other branch edits from prior sessions) | L4 augment, export_l3_cache, etc. |

```bash
cd /unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project-phase3-commit
git diff --stat
```

---

## Homework checklist

Work through in order. Each step has a verification gate.

### Step 0 — Environment

```bash
conda activate nespreso
cd /unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project-phase3-commit/NeSPReSO2_onTemplate
```

Gate:

```bash
conda run -n nespreso python3 -c "import copernicusmarine, cdsapi, netCDF4; print('deps ok')"
copernicusmarine login   # only if credentials expired
```

---

### Step 1 — CDS API credentials (required for ERA5)

ERA5 wind is **not** on Copernicus Marine; it uses the Climate Data Store API.

1. Register / log in: https://cds.climate.copernicus.eu/
2. Profile → API key → copy UID and API key.
3. Create `~/.cdsapirc`:

```ini
url: https://cds.climate.copernicus.eu/api/v2
key: <uid>:<api-key>
```

4. Restrict permissions (recommended):

```bash
chmod 600 ~/.cdsapirc
```

Gate:

```bash
test -f ~/.cdsapirc && echo "cdsapirc present"
conda run -n nespreso python3 -c "import cdsapi; cdsapi.Client(); print('CDS client ok')"
```

**Note:** First CDS request may sit in queue for minutes to hours. That is normal.

---

### Step 2 — Let 2020 SSH finish (or resume)

The background job loops all **366 days of 2020** into `~/SubsurfaceFields/Data/raw/altimetry_l3/`.

Monitor:

```bash
tail -f ~/SubsurfaceFields/Data/raw/ssh_l3_2020_download.log
```

If the job died, resume manually (idempotent — skips existing files):

```bash
cd NeSPReSO2_onTemplate
DATA_ROOT=/unity/g2/jmiranda/SubsurfaceFields/Data/raw

for d in $(seq 1 366); do
  day=$(python3 -c "import datetime as dt; print((dt.date(2020,1,1)+dt.timedelta(days=$d-1)).isoformat())")
  echo "=== $day ==="
  conda run -n nespreso python3 scripts/download_l3_products.py \
    --product ssh_l3_historical --date "$day" --data-root "$DATA_ROOT" \
    || echo "FAILED $day"
done
```

Gate (expect hundreds of files, not just 6):

```bash
find ~/SubsurfaceFields/Data/raw/altimetry_l3 -name '*20200115*.nc' | wc -l   # should be >6
find ~/SubsurfaceFields/Data/raw/altimetry_l3 -name '*.nc' | wc -l            # grows toward ~thousands for full year
```

**Runtime expectation:** ~2–3 min/day × 366 ≈ **12–18 hours** for all of 2020.

---

### Step 3 — Download ERA5 wind for 2020

GoM bbox is already the script default: `[N,W,S,E] = [35, -100, 15, -75]`.

```bash
cd NeSPReSO2_onTemplate
DATA_ROOT=/unity/g2/jmiranda/SubsurfaceFields/Data/raw

for m in $(seq 1 12); do
  echo "=== ERA5 2020-$(printf '%02d' $m) ==="
  conda run -n nespreso python3 scripts/download_l3_products.py \
    --product era5_wind --year 2020 --month $m --data-root "$DATA_ROOT"
done
```

Gate:

```bash
ls -lh ~/SubsurfaceFields/Data/raw/era5/era5_u10_v10_2020*_gom.nc
# expect 12 files, each roughly tens of MB
```

Single-month smoke (if you want to test CDS before committing to all 12):

```bash
conda run -n nespreso python3 scripts/download_l3_products.py \
  --product era5_wind --year 2020 --month 1 \
  --data-root /unity/g2/jmiranda/SubsurfaceFields/Data/raw
```

---

### Step 4 — Rebuild L3 processed samples + train cache

After SSH **and** ERA5 exist for your target window:

```bash
cd NeSPReSO2_onTemplate

# Small smoke (profiles near 2020-01-15)
srun --ntasks=1 --cpus-per-task=8 python3 scripts/build_l3_samples.py \
  -c config/argo/config_argo_l3_smoke.json --max-samples 20 --anchor-date 2020-01-15 --force

# Full dataset (4145 profiles) — slow; run when raw coverage is adequate
srun --ntasks=1 --cpus-per-task=8 python3 scripts/build_l3_samples.py \
  -c config/argo/config_argo_l3_smoke.json --export-train-cache --force
```

Gate — non-zero SSH **and** wind coverage:

```bash
conda run -n nespreso python3 <<'PY'
import pickle, glob
p = sorted(glob.glob("../data/processed/l3_samples_*.pkl"))[-1]
print("file:", p)
with open(p, "rb") as f:
    batch = pickle.load(f)
for s in batch["samples"][:5]:
    t = s["target"]["time"]
    ssh = s["coverage"]["ssh"]["coverage_fraction"]
    wu = s["coverage"]["wind_u"]["coverage_fraction"]
    print(t, "ssh", round(ssh, 4), "wind_u", round(wu, 4))
PY
```

**Pass criteria for smoke:** `ssh > 0` and `wind_u > 0` on 2020-Jan profiles.  
If `ssh == 0` with files present, check lon normalization fix is in your checkout (`preproc/l3_rasterize.py`).

---

### Step 5 — Run selfcheck + optional smoke train

```bash
srun --ntasks=1 --cpus-per-task=8 python3 selfcheck.py

srun --ntasks=1 --cpus-per-task=8 --gres=gpu:1 python3 train.py -c config/argo/config_argo_l3_smoke.json
```

Gate: selfcheck exits 0; train completes 2 epochs under `saved/smoke_argo_l3/`.

---

### Step 6 — (Later) Extend downloads to full ARGO span

ARGO export is **2015–2022** (4145 profiles). For dissertation training you eventually need SSH + ERA5 across that span, not just 2020.

**SSH** — loop years/months or days (same script, idempotent):

```bash
DATA_ROOT=/unity/g2/jmiranda/SubsurfaceFields/Data/raw
for year in 2015 2016 2017 2018 2019 2020 2021 2022; do
  # leap years: use python to enumerate days
  python3 - <<PY
import datetime as dt, subprocess
start = dt.date($year, 1, 1)
end = dt.date($year, 12, 31)
d = start
while d <= end:
    print(d.isoformat())
    d += dt.timedelta(days=1)
PY
done | while read day; do
  conda run -n nespreso python3 scripts/download_l3_products.py \
    --product ssh_l3_historical --date "$day" --data-root "$DATA_ROOT" \
    || echo "FAILED $day"
done
```

**ERA5** — 96 monthly files (2015-01 … 2022-12):

```bash
for year in $(seq 2015 2022); do
  for m in $(seq 1 12); do
    conda run -n nespreso python3 scripts/download_l3_products.py \
      --product era5_wind --year $year --month $m --data-root "$DATA_ROOT"
  done
done
```

**Scope warning:** Full daily SSH for 8 years is **~2900 days** × ~2–3 min ≈ **4–6 days** of wall time. Consider downloading year-by-year and rebuilding caches incrementally, or batching on HPC with `srun`/`sbatch`.

**2022+ SSH NRT:** profiles in 2022 may need `ssh_l3_nrt` (`SEALEVEL_GLO_PHY_L3_NRT_008_044`) in addition to historical product — deferred until you validate 2015–2021.

---

## Quick reference commands

```bash
# List available download products
python3 scripts/download_l3_products.py --product all_scaffold

# One SSH day
python3 scripts/download_l3_products.py --product ssh_l3_historical \
  --date 2020-06-01 --data-root ~/SubsurfaceFields/Data/raw

# One ERA5 month
python3 scripts/download_l3_products.py --product era5_wind \
  --year 2020 --month 6 --data-root ~/SubsurfaceFields/Data/raw

# L3 raw coverage report
srun --ntasks=1 --cpus-per-task=8 python3 scripts/data_census.py -c config/argo/config_argo_l3_smoke.json
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `MutuallyExclusiveArguments: skip_existing, no_directories` | Old downloader | Pull/use fixed `download_l3_products.py` (this session) |
| SSH files on disk but `ssh coverage == 0` | Lon 0–360 vs −180–180 | Ensure `l3_rasterize.py` has `_normalize_lon_deg` + `rglob` |
| `find_ssh_files_for_day` returns 0 | Files nested in CMEMS subdirs | Same `rglob` fix |
| ERA5 `Missing/incomplete configuration file` | No `~/.cdsapirc` | Homework step 1 |
| CDS request hangs | Queue | Wait; check https://cds.climate.copernicus.eu/live/status |
| Wind coverage 0, SSH OK | No ERA5 files | Homework step 3 |
| Full cache build very slow | 4145 profiles × file I/O | Normal; use `--max-samples` for dev |
| `wind_u` OK but stratified eval empty | Checkpoint/cache mismatch | Pair checkpoint with the cache it was trained on |

---

## What “done” looks like

Minimum viable (thesis smoke path):

- [ ] `~/.cdsapirc` configured
- [ ] ERA5 2020-01 … 2020-12 in `~/SubsurfaceFields/Data/raw/era5/`
- [ ] SSH 2020 daily files in `~/SubsurfaceFields/Data/raw/altimetry_l3/`
- [ ] `build_l3_samples.py --anchor-date 2020-01-15` shows **ssh > 0 and wind_u > 0**
- [ ] `selfcheck.py` passes
- [ ] Optional: 2-epoch `config/argo/config_argo_l3_smoke.json` train completes

Dissertation training path:

- [ ] SSH + ERA5 for **2015–2022**
- [ ] Full `build_l3_samples.py --export-train-cache --force` (no `--max-samples`)
- [ ] L3 smoke train on full cache, then stratified eval

---

## Related docs

| Doc | Purpose |
|-----|---------|
| [`HANDOFF.md`](HANDOFF.md) | Agent session status, phase table, eval rules |
| [`PLAN-phase3-l3-rasterization.md`](PLAN-phase3-l3-rasterization.md) | Phase 3 design |
| [`context.txt`](context.txt) | Product IDs, download patterns, scientific rationale |
| [`NeSPReSO2_onTemplate/scripts/download_l3_products.py`](NeSPReSO2_onTemplate/scripts/download_l3_products.py) | Downloader implementation |

---

## One-liner resume (after CDS is set up)

```bash
conda activate nespreso
cd /unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project-phase3-commit/NeSPReSO2_onTemplate

# 1) finish ERA5 2020
for m in $(seq 1 12); do python3 scripts/download_l3_products.py --product era5_wind --year 2020 --month $m --data-root ~/SubsurfaceFields/Data/raw; done

# 2) rebuild cache
srun --ntasks=1 --cpus-per-task=8 python3 scripts/build_l3_samples.py -c config/argo/config_argo_l3_smoke.json --max-samples 20 --anchor-date 2020-01-15 --export-train-cache --force

# 3) gate
srun --ntasks=1 --cpus-per-task=8 python3 selfcheck.py
```
