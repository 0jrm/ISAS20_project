# Handoff — L4 patch underperformance: root cause + data-gap downloads (2026-07-03)

**Question investigated:** why do the ARGO L4-patch models (raw 1.614, anom 1.767 T RMSE) land at/below climatology (1.657) on the chronological test split while the point model gets 0.416?

**Answer: stale satellite data, not architecture or hyperparameters.** Plus a secondary, independently fixable training issue (missing input standardization).

**tmux session with the downloads running: `satdl`** (windows: `ssh`, `sss`, `ostia`). Attach with `tmux attach -t satdl`.

---

## 1. Root cause (primary)

- The local satellite archives end at: OSTIA SST **2021-01-01**, CMEMS SSS **2020-12-31**, CMEMS SSH **`SSH_2020.nc`**. The OSTIA PO.DAAC bulk script's URL list was generated ending exactly at 2021-01-01, so the truncation was baked into the original download.
- `utils/retrieve_sat.py::select_candidate_file` picks the **nearest-dated file with no tolerance limit** → every profile after ~2021-01-01 silently received the last archived field. Fingerprint: from Feb 2021 on, **100% of stations have satellite patches identical at all 7 time steps** (all 7 daily lookups clamp to the same file).
- Split contamination (chronological 70/15/15): **train 0% stale (2015-03..2020-12), val 86.5% stale (2020-12..2021-05), test 100% stale (2021-05..2022-02)**. Test-window L4 SST is frozen near 24 °C through summer (truth ~29.5 °C); center-pixel corr vs the point cache drops from 0.95–0.99 to 0.1–0.5.
- Also broken: `basin_sss/sst/ssh` scalars are exactly **0.0 for every test sample** (basin daily means end too; `np.nan_to_num` zero-fills silently).
- The point model is unaffected because the argo_v2 point cache takes sat values from the v2 pickle's pre-matched data, not from these archives.
- **Date-bug warning:** cache `JULD` is MATLAB datenum; `date.fromordinal(int(juld))` overstates dates by 366 days. Use the HDF5 `stations/julian_date` (astropy JD; 2021-01-01 = JD 2459215.5) for true dates. This offset is what hid the staleness in earlier checks.
- Consequences: **every patch-vs-point comparison on the chronological split to date is invalid**, and val-based early stopping was selecting models that ignore satellite inputs (val itself is 86% garbage).

## 2. Secondary finding: train-side underfit → standardize inputs

Patch train T RMSE was 1.19 vs point 0.40 even on clean (train) data. Controlled retrains (10-min diagnostics, `saved/models/NeSPReSO2_ARGO_GoM_patch_l4/diag_e1_bs512` and `diag_e2_bs512_std`):

| run | change | best train profile_rmse (T+S)/2 |
|---|---|---|
| 0701_102436 (original) | batch 2755 (auto full-batch from `batch_size: 0`) | 0.648 |
| diag_e1_bs512 | batch 512 only | 0.763 (no help) |
| diag_e2_bs512_std | batch 512 + **z-scored inputs** (train-split stats) | **0.150** in 529 epochs |

Raw input scales (bathy ~2700, SSS ~36, SST ~25, SSH ~0.4, harmonics ~1) wreck conditioning. Standardized cache built by `NeSPReSO2_onTemplate/diagnostics/stale_sat/make_std_cache.py`. Also noted: the conv branch (GroupNorm + AvgPool→1×1×1) extracts only patch level (center-broadcast ablation changes nothing), and local sss/sst/ssh never enter the scalar encoder (`n_enc=10` = harmonics+basin+bathy).

**Diagnostic scripts preserved in `NeSPReSO2_onTemplate/diagnostics/stale_sat/`:** `diag_patch.py` (per-split RMSE, drift, ablations), `e0_point_equiv.py` (MLP on point-equivalent features), `cmp_sat_sources.py` (L4 vs point-cache corr by time), `h5_stale_check.py` (time-constant patch fingerprint), `split_vs_stale.py` (stale fraction per split), `make_std_cache.py`. Memory: `l4-patch-stale-sat-root-cause`.

## 3. Data downloads (RUNNING in tmux `satdl`)

Coverage needed: through 2022-02-27 (last profile) + temporal_pad. ~54 GB total; 1.1 TB free. Credentials already configured: `~/.copernicusmarine/` (Copernicus, works with copernicusmarine 2.1.2 in env `nespreso`) and `~/.netrc` (Earthdata, for PO.DAAC).

| window | what | command (exact, restartable) |
|---|---|---|
| `ssh` | C3S DUACS L4 daily SSH, years 2021+2022 (~21 GB, yearly files) | `/conda/jmiranda/miniconda/envs/nespreso/bin/python /unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project/utils/download_SSH.py 2021 2022 2>&1 \| tee -a /unity/g2/jmiranda/SubsurfaceFields/Data/CMEMS/SSH/download_2021_2022.log` |
| `sss` | CMEMS MULTIOBS daily SSS 2021-01-01..2022-02-28 (~424 files × 83 MB; skips existing) | `/conda/jmiranda/miniconda/envs/nespreso/bin/python /unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project/utils/download_SSS_range.py 2>&1 \| tee -a /unity/g2/jmiranda/SubsurfaceFields/Data/CMEMS/SSS/download_2021_2022.log` |
| `ostia` | OSTIA L4 REP SST 2021-01-02..2022-03-01 (~421 files × 16 MB) | `/conda/jmiranda/miniconda/envs/aieoastorch/bin/podaac-data-downloader -c OSTIA-UKMO-L4-GLOB-REP-v2.0 -d /unity/g2/jmiranda/SubsurfaceFields/Data/OISST/OSTIA --start-date 2021-01-02T00:00:00Z --end-date 2022-03-01T00:00:00Z 2>&1 \| tee -a /unity/g2/jmiranda/SubsurfaceFields/Data/OISST/OSTIA/download_2021_2022.log` |

Script changes made: `utils/download_SSH.py` now takes years as CLI args and no longer passes `force_download` (removed in copernicusmarine 2.x); new `utils/download_SSS_range.py` (skip-existing date loop). OSTIA REP v2.0 ends 2022-05-31 — fine for this record, remember if profiles ever extend.

Completion checks:
```bash
ls /unity/g2/jmiranda/SubsurfaceFields/Data/CMEMS/SSH/SSH_202[12].nc
ls /unity/g2/jmiranda/SubsurfaceFields/Data/CMEMS/SSS/SSS_2021*.nc /unity/g2/jmiranda/SubsurfaceFields/Data/CMEMS/SSS/SSS_2022*.nc | wc -l   # expect 424
ls /unity/g2/jmiranda/SubsurfaceFields/Data/OISST/OSTIA/2021*.nc /unity/g2/jmiranda/SubsurfaceFields/Data/OISST/OSTIA/2022*.nc | wc -l        # expect ~421
grep -i "FAILED\|Error" /unity/g2/jmiranda/SubsurfaceFields/Data/CMEMS/SSS/download_2021_2022.log
```

## 4. Next steps after downloads finish

All from `NeSPReSO2_onTemplate/`, conda env `nespreso`.

1. **Regenerate stale satellite batches** (batch files are by station index = pickle order, not date, so find the affected ones by date):
   ```bash
   python - <<'EOF'
   import h5py, glob, os
   SAT = "/unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project/data/NeSPReSO_v2_ARGO_GoM_sat"
   stale = []
   for f in sorted(glob.glob(f"{SAT}/argo_sat_batches/argo_sat_batch_b0100_*.h5")):
       with h5py.File(f) as h:
           if h["stations"]["julian_date"][:].max() > 2459215.5:  # 2021-01-01
               stale.append(f)
   print(f"Deleting {len(stale)} stale batch files")
   for f in stale:
       os.remove(f)
   merged = f"{SAT}/satellite_NeSPReSO_v2_ARGO_GoM.h5"
   if os.path.exists(merged):
       os.remove(merged)
   EOF
   # re-run (resumes, refetching only missing batches):
   srun --ntasks=1 --cpus-per-task=8 python ../utils/generate_argo_satellite_data.py -c config/argo/config_argo_patch_l4.json
   ```
2. **Guard against recurrence** (small code fixes, not yet done):
   - `utils/retrieve_sat.py::select_candidate_file`: reject candidate if |file_date − query| > ~2 days for daily products → return None → NaN+mask instead of stale data.
   - `preproc/basin_stats.compute_basin_daily_means` and `build_argo_l4_input_matrix`: fail loudly (or emit NaN+mask) instead of zero-filling missing basin means.
   - Re-check `h5_stale_check.py` shows 0% time-constant patches after regeneration.
3. **Rebuild caches** (config hash changes only if config changes — use `--force`):
   ```bash
   python preproc/export_argo_l4_cache.py -c config/argo/config_argo_patch_l4.json --force
   python preproc/export_argo_l4_cache.py -c config/argo/config_argo_patch_l4_anom.json --force
   ```
4. **Add input standardization to the patch pipeline** (E2 proved the win: train 0.65→0.15). Either bake z-scoring into `export_argo_l4_cache.py` (store mean/std of the **train split** in the cache, as `make_std_cache.py` does) or normalize in the model. Then full retrains with explicit batch size:
   ```bash
   python train.py -c config/argo/config_argo_patch_l4.json --bs 512 -id patch_l4_fixedsat
   python train.py -c config/argo/config_argo_patch_l4_anom.json --bs 512 -id patch_l4_anom_fixedsat
   ```
5. **Full eval**: rerun the comparison notebook (`notebooks/build_anom_notebook.py` → `notebooks/compare_anom_point_patch.ipynb`) and re-judge patch-vs-point. Expect val-based early stopping to behave now that val inputs are real. Only after this, consider architecture work (feed center-pixel sss/sst/ssh into the scalar encoder; revisit GroupNorm/AvgPool in the conv branch).
6. Independent of all this: anom **point** loss-scale retune is still open (memory `anom-phase-a-results`).
