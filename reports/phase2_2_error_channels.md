# Phase 2.2 — verified error field names (2026-07-16)

Recorded before regenerating ARGO satellite HDF5. Do not invent names.

| Product | Value var (v2) | Error var (on disk) | Notes |
|---------|----------------|---------------------|-------|
| SSH (DUACS L4) | `adt` / `sla` / `ugos` / `vgos` | **`err_sla`** | `err_ugosa` / `err_vgosa` listed in download script but **absent** from current `SSH_YYYY.nc` files |
| SST (OSTIA) | `analysed_sst` | **`analysis_error`** | Already in downloaded granules; no re-download |
| SSS (CMEMS multiobs) | `sos` | **`sos_error`** | Also has `dos_error`; use `sos_error` |

Config: `utils/v3.json` (`error_channels` + product var lists). Smoke: `utils/check_v3_errors.py`.

Cache schema (when `io.error_groups` set): `inputs_err`, `err_missing`, `input_error_channels`, `input_error_standardization` (log-zscore, e0=1e-6, NaN→train p90). Not concatenated into model `inputs` until Phase 4 `io.use_error_channels`.
