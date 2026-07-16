# Session handoff — dissertation data foundation

**Branch:** `residual_cube`  
**Updated:** 2026-07-16  
**Code home:** [`NeSPReSO2_onTemplate/`](NeSPReSO2_onTemplate/)  
**Conda:** `nespreso`

---

## STOP — Phase 4 still blocked

Phase 3 acceptance **did not fully clear**. Do **not** start Phase 4 until the deep-band RMSE cost is addressed (or the gate is explicitly waived).

| Gate | Result |
|------|--------|
| Round-trip (500 test, ref `gsw`) | **PASS** — max\|ΔT\|≈3e-14, max\|ΔS\|≈7e-15, Newton fail=0 ([`reports/phase3_roundtrip.json`](reports/phase3_roundtrip.json)) |
| Truth-projection softplus E vs A T-RMSE ≤10%/band | **FAIL** — T\[>800\] E/A=**1.629** (0.026 vs 0.016 °C). Other bands PASS (0.76 / 0.70 / 0.83). See [`reports/t1_basis_stability.md`](reports/t1_basis_stability.md) row `E_softplus_phase3` |
| Full HDF5 contaminated by error schema | **NO** — `satellite_NeSPReSO_v2_ARGO_GoM.h5` still value-only (`sos`/`analysed_sst`/`adt`) |
| Batch-schema resume guard | **Landed** + regression green (`utils/test_batch_schema_guard.py`) |
| selfcheck (phase-boundary) | **Green** + documented R4 golden skip |

---

## Done this close-out

1. Commits pushed earlier: `644d9dc` feat(data) v3 errors; `500e635` feat(repr) density+spice.
2. Merge completeness: B/C historical σ₀ rows + F.3 max-abs/RMS table present (no restore needed).
3. Batch-schema guard: refuses resume when error vars missing — *"Regenerate from scratch or use v2 config."*
4. Full density_spice cache: `data/cache/train_ready_cd9e08b6c630.pkl` (N=4145).
5. `λ_ρ`/`λ_τ` frozen in `config/argo/config_argo_densityspice.json` (`8e-06` / `0.008256`).
6. T1 table extended with **E_softplus_phase3** (price of hard stability via Phase 3.2+3.3 path).

---

## Next (before Phase 4)

1. Diagnose T\[>800\] softplus projection RMSE (control-grid spacing / PCHIP clamp at deep end / spice PCA).
2. Re-run T1 E after fix; require E/A ≤ 1.10 for **all** T depth bands.
3. Then Phase 4 is unblocked.

**R4 golden:** still Phase 5 prerequisite only.

```bash
cd NeSPReSO2_onTemplate
srun --ntasks=1 --cpus-per-task=8 conda run -n nespreso python scripts/phase3_roundtrip.py --cache ../data/cache/train_ready_cd9e08b6c630.pkl
srun --ntasks=1 --cpus-per-task=8 conda run -n nespreso python scripts/t1_basis_stability.py --gsw-backend gsw --out-md ../reports/t1_basis_stability.md
cd ../utils && srun --ntasks=1 --cpus-per-task=8 conda run -n nespreso python test_batch_schema_guard.py
```
