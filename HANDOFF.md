# Session handoff — dissertation data foundation

**Branch:** `residual_cube`  
**Updated:** 2026-07-16  
**Code home:** [`NeSPReSO2_onTemplate/`](NeSPReSO2_onTemplate/)  
**Conda:** `nespreso`

---

## Status: Phase 4 smoke acceptance CLEARED

Phase 3 hard-stability gate cleared earlier today; Phase 4 probabilistic smoke is green. See [`reports/phase4_prob_smoke.md`](reports/phase4_prob_smoke.md).

| Phase | Status |
|-------|--------|
| 0–1 | Done (evalphys freeze, T1 mechanism sign-off) |
| 2 | Partial — v3 error fields + census exist; full SSS/v3 cache into density_spice still open |
| 3 | **Done** — monotone density+spice; E/A PASS; framing signed off |
| 4 | **Smoke done** — CRPS/NLL/quantile train; two-stage launcher; dacov; decomposition script |
| 5–6 | Blocked on R4 golden (Phase 5 prereq) + full Phase-4 eval table on real runs |

---

## Human sign-off (representation)

> *No soft basis fixes stability; hard monotonicity does, at cost X.*

Big win: E beats A on T RMSE in all bands while σ₀ profile rate 0.215→0.000 ([`reports/t1_basis_stability.md`](reports/t1_basis_stability.md)).

---

## Phase 4 artifacts

- Config: `config/argo/config_argo_densityspice_prob_smoke.json`
- Launcher: `scripts/train_prob_twostage.py`, `scripts/phase4_prob_smoke.py`
- Cov: `dacov/`
- Decomp: `reports/uncertainty_decomposition.json`

---

## Next

1. Wire Phase 2 remainder: rebuild density_spice cache with `inputs_err` once v3 HDF5 ready; turn `use_error_channels` on for a prob smoke.
2. Longer CRPS two-stage run on full cache (not smoke `max_samples`).
3. Do **not** start Phase 5 matrix until R4 golden is rooted/waived.
4. Optional: T/S Σ via inversion Jacobian in `dacov` (currently ctrl+spice seed only).

```bash
cd NeSPReSO2_onTemplate
srun --ntasks=1 --cpus-per-task=8 --gres=gpu:1 conda run -n nespreso \
  python scripts/phase4_prob_smoke.py
srun --ntasks=1 --cpus-per-task=8 conda run -n nespreso python selfcheck.py
```
