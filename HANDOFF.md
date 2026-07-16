# Session handoff — dissertation data foundation

**Branch:** `residual_cube`  
**Updated:** 2026-07-16  
**Code home:** [`NeSPReSO2_onTemplate/`](NeSPReSO2_onTemplate/)  
**Conda:** `nespreso`

---

## Phase 3 acceptance — CLEARED → Phase 4 unblocked

| Gate | Result |
|------|--------|
| Round-trip (500 test, ref `gsw`) | **PASS** — max\|ΔT\|/\|ΔS\| ~1e-14, Newton fail=0 |
| Truth-projection softplus E vs A T-RMSE ≤10%/band | **PASS** — T E/A = 0.755 / 0.700 / 0.821 / **0.970** ([`reports/t1_basis_stability.md`](reports/t1_basis_stability.md)) |
| Deep-band root cause + fix | **Done** — isotonic-before-softplus encode ([`reports/e_deep_band_diagnostic.md`](reports/e_deep_band_diagnostic.md)) |
| Batch-schema resume guard | Landed + regression green |
| selfcheck | Green + density_spice isotonic encode check |
| λ_ρ / λ_τ | Re-derived after cache rebuild; unchanged `8e-06` / `0.008256` |

---

## Human sign-off (2026-07-16) — representation framing

**Approved dissertation framing:**

> *No soft basis fixes stability; hard monotonicity does, at cost X.*

- Soft changes (B joint EOF, C density/spice PCA) leave historical σ₀ profile rate ≈ A (~0.22).
- Load-bearing mechanism is **truncation itself**, not T/S basis separateness.
- Hard softplus control-grid (E) → historical σ₀ profile rate **0.0000**.
- **Cost X is negative in the upper ocean** (big win below).

---

## Big win (document in dissertation / papers)

Hard monotonicity **improves** T RMSE vs separate PCA-16 while zeroing σ₀ violations:

| band | A T-RMSE | E T-RMSE | E/A |
|------|----------|----------|-----|
| 0–50 m | 0.202 | 0.153 | **0.76** |
| 50–200 m | 0.223 | 0.156 | **0.70** |
| 200–800 m | 0.107 | 0.088 | **0.82** |
| >800 m | 0.016 | 0.016 | **0.97** |

Historical σ₀ profile rate: **0.215 → 0.000**. This T1 table is a dissertation figure, not just a gate.

Deep-band FAIL (E/A=1.63) was a softplus-clamp artefact: ~12% of linear-interp control-grid increments are negative; clamping injects cumulative σ₀ bias that peaks below 800 m. Fix: `project_monotone_sigma0_ctrl` (isotonic) before `encode_a_from_sigma0_ctrl` (default). After fix E matches D and beats A in every band.

---

## Done this close-out

1. Framing sign-off documented in T1 report + this handoff + PLAN changelog.
2. Diagnostic `scripts/diagnose_e_deep_band.py` → `reports/e_deep_band_diagnostic.md`.
3. Fix in `model/density_spice.py`; wired into T1, proj-cost, and `export_v2_cache` targets.
4. Density_spice cache rebuilt: `data/cache/train_ready_cd9e08b6c630.pkl`.
5. λ scales re-derived (unchanged).

---

## Next — Phase 4 (unblocked)

Start Phase 4 from [`PLAN-v2-recovery.md`](PLAN-v2-recovery.md) §4: probabilistic head (CRPS preferred), latent→profile covariance, input-error conditioning.

```bash
cd NeSPReSO2_onTemplate
srun --ntasks=1 --cpus-per-task=8 conda run -n nespreso python scripts/t1_basis_stability.py --gsw-backend gsw --out-md ../reports/t1_basis_stability.md
srun --ntasks=1 --cpus-per-task=8 conda run -n nespreso python -m model.density_spice
```

**R4 golden:** still Phase 5 prerequisite only.
