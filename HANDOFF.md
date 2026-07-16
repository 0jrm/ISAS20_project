# Session handoff — dissertation data foundation

**Branch:** `residual_cube`  
**Updated:** 2026-07-16  
**Code home:** [`NeSPReSO2_onTemplate/`](NeSPReSO2_onTemplate/)  
**Conda:** `nespreso`

---

## Status: density-shift diags → representation plumbing (not schedule)

| Phase | Status |
|-------|--------|
| 0–1 | Done |
| 2 | Partial — T2 OPEN; v3 HDF5 regen **advancing** (batch 100–200 of 4145; not a restart) |
| 3 | Soft gate FAIL; densonly + shift diags done — **branch = representation_plumbing** |
| 4 | Informational — Spearman 0.65 PASS / ENCE 0.33 MISS |
| 5–6 | Blocked until skill recovery (or §3.6 option-2) + R4 golden |

**Do not merge to main until a phase gate passes.**

---

## Density-shift diagnostics (eval-only)

Report: [`reports/phase3_density_shift_diag.md`](reports/phase3_density_shift_diag.md)

| era | clim mse_σ | densonly | v10 | **argo16** |
|-----|------------|----------|-----|------------|
| val | 1.14 | 0.43 | 1.16 | **0.13** |
| test | 1.26 | 0.91 | 1.74 | **0.21** |
| test/val | **1.11** | **2.12** | 1.50 | **1.57** |

**Read:**
1. Clim hardness only +11% val→test — densonly’s 2.1× jump is **not** “the era got 2× harder vs clim.”
2. **argo16 implied density (0.21 test) ≪ densonly (0.91)** and degrades less → signal is in the inputs and extractable. Monotone / clim-residual plumbing is the suspect, not a pure informational ceiling.
3. Shrinkage `var(δa_pred)/var(δa_true) ≈ 0.0002` both eras; mean|δa|_pred ≈ 0.09–0.15 vs true ≈ 5.7–6.9 → Finding-2 under-correction confirmed (δa collapsed to clim).
4. Monthly test errors (2021-05 → 2022-02): densonly tracks clim seasonality with a high floor; argo16 stays low — no flat-then-jump SSS fingerprint.

**Keep v10 spice frozen** (blame-split true σ₀+pred τ = 0.393). §3.6 option 2 still the floor.

---

## Next (no more λ / densonly retries)

1. Input-ablation: does SSH/adt move the density branch?
2. Month-resolved climatology so δa carries interannual only; check softplus / train-era std on test targets.
3. Continue v3 HDF5 → error channels (still useful; not the primary density fix per argo16 control).
4. If still stuck → best-skill + isotonic projection (§3.6 opt-2).

```bash
tmux attach -t v3_hdf5_regen
srun --ntasks=1 --cpus-per-task=8 conda run -n nespreso \
  python scripts/phase3_density_shift_diag.py
```
