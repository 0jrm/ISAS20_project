# Phase 3 — density_spice eval

**Checkpoint:** `saved/argo_densityspice_lowrank/models/NeSPReSO2_ARGO_GoM_densityspice_lowrank_spice/lowrank_sigma0_spice_v3/model_best.pth`  
**Cache:** `../data/cache/train_ready_0f6129b27ddb.pkl`  
**Split:** test n=623

**Gate:** PASS

- σ₀ profile rate (post-inv): 0.3066; pre-inv neg: 0; near-zero OK: True
- N² profile / level @ 1e-8: 0.2183 / 0.002360
- overall T RMSE: 0.5621
- vs clean chrono argo16 0.5367: ratio 1.047 (floor 0.5904, pass=True)
- vs published-random argo16: ratio 1.352 (side-by-side only)
- MLD RMSE: 34.5294868566109
- dρ/dz RMSE: 0.007448592943356057
- inversion fail frac: 0.0000

## T/S RMSE by depth band

| band | T RMSE | S RMSE | vs T1-A recon |
|------|--------|--------|---------------|
| 0-50 | 1.1415 | 0.3296 | 5.637 |
| 50-200 | 1.2962 | 0.1690 | 5.820 |
| 200-800 | 0.6257 | 0.0875 | 5.875 |
| >800 | 0.1303 | 0.0102 | — |

_Gate note:_ STOP uses pre-inv σ₀ (low-rank) or post-inv near-zero (full-rank) + overall T ≤ clean-chrono argo16×1.10 (floor 0.5903). Published-random 0.4158 reported side-by-side only.

## Phase 2 caveat

T2 stale gate OPEN (0% SSS/SST/SSH on val/test). Full HDF5 lacks v3 error fields (`err_sla` / `analysis_error` / `sos_error` only in `*_err_smoke.h5`). density_spice cache has no `inputs_err` — headline metrics may be SSS-confounded only if stale returns; currently not. Formal product-error channels not in model inputs.

## Stability architecture (required claim)

**Guarantee:** `inference_isotonic` — *stable by construction at inference*, **not** hard constraint in the head.

The low-rank head is `σ̂₀ = clim + scores @ V` (linear in σ₀). Measured on this test set **before** isotonic:
- neg profile rate: **0.456** (284 / 623)
- neg iface rate: 0.0776

Isotonic projection (`project_monotone_sigma0_ctrl`, §3.6 opt-2 / T1-D) is **mandatory** on the inference path (wired in `decode_density_spice_to_ts`). Cost: σ₀ RMSE 0.0076; ΔT RMSE ≈ −0.0002 °C.

## Blame-split (T RMSE, chrono test)

| recipe | T RMSE |
|--------|-------:|
| oracle R=16 scores + true spice | 0.098 |
| pred dens + true spice | **0.250** |
| true dens + pred spice | **0.404** |
| both predicted (headline) | **0.562** |

Spice quality (~0.404 with oracle dens) is in line with v10-era ~0.393; density is carrying more of the pass.

## Eval hygiene

- **Test evals consumed** for σ₀-space candidate family: **2** (v2 near-miss 0.598 → spice_v3 pass 0.562). Iteration was toward the gate; garden-of-forking-paths noted.
- Rejected a-space sibling: 1 test eval (T=0.830) — see `finding_compress_physical_space.md`.
- Floor provenance: `gate_floor_provenance.md` (clean chrono argo16 0.5367 → floor 0.5903; leakage erratum on density-shift diag).
- **Dissertation claim:** Phase 5 matrix 3-seed mean ± std. Today's pass = **matrix admission**, not the headline number.
