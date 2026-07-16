# E deep-band diagnostic (>800 m)

Cache: `/unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project/data/cache/train_ready_4411c65ee518.pkl`  |  test n=623  |  gsw=`gsw`

## Control grid below 800 m

| z_ctrl | Δz phys | Δz̃ |
|--------|---------|-----|
| 881.5 | 98.9 | 3.463 |
| 992.9 | 111.4 | 3.900 |
| 1118.4 | 125.5 | 4.393 |
| 1259.7 | 141.3 | 4.948 |
| 1418.8 | 159.2 | 5.573 |
| 1598.1 | 179.3 | 6.278 |
| 1800.0 | 201.9 | 7.071 |

## Softplus increment pathology (linear-interp ctrl, deep levels)

- Fraction of deep raw increments `< 0`: **0.0000**
- Fraction hitting softplus floor (`raw < 1e-12`): **0.0000**
- Max |encode∘decode − id| on ctrl: 3.942e-02 (deep-only 3.942e-02)
- Max |encode∘decode − id| after isotonic: 2.159e-06

## Ablation (T/S via Newton; spice = PCA-16 unless oracle)

| variant | σ₀ RMSE>800 | T>800 oracle-τ | T>800 spice-PCA | S>800 spice-PCA | pre-inv Δσ₀<0 |
|---------|-------------|----------------|-----------------|-----------------|---------------|
| A_grid_pchip | 0.00256 | 0.0096 | 0.0157 | 0.0016 | 4425 |
| B_softplus_rt | 0.00582 | 0.0233 | 0.0264 | 0.0039 | 0 |
| C_isotonic_pchip | 0.00256 | 0.0096 | 0.0157 | 0.0016 | 0 |
| D_iso_softplus | 0.00256 | 0.0096 | 0.0157 | 0.0016 | 0 |
| E_encode_monotone | 0.00256 | 0.0096 | 0.0157 | 0.0016 | 0 |

## Interpretation keys

- If `A ≈ B` and both ≫ `C`: softplus roundtrip is fine; **linear interp without isotonic** is the deep cost.
- If `B ≫ A`: softplus floor / negative-increment clamp is the cost.
- If oracle-τ ≪ spice-PCA for all: spice PCA (not density) drives deep T error.
- `D_iso_softplus` ≈ `C` ⇒ isotonic-before-encode is the fix for the Phase-3 path.

## Conclusion (2026-07-16)

**Verdict:** `B ≫ A` and `E_encode_monotone ≡ C ≡ D` — the deep-band T1-E FAIL was the softplus negative-increment clamp, not PCHIP, not spice PCA, not deep grid spacing.

- Deep ctrl increments themselves are almost never negative; the bias is injected by **upper-ocean** negative increments (~12% of all ctrl interfaces; peak ~38% near level 5) that softplus clamps to ~0. Cumsum then carries a σ₀ offset into the deep column (max |encode∘decode−id| ≈ 0.039).
- Isotonic projection before encode removes the pathology (roundtrip error → 2e-6; T>800 spice-PCA 0.026 → 0.016).
- Fix landed in `model.density_spice.project_monotone_sigma0_ctrl` / default `encode_a_from_sigma0_ctrl(..., monotone=True)`, wired into T1, proj-cost, and cache export targets.
- Re-run T1: Phase-3 gate **PASS** (T[>800] E/A=0.970). See `reports/t1_basis_stability.md` Big win section.

