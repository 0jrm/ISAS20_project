# Phase 5 — Ablation summary (physical-space ruler)

**Status:** physical-space matrix complete 2026-07-17 (`p5phys` exit=0)  
**Prereg:** [`ablation_preregistration.md`](ablation_preregistration.md) §3 (amended 2026-07-17)  
**Artifacts:** `NeSPReSO2_onTemplate/saved/runs/phase5_matrix/eval/physical/`  
**Rollup:** `matrix_physical_summary.json` (rebuilt `--aggregate-only` from all 9×3 seed JSONs)

Protocol: chronological split; seeds {42,43,44}; stage-2 val-ENCE early-stop; val-only σ recalib; **one** latent test score then **one** physical-space rescoring consultation. Judgment = **cell mean±std** on physical metrics after decode (M=100). A/B **mean-path** isotonic reported; member-wise `--iso-ensemble` only where already present on all seeds (A×\*, B×CRPS).

Latent-space table retained as diagnostics only — see prior revision / `reports/phase5_*.md`. **Do not rank cross-rep on latent CRPS.**

---

## 1. Full physical-space table (mean±std)

| cell | T RMSE | phys CRPS (T+S) | phys ENCE | Sp(T) | σ₀ viol (prof) | mean-path iso ΔT | post-iso σ₀ |
|------|--------|-----------------|-----------|-------|----------------|------------------|---------------|
| A×CRPS | 0.559±0.005 | **0.119±0.001** | **0.153±0.007** | 0.756 | 0.614±0.077 | −0.0001 | 0.304±0.021 |
| A×NLL | 0.575±0.023 | 0.122±0.005 | **0.162±0.019** | 0.768 | 0.531±0.111 | −0.0000 | 0.228±0.099 |
| A×det | **0.541±0.004** | — | — | — | 0.408±0.126 | −0.0000 | 0.218±0.014 |
| B×CRPS | 0.586±0.054 | 0.133±0.009 | 0.247±0.003 | 0.778 | 0.315±0.109 | −0.0001 | 0.147±0.094 |
| B×NLL | 0.563±0.011 | 0.128±0.002 | 0.299±0.013 | 0.767 | 0.319±0.020 | −0.0000 | 0.114±0.038 |
| B×det | **0.534±0.001** | — | — | — | 0.433±0.086 | −0.0001 | 0.266±0.061 |
| C×CRPS | 0.618±0.103 | 0.139±0.022 | 0.384±0.010 | 0.757 | 0.499±0.029 | — | — |
| C×NLL | 0.694±0.081 | 0.157±0.018 | 0.397±0.011 | 0.784 | 0.516±0.035 | — | — |
| C×det | 0.609±0.012 | — | — | — | 0.190±0.068 | — | — |

Skill floor: T ≤ **0.5903**. Phys ENCE gate: &lt; **0.20**.

**Labeled:** C×det miss matches [`finding_C_det_gate_overfit.md`](finding_C_det_gate_overfit.md). Latent C×NLL ENCE pass (0.120) **does not** survive physical ENCE (0.397).

B×NLL: iso-ensemble aggregate skipped (1/3 seeds had `isotonic.prob` mix).

---

## 2. §3 decision rule — applied as written (physical)

**Step 1 survivors**

| class | cells |
|-------|-------|
| Prob (phys ENCE &lt; 0.20) | **A×CRPS**, **A×NLL** |
| Det (T ≤ 0.5903) | **A×det**, **B×det** |
| Out | B×CRPS, B×NLL, C×CRPS, C×NLL, C×det |

**Step 2 (lowest phys ensemble CRPS among prob survivors):** **A×CRPS** (0.119 &lt; A×NLL 0.122).

**Step 3–4:** dρ/dz not yet in this rollup (deferred strata). Spearman: A×CRPS **0.756 ≫ 0.12** — DA-ranking claim allowed.

**Stability:** A×CRPS raw σ₀ viol high (~0.61); mean-path iso ΔT ≈ 0 with post-iso σ₀ ~0.30 (not ≡0 — ensemble-mean path). Member-wise iso CRPS already on disk for A×CRPS: 0.119±0.001 (ENCE 0.157) — essentially unchanged vs raw ensemble.

**§3.1 no-survivors fallback:** not triggered (two prob survivors).

**Mechanical dissertation default:** **A×CRPS**.

---

## 3. Chapter anchors (non-conflicting labels)

| role | cell | why |
|------|------|-----|
| **Default / UQ + DA R** | **A×CRPS** | §3 mechanical winner; best phys CRPS among ENCE survivors; Sp 0.756 |
| Best skill-floor det | **B×det** | lowest T 0.534; not the UQ default (no Σ path) |
| Stability / dens–spice | — | no C cell clears phys ENCE or (for det) skill floor under this ruler |

---

## 4. Next / strata

**Winner strata (done):** [`phase5_A_CRPS_physical_strata.md`](phase5_A_CRPS_physical_strata.md)

| readout | value |
|---------|-------|
| Overall phys CRPS | 0.119±0.001 (matches matrix) |
| Overall ENCE(**T only**) | 0.236±0.005 — higher than matrix `ence_mean_TS` 0.153 (T+S mean); T is the hard band |
| Best season CRPS | MAM 0.103; worst JJA 0.126 |
| Worst season ENCE(T) | MAM 0.343 (small n=52) |
| Depth | CRPS peaks 50–200 m; ENCE(T) worst in 0–50 / JJA (0.66) |

**Still open**

1. Phase 6: wire `run_osse.py` truth/cast loop; E3–E5 = A×CRPS + `export_ts_covariance_pca`.
2. Optional clean `--iso-ensemble` on A×CRPS.
3. Quantile + error-channel gated on v3 HDF5.

Env pin: `saved/runs/phase5_matrix/conda-env.sha256` = `621d0d65…`.
