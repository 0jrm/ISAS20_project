# Phase 4 — full CRPS two-stage eval (4.8)

> **Informational only:** Phase 3 skill gate FAIL (overall T RMSE 0.72 vs argo16 0.42). Do not use ENCE as a dissertation headline until density skill recovers. **Spearman is the real pass** — keep it in the record.

**Checkpoint:** `saved/argo_densityspice_crps/models/NeSPReSO2_ARGO_GoM_densityspice_crps_phase4_crps_v2_s2/phase4_crps_v2_s2/model_best.pth`  
**Cache:** `../data/cache/train_ready_cd9e08b6c630.pkl`  
**Split:** test n=623

**Anchors:** ENCE MISS (<0.20); Spearman **PASS** (0.65 ≫ 0.12 session baseline) — RC-4 killer / DA-relevant ranking.

## Overall (standardized σ₀_ctrl + spice PCs)

| CRPS | ENCE | PIT sup-dev | spread-skill slope | σ–|err| Spearman |
|------|------|-------------|--------------------|-----------------|
| 1.1526 | 0.32703807277486163 | 0.04373996789727126 | 1.4406680951006945 | 0.6490726987399406 |

## By season

| season | CRPS | ENCE | PIT | slope | Spearman | n |
|--------|------|------|-----|-------|----------|---|
| DJF | 1.0152 | 0.39074485798918224 | 0.06204379562043795 | 1.106658510136377 | 0.6667903953480032 | 10960 |
| MAM | 0.9839 | 0.48950126379933623 | 0.17740384615384613 | 1.492554342864442 | 0.6347540417717349 | 4160 |
| JJA | 1.3011 | 0.32221375456017604 | 0.05795454545454545 | 1.720704151760605 | 0.6245899513538762 | 20240 |
| SON | 1.0975 | 0.2845236395408529 | 0.06802486187845304 | 1.2053446035167772 | 0.6215812559613205 | 14480 |

## By depth band (density ctrl only) × season

### 0-50 m

| season | CRPS | ENCE | Spearman |
|--------|------|------|----------|
| DJF | 0.3021 | 0.5898055144013247 | 0.2292566633560065 |
| MAM | 0.2703 | 0.7294651602116283 | 0.2634733128037894 |
| JJA | 0.4715 | 0.4422615174289172 | 0.541514081299516 |
| SON | 0.4247 | 0.4310424575536779 | 0.6759707989244753 |

### 50-200 m

| season | CRPS | ENCE | Spearman |
|--------|------|------|----------|
| DJF | 0.5164 | 0.43359022253487983 | 0.7426364931581405 |
| MAM | 0.4312 | 0.44216405370054135 | 0.4946883812816397 |
| JJA | 0.6880 | 0.23154894592016775 | 0.6966577577471129 |
| SON | 0.6698 | 0.20554119732400622 | 0.7506548052395259 |

### 200-800 m

| season | CRPS | ENCE | Spearman |
|--------|------|------|----------|
| DJF | 0.4461 | 0.4803493915493259 | 0.5189714426197634 |
| MAM | 0.3538 | 0.5936349510883961 | 0.43379785158661566 |
| JJA | 0.5432 | 0.34921943353566953 | 0.5026685212343567 |
| SON | 0.4453 | 0.43397086801231255 | 0.4556645025909683 |

### >800 m

| season | CRPS | ENCE | Spearman |
|--------|------|------|----------|
| DJF | 2.1755 | 0.5751328130294013 | 0.7829693177043765 |
| MAM | 1.8310 | 0.7571338505151703 | 0.6023995929350848 |
| JJA | 1.9450 | 0.6829030207958243 | 0.6534430692878404 |
| SON | 2.0004 | 0.6733287143620096 | 0.5447598429888509 |

## Physical (point μ after inversion)

- σ₀ profile rate: 0.0000
- N² profile rate: 0.0000
- MLD RMSE: 42.31368330748723
- dρ/dz RMSE: 0.00843032958999481

**Caveat:** No inputs_err / input-error tercile stratum (Phase 2.2 full HDF5 blocker). T2 stale gate OPEN. Formal product errors are relative indicators only.

## Anchor miss / pass

- ENCE=0.327 (need < 0.2) — absolute variance scale off on a near-climatology mean; unsurprising.
- Spearman=0.649 — model ranks which profiles it is uncertain about; do not bury this under the ENCE miss.
- **Deferred:** post-hoc scalar σ recalibration on val (prospectus-allowed) — polish only after mean recovers; then redo CRPS stage-2 → scalar calib → re-judge ENCE.
