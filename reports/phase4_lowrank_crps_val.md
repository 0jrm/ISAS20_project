# Phase 4 — full CRPS two-stage eval (4.8)

**Checkpoint:** `saved/argo_densityspice_lowrank_crps/models/NeSPReSO2_ARGO_GoM_densityspice_lowrank_crps_lowrank_crps_v1_s2/lowrank_crps_v1_s2/model_best.pth`  
**Cache:** `../data/cache/train_ready_0f6129b27ddb.pkl`  
**Split:** val n=621

**Anchors:** MISS (ENCE < 0.2: NO; Spearman ≫ 0.12: yes)

## Overall (standardized σ₀_ctrl + spice PCs)

| CRPS | ENCE | PIT sup-dev | spread-skill slope | σ–|err| Spearman |
|------|------|-------------|--------------------|-----------------|
| 0.5846 | 0.2857904222619385 | 0.03719806763285023 | 1.5082047968605936 | 0.4410839805469574 |

## By season

| season | CRPS | ENCE | PIT | slope | Spearman | n |
|--------|------|------|-----|-------|----------|---|
| DJF | 0.6003 | 0.3401286785419119 | 0.047011784511784505 | 1.5262381296103136 | 0.42264274275606234 | 23760 |
| MAM | 0.5701 | 0.306864066709536 | 0.028202160493827164 | 1.4918165239683168 | 0.45792479498521255 | 25920 |

## By depth band (density ctrl only) × season

### 0-50 m

| season | CRPS | ENCE | Spearman |
|--------|------|------|----------|
| DJF | 0.1236 | 0.18281535542215244 | -0.11747347418493954 |
| MAM | 0.1221 | 0.131217063499991 | 0.048793285693136984 |

### 50-200 m

| season | CRPS | ENCE | Spearman |
|--------|------|------|----------|
| DJF | 0.2255 | 1.5271790385105306 | -0.018815481779740826 |
| MAM | 0.2024 | 1.3963889023304392 | 0.10828416988386416 |

### 200-800 m

| season | CRPS | ENCE | Spearman |
|--------|------|------|----------|
| DJF | 0.2250 | 0.20917320778271345 | 0.19665082809812573 |
| MAM | 0.1738 | 0.24035496854612157 | 0.2484381517175002 |

### >800 m

| season | CRPS | ENCE | Spearman |
|--------|------|------|----------|
| DJF | 0.4653 | 0.506439190564481 | -0.015671650560967212 |
| MAM | 0.4498 | 0.6561272455875291 | -0.10668192750430026 |

## Physical (point μ after inversion)

- σ₀ profile rate: 0.4042
- N² profile rate: 0.2689
- MLD RMSE: 31.41897722951038
- dρ/dz RMSE: 0.005122248973892286

**Caveat:** No inputs_err / input-error tercile stratum (Phase 2.2 full HDF5 blocker). T2 stale gate OPEN. Formal product errors are relative indicators only.

## Anchor miss

- ENCE=0.2857904222619385 (need < 0.2)
