# Session handoff — dissertation data foundation

**Branch:** `residual_cube` (merged → `master` @ `56f1e18`)  
**Updated:** 2026-07-17  
**Code home:** [`NeSPReSO2_onTemplate/`](NeSPReSO2_onTemplate/)  
**Conda:** `nespreso` (pinned: `reports/phase5_conda-env.lock.yml` / sha in `phase5_conda-env.sha256`)

---

## Status: Phase 5 matrix ready — B closed, env pinned, DA path wired

| Phase | Status |
|-------|--------|
| 0–1 | Done |
| 2 | v3 HDF5 regen **advancing** (`tmux attach -t v3_hdf5_regen`) |
| 3 | **PASS** — inference-isotonic claim |
| 4 | **PASS** — low-rank CRPS + per-dim α; test ENCE 0.160 / Spearman 0.540 |
| 5 | Prereg locked + R4 cleared + **B joint-EOF train path live** + §5.1 archive done + env pin |
| 6 | OSSE **prereg locked** (`reports/osse_preregistration.md`); Σ_T/Σ_S Jacobian export in `dacov` |

---

## Architecture claim (do not silent)

Winning path is **σ₀-space** low-rank: `σ̂₀ = clim + scores @ V`.  
**Not** monotone in the head. Pre-isotonic test: **45.6%** profiles violate.  
**Guarantee:** *stable by construction at inference* via mandatory isotonic
(`project_monotone_sigma0_ctrl`) — §3.6 opt-2 / T1-D. Cost ~0.0015 °C / ΔT ≈ −0.0002 °C.  
Prob σ lives on scores; cov export `Σ_ρ = V diag((α σ_z)²) Vᵀ` with val-fitted α;  
**DA R:** `dacov.ts_covariance_from_sigma0_spice` / `export_ts_covariance_lowrank` → `Σ_T`/`Σ_S` (Phase 6 E4/E5).  
**Calibration:** val-fitted per-dim σ scales are part of the inference recipe (not optional polish).

---

## Headline numbers

### Phase 3 μ (deterministic)

| quantity | value |
|----------|------:|
| chrono T RMSE (spice_v3) | **0.562** |
| same-split floor (×1.10) | **0.590** |
| ratio | 1.047 |

### Phase 4 low-rank CRPS — headline (`lowrank_crps_v1_s2b` + per_dim)

| quantity | val per_dim | test per_dim |
|----------|------------:|-------------:|
| CRPS | 0.577 | **0.698** |
| ENCE | 0.058 | **0.160** |
| Spearman | 0.528 | **0.540** |
| slope | 1.36 | 1.61 |

Anchors: ENCE < 0.20 **PASS**; Spearman ≫ 0.12 **PASS**.  
Val→test ENCE gap (~3×) is prospectus §3.6.6 — winner summary must include **depth × season** strata.

**Ckpt:** `saved/.../lowrank_crps_v1_s2b/model_best.pth`  
**Cache:** `../data/cache/train_ready_0f6129b27ddb.pkl`  
**Reports:** [`reports/phase4_ence_recalib_s2b.md`](reports/phase4_ence_recalib_s2b.md)

---

## Record (hygiene)

| doc | role |
|-----|------|
| [`reports/ablation_preregistration.md`](reports/ablation_preregistration.md) | Phase 5 lock + winner rule + DA critical path |
| [`reports/osse_preregistration.md`](reports/osse_preregistration.md) | Phase 6 E0–E5 locked pre-winner |
| [`reports/phase5_conda-env.lock.yml`](reports/phase5_conda-env.lock.yml) | env pin (launcher asserts sha) |
| [`config/archive/`](NeSPReSO2_onTemplate/config/archive/) | §5.1 kill list (before results) |
| eval hygiene | fit scales on val; one test score per frozen cell; report **all** cells |

---

## Next

1. **Launch matrix** — `scripts/launch_matrix.py --launch --only C,CRPS` (env hash asserted).
2. Parallel CPU filler: finish ISOP/MODAS cache-backed fit; Jacobian MC agreement on real profiles.
3. Error-channel axis only after v3 HDF5; month-clim deferred.
4. After matrix: **mechanical** winner (ENCE<0.20 → best CRPS → dρ/dz); full tables for all cells + winner depth×season strata → toy DA chapter.

```bash
tmux attach -t v3_hdf5_regen
cd NeSPReSO2_onTemplate
srun --ntasks=1 --cpus-per-task=2 conda run -n nespreso \
  python3 scripts/launch_matrix.py --selfcheck
srun --ntasks=1 --cpus-per-task=2 conda run -n nespreso \
  python3 scripts/launch_matrix.py --dry-run --only C,CRPS
# then --launch
```
