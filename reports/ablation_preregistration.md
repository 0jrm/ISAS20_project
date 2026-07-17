# Phase 5 — Ablation matrix preregistration

**Status:** PRE-REGISTERED 2026-07-16; **R4 CLASSIFIED + GOLDENS RESTORED** (same day)  
**Plan:** [`PLAN-v2-recovery.md`](../PLAN-v2-recovery.md) §5.2  
**Branch:** `residual_cube`  
**Gate:** do not launch matrix cells until this file exists in git history ahead of the first matrix commit (timestamp check).

---

## 0. Frozen context (do not relitigate)

| pin | value |
|-----|-------|
| Split | chronological (`train_frac/val_frac/test_frac` 0.7/0.15/0.15, seed 42) |
| Cache | `../data/cache/train_ready_0f6129b27ddb.pkl` (or rebuild with identical `config_hash` inputs) |
| Backbone | `PatchConvMLP` d_model=128, head `[1024,1024]`, dropout 0.2 |
| Domain | GoM ARGO/CORA targets; L4 not hidden truth |
| Eval | frozen `evalphys` only; headline metrics via reference `gsw` |
| Phase 3 μ winner | low-rank δσ₀ (R=16) + spice; inference isotonic mandatory |
| Phase 4 σ recipe | two-stage CRPS; **val-fitted per-dim σ scales** are part of inference (`sigma_recalib_per_dim.json`) |
| Skill floor | same-split chrono argo16 raw × 1.10 = **0.5903** (see `gate_floor_provenance.md`) |

**Claim split (representation chapter):**

- **Hard-head monotone:** full-rank softplus+cumsum on ctrl increments (appendix only; not a matrix cell).
- **Inference-stable (matrix C):** low-rank δσ₀ + `project_monotone_sigma0_ctrl` at inference (Phase 3/4 winner).  
  Do not conflate these in matrix readouts.

---

## 1. Matrix (locked)

```
representation ∈ {
  A: separate T/S PCA-16 (chrono argo16 protocol),
  B: joint T/S EOF-32 (T1 variant B),
  C: density_spice — low-rank δσ₀ R=16 + spice-16 + inference isotonic
}
head ∈ { det-MSE, CRPS, NLL-β }
```

| cell count | formula |
|------------|---------|
| Core | 3 × 3 = **9** cells × **3 seeds** = **27** runs |
| Quantile | only on winning (representation, head) after core — **+3** seeds |
| Error-channels | `on`/`off` only on winning (representation, head) — **+2** (**gated on Phase 2.2 v3 HDF5**; skip with explicit note if still blocked) |
| **Total** | ≤ **32** GPU runs |

### C definition (locked to Phase 3/4 winner — not the original plan wording)

Original plan C was in-head monotone softplus-cumsum. **Current C** is:

- `io.representation=density_spice`, `io.lowrank_target=sigma0`
- `outputs.density_ctrl=16`, `outputs.spice=16`
- Head predicts low-rank scores; decode `σ̂₀ = clim + scores @ V`
- **Inference path includes** mandatory isotonic (`project_monotone_sigma0_ctrl`) — cost ~**0.0015 °C** (argo16 gate) / ΔT ≈ −0.0002 °C on the low-rank winner
- Prob σ on scores; cov export `Σ_ρ = V diag((α σ_z)²) Vᵀ` with val-fitted α

Full-rank softplus+cumsum is **not** a matrix cell; it remains the hard-head claim appendix.

### Identical across every probabilistic cell

- Data, split, backbone width, epoch/early-stop policy, loss-scale derivation procedure
- Seed set `{42, 43, 44}`
- **Two-stage CRPS/NLL schedule:** stage-1 μ MSE (σ frozen) → stage-2 unfreeze σ, μ LR × 0.1; epoch budget matches Phase 4 s2b (stage-2 continue allowed if val CRPS still falling)
- **Val-fit per-dim σ recalibration** (`ence_recalib_val.py` → `sigma_recalib_per_dim.json`) applied identically before CRPS/ENCE/PIT/spread-skill and before §4.4 Σ export

**Det-MSE heads:** no σ branch; calibration metrics marked N/A (not FAIL).

### Eval hygiene (matrix rule — explicit)

- All model selection, early-stop, and σ recalibration on **val only**
- Test scored **exactly once per cell** after recipe frozen (ckpt + α + isotonic path)
- Dissertation number = **3-seed mean ± std** — never a single-seed cherry-pick

### Run order

1. **C × CRPS × seeds {42,43,44} first** (headline cell lands early)
2. Remaining C heads, then A, then B
3. Error-channel `on`/`off` pair **only after** v3 HDF5 finishes (confirm `tmux attach -t v3_hdf5_regen` still advancing before GPU matrix saturates)

---

## 2. Pre-registered readouts

Every cell × seed emits evalphys JSON; aggregate = mean ± std over seeds.

| # | readout | notes |
|---|---------|-------|
| 1 | σ₀ / N² violation rates | A,B expected > 0; C pre-isotonic reported + post-isotonic ≡ 0 by construction |
| 2 | T/S RMSE by depth band | native levels; overall T for skill-floor compare |
| 3 | CRPS + ENCE + Spearman | prob heads only; ENCE uses val per-dim σ scales then **one** test score |
| 4 | MLD & D26 RMSE | coverage reported |
| 5 | dρ/dz RMSE | overall + by band |

**Strata (when available):** depth band × season; input-error tercile only if `inputs_err` present (else skip, never fake).

---

## 3. Decision rule (dissertation default model)

1. Restrict to cells with **ENCE < 0.20** on chronological test (prob heads) **or** det-MSE cells that clear the Phase 3 skill floor (T RMSE ≤ 0.5903) with reported stability cost.
2. Among survivors: **lowest CRPS** (prob) / lowest T RMSE (if comparing to a lone det survivor).
3. Ties: lower **dρ/dz RMSE**.
4. Secondary: Spearman ≫ 0.12 required for any DA-ranking claim.

**Mechanical pick:** apply the rule above; do not hand-edit the winner.
**Report all cells:** `reports/ablation_summary.md` must include the **full evalphys table for every cell** (mean±std over seeds) — losing cells are the comparative-architecture chapter, not footnotes.
**Pre-committed stratified readout:** the winner's calibration table **depth band × season** is reported alongside the headline CRPS/ENCE/Spearman (so the Phase 4 val→test shift finding carries into final numbers, not only an s2b footnote).

---

## 4. Launch blockers — R4 classification (not a waive)

| ID | blocker | resolution |
|----|---------|------------|
| R4 | `test_combined_pca_loss_v2` combined/wmse golden | **CLASSIFIED → regenerate** (below) |
| P2.2 | error-channel axis | defer `on`/`off` until v3 HDF5 lands; core 27 may proceed |
| Scales | per-dim α persistence | required for every CRPS/NLL cell (`sigma_recalib_per_dim.json` + `dacov` α kwarg) |

### R4 outcome (2026-07-16) — classification, not waive

**Human ACK:** conditional — classify first; regenerate only if env/procedural; block if semantic.

| check | result |
|-------|--------|
| Parent worktree `820e598` live | combined **0.008507695**, wmse **0.000430556** |
| HEAD live (same fixture, seed 42) | **identical** floats |
| Wrong GOLDEN dict (since `3699887`) | combined 0.050719 / wmse 0.002583 |
| Ratio wrong/live | **exactly 6.0** (= `2 × n_components`) |
| Cause | Bad regen at `3699887`: wmse recorded **without** `weights/sum(weights)` normalization in `genWeightedMSELoss`. Loss math unchanged; PCA recon + `pca_loss` always matched. |
| Class | **Procedural golden corruption** (not semantic CombinedPCALoss drift; not 1e-5 env numerics) |

**Action taken:** restore GOLDEN to original `b34efc8` values; hard-assert again in `selfcheck.py`.  
**Sign-off:** this human ACK message (conditional yes → classify → regenerate).  
**Matrix:** unblocked for launch after this file + restored goldens are in the working tree ahead of the first matrix job.

---

## 5. Artifacts

| path | role |
|------|------|
| this file | preregistration (must precede launch) |
| `saved/runs/phase5_matrix/manifest.json` | per-cell config, seed, ckpt, eval paths |
| `saved/runs/phase5_matrix/conda-env.lock.yml` + `.sha256` | env pin (mirrored in `reports/phase5_conda-env.*`) |
| `scripts/launch_matrix.py` | launcher; **asserts env hash before every run** |
| `config/argo/config_argo_joint_eof.json` | matrix B template (`io.representation=joint_eof`) |
| `config/archive/` | §5.1 kill list (archived **before** results) |
| `scripts/results_table.py` (extend) | evalphys mean±std table — **all cells** |
| `reports/ablation_summary.md` | one-page interpretation vs §3 rules + winner strata |

**Winning Phase 4 reference (not a matrix seed):**  
`saved/.../lowrank_crps_v1_s2b/model_best.pth` + `sigma_recalib_per_dim.json` — test ENCE 0.160 / Spearman 0.540 / CRPS 0.698.

### Critical path for Phase 6 (parallel to matrix GPU)

| item | why gate-critical |
|------|-------------------|
| `dacov` Σ → T/S Jacobian export | E4/E5 consume `Σ_T`/`Σ_S`; without it the DA chapter degrades to R_fixed-only |
| `reports/osse_preregistration.md` | lock E-table before winner known |
| `scripts/isop_modas_baseline.py` | E2 baseline; CPU-side |

---

## 6. Explicit non-goals

- No architecture search (transformer / diffusion / GAN).
- No random-split dissertation numbers.
- No cross-tag raw RMSE (use `eval_matched.py` only if ISAS appendix needed).
- No silent L4-as-truth.
- No burning test scores while tuning α — fit on val only; **one test score per frozen cell**.
