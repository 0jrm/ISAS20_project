# HANDOFF — Cube-native point anchor (Option B)

**Doc:** HANDOFF-2026-07-05-cube-native-point-anchor  
**Plan:** `/unity/g2/jmiranda/.cursor/plans/cube-native_point_anchor_f1fc2cf1.plan.md`  
**Design context:** `HANDOFF_residual-cube.md`  
**Repo root:** `/unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project/NeSPReSO2_onTemplate`  
**Conda env:** `nespreso` (`/conda/jmiranda/miniconda/envs/nespreso/bin/python`)  
**Date:** 2026-07-05

---

## 1. Executive summary

The **cube-native anchor program (Option B)** is largely implemented. The cube was rebuilt (`DATA_REVISION=2`), caches regenerated, **`point_cube`** trained, and M6 interpret eval re-run with **fixed S3a/S3b inputs**.

> ### ⚠️ 2026-07-05 CORRECTION — read before trusting the numbers below
> A provenance/eval audit (see new **§11**) overturned three results that drove the branch decision:
> 1. **The golden "0.416" is a *random-split* number**, not the split this program uses. Scored on the **chronological** test split with the *same* checkpoint and the *same* evaluator, golden = **0.514** (T), 0.083 (S). The evaluator is **not** buggy and is self-consistent (`eval_run.py` and `eval_residual_cube.py` agree at 0.514). Regression test: `tests/test_golden_reproduction.py -m golden_repro`.
> 2. **The apples-to-apples gap is ~0.063, not 0.161.** On the chronological split: golden **0.514** vs point_cube **0.577**. The "CMEMS/OSTIA products are much weaker" framing was based on comparing a random-split golden (0.416) against a chronological-split cube model (0.577). **The contingency branch is not yet justified.**
> 3. **`residual_v1` = 0.631 is a stale artifact.** That checkpoint (`model_best.pth`, mtime 07-04 22:59) was **never retrained** against the new anchor/cache: its embedded config records `cache_path=train_ready_7b9094413d13.pkl`, `warmstart_ckpt=argo16_scales` (golden, not point_cube), and `center_{sss,sst,ssh}=true` (the old centering config removed this session). The 0.631 == prior-run 0.631 identity is the same model re-scored. Any residual-vs-anchor comparison is invalid until it is retrained with the current `config_argo_residual_cube.json`.

**Branch decision (plan §5): SUSPENDED pending §11.** Prior text: `point_cube` T=0.577 ≥ 0.55 → contingency. But the correct chronological-split reference is golden **0.514**, so the cube-native floor trails the legacy pipeline by only ~0.06 on equal footing. Run the cheap experiments (centered-SSH `point_cube_v2`, lr/bs sweep) before deciding on SMAP/AVISO/MUR ingestion.

**Headline test RMSE — CORRECTED (chronological test, n=623):**

| Model | T RMSE | S RMSE | Notes |
|-------|--------|--------|-------|
| golden (`argo16_scales`) — **chronological** | **0.514** | **0.083** | apples-to-apples reference for cube models |
| golden (`argo16_scales`) — random split | 0.416 | 0.072 | historical "program target"; **wrong split** for comparison |
| **point_cube** (cube anchor) | **0.577** | **0.089** | new S0 reference; trails golden by ~0.06 on equal footing |
| ~~residual_v1~~ (0.631) | — | — | **STALE checkpoint, never retrained; ignore until re-run** |

---

## 2. Plan todo status

| Plan id | Status | Notes |
|---------|--------|-------|
| forward-fill | **Done** | Trailing forward-fill removed; whitelist-or-fail |
| av5-mask | **Done** | Per-product ocean mask in A-V5 |
| decode-bug | **Done** | `set_auto_maskandscale(False)` before slice; SSS headers are sf=1 (no value change) |
| sampler-missing | **Done** | `MissingCubePlaneError` + test |
| data-revision | **Done** | `DATA_REVISION=2` in `cube_schema.py`, participates in cache hash |
| cube-rebuild | **Done** | Validated; SSS `2022-03-01` whitelisted (no `SSS_20220301.nc`) |
| center-flags | **Done** | Removed from `config_argo_residual_cube.json` |
| point-cube-cache | **Done** | `train_ready_308c9ec95f5a.pkl` (9-D, z-scored sats) |
| point-cube-train | **Done** | Early-stop epoch 528; best val_loss=0.135 |
| point-cube-branch | **Reopened** | T=0.577 vs chronological golden **0.514** (not 0.416); gap ~0.06 → run cheap experiments first (§11.4) |
| s0b-test | **Done** | `test_s0b_residual_init_matches_point_cube` passes |
| residual-recache | **Done** | `train_ready_2ab55b15b14f.pkl` (41-D) |
| m6-rerun | **Partial** | Test + interpret + feature importance + S3a/S3b; **stratification / depth RMSE not implemented** |
| m7-rerun | **In progress** | Fresh ablation run started with fixed script; see §4 |
| m2-script | **Done** | `scripts/m2_spotcheck.py` → `saved/results/m2_spotcheck.json` |

**Not done from plan §4:** lr/batch sanity sweep for `point_cube` (single default recipe used).

---

## 3. Key artifacts

### Cube
- Path: `data/cube/gom_cube.zarr`
- `DATA_REVISION=2`, `data_through=2022-03-01`
- `ALLOWED_MISSING_DAYS`: `["2022-03-01"]` (SSS archive gap; pad downloaded Mar 2+ is outside `TIME_END`)
- Validation: `data/cube/validation_report.json` passes
- Corrupt SSH yearly files outside 2015–2022 skipped in `build_cube.py`

### Caches (do not mix hashes)
| Cache | Hash | D | Purpose |
|-------|------|---|---------|
| point | `308c9ec95f5a` | 9 | `point_cube` training |
| residual | `2ab55b15b14f` | 41 | `residual_v1` + ablations base |

### Checkpoints
| Run | Path |
|-----|------|
| point_cube | `saved/models/NeSPReSO2_ARGO_GoM/point_cube/model_best.pth` |
| residual_v1 | `saved/models/NeSPReSO2_ARGO_GoM_residual_cube/residual_v1/model_best.pth` |
| golden (PCA only) | `saved/models/NeSPReSO2_ARGO_GoM/argo16_scales/model_best.pth` |

### Eval outputs
| File | Content |
|------|---------|
| `saved/eval_point_cube_test.json` | Anchor test RMSE |
| `saved/eval_residual_v1_test.json` | Residual test RMSE |
| `saved/eval_residual_v1_interpret.json` | S3a/S3b, feature importance, gates (**fixed eval**) |
| `saved/results/m2_spotcheck.json` | Cross-source bias/slope/corr vs v2 pickle |

---

## 4. Active / stale jobs

```bash
# What's running
pgrep -af "run_residual_ablation|train.py"

# Ablation log (authoritative for M7)
tail -f /tmp/residual_ablations.log

# Stale — do not treat as live
tail -f /tmp/cube_native_continue.log
```

**M7 ablation script** (`scripts/run_residual_ablations.sh`, fixed 2026-07-05):
- `RETRAIN=1`: `rm -rf` old model dirs before train (fixes `FileExistsError`)
- Uses `--golden-ckpt` in interpret eval
- Writes `saved/results/residual_cube_ablations/ablation_summary.json` at end
- Batch size 128, local GPU (`CUDA_VISIBLE_DEVICES=0`), no Slurm

Partial/old ablation JSON in `saved/results/residual_cube_ablations/` from pre-fix runs — **will be overwritten** when current run completes.

---

## 5. Bugs fixed this session (read before trusting old logs)

1. **`eval_residual_cube.py` S3a bug (critical):** Was feeding `point_cube` the **raw** first 9 cols of the 41-D residual cache → bogus point T RMSE ~7.25. **Fixed:** load z-scored inputs from 9-D point cache (`build_point_cube_cache`). S3b golden now uses v2 `config_argo.json` cache.
2. **`sync_arch_with_io`:** `cache_kind=point_cube` → `input_dim=9` (was 0, broke training).
3. **`run_cube_native_continue.sh`:** syntax error on empty `POINT_TRAIN_PID=` (fixed); script resumed successfully then killed on ablation restart.
4. **A-V2 endpoint:** whitelisted missing SSS day passes via manifest lookup.
5. **Ablation `FileExistsError`:** retrain now deletes old save dirs.

---

## 6. Known issues / tech debt

| Issue | Severity | Action |
|-------|----------|--------|
| sklearn PCA 1.2 vs 1.5 warning | Low | Harmless for targets; optional refit |
| Log lines doubled in continue log | Cosmetic | Remove `>>` append or drop inner `tee` |
| `residual_v1` trained before `point_cube` finished | Low | Same `model_best.pth` (best val early); OK for S0b |
| M6 stratification vs \|∇SSH\| not implemented | Medium | Add to `eval_residual_cube.py` per design spec |
| `point_cube` no lr/bs sweep | Medium | Plan §4 optional sweep before declaring anchor final |
| M2 spotcheck SST +22°C bias vs v2 | **Medium (misdiagnosed)** | NOT Kelvin/°C convention — it is the **double-decode** bug (§11.5). Model-neutral under z-scoring but rebuild cube for physical correctness |
| Cube SST/SSH double-decoded on disk | **Medium** | §11.5: stale pre-fix cube; rebuild + harden A-V4 + record build provenance. Does not change model numbers (affine + z-score) |
| Contingency branch (T=0.577) | **Program** | Add SMAP/AVISO/MUR channels to cube or accept anchor |

---

## 7. Commands for a fresh agent

```bash
cd /unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project/NeSPReSO2_onTemplate
PY=/conda/jmiranda/miniconda/envs/nespreso/bin/python
export CUDA_VISIBLE_DEVICES=0

# Verify S0b
$PY -m pytest tests/test_residual_init.py -m s0b_gate -q

# Re-run M6 interpret (fast)
$PY diagnostics/residual_cube/eval_residual_cube.py \
  -c config/argo/config_argo_residual_cube.json \
  -r saved/models/NeSPReSO2_ARGO_GoM_residual_cube/residual_v1/model_best.pth \
  --point-ckpt saved/models/NeSPReSO2_ARGO_GoM/point_cube/model_best.pth \
  --golden-ckpt saved/models/NeSPReSO2_ARGO_GoM/argo16_scales/model_best.pth \
  --split test --out saved/eval_residual_v1_interpret.json

# Finish / restart M7 ablations
nohup bash scripts/run_residual_ablations.sh > /tmp/residual_ablations.log 2>&1 &

# Full rebuild path (only if cube content changes — bump DATA_REVISION first)
$PY preproc/cube/build_cube.py --product all --workers 8 --resume
$PY preproc/features/export_feature_cache.py -c config/argo/config_argo_point_cube.json --point-only --force
$PY preproc/features/export_feature_cache.py -c config/argo/config_argo_residual_cube.json --force
```

**Training:** use `GPU_MODE=local` or direct `python train.py` — **no Slurm**. VRAM budget ~6 GB: `point_cube` bs=256, residual bs=128.

---

## 8. Recommended next steps (priority order — revised per §11)

1. **DONE this session:** provenance (§11.1), golden-repro test (§11.2), scoreboard correction.
2. **Retrain `residual_v1`** with current `config_argo_residual_cube.json` (warmstart `point_cube`, no center flags, cache `2ab55b15b14f`) — without this, anomaly 2 is not a real comparison. Then re-run M6 interpret.
3. ~~Centered-SSH `point_cube_v2`~~ **DONE (§11.4): did not help** (0.579 vs 0.577). Redundant with day-of-year harmonics. Do not pursue centered SSH further for the point model.
4. **Mandated `point_cube` lr/bs sweep** (plan §4, not done) + **per-product NaN/imputation fraction report** from `DATA_REVISION=2` validation — required before declaring any anchor final (esp. GoM CMEMS SSS imputation).
5. **Only if** centered SSH + sweep still leave a large gap to golden 0.514: ingest v2 SMAP/AVISO/MUR into the cube (`PRODUCT_SPECS` extension) and re-anchor.
6. **Implement M6 gaps:** depth-resolved RMSE, stratification by \|∇SSH\|/\|∇²SSH\| in `eval_residual_cube.py`. Let M7 ablations finish but **do not interpret** until items 2–3 clear (they inherit the stale-residual / split issues).
7. **Commit** code changes when ready — nothing committed during this agent session per user rules.

---

## 9. File index (code touched)

| Area | Files |
|------|-------|
| Cube build | `preproc/cube/build_cube.py`, `cube_schema.py`, `validate_cube.py` |
| Features | `preproc/features/sampler.py`, `export_feature_cache.py` |
| Training glue | `preproc/l3_input.py`, `parse_config.py` (unchanged) |
| Configs | `config/argo/config_argo_point_cube.json`, `config_argo_residual_cube.json`, ablations/*.json |
| Tests | `tests/test_sampler.py`, `tests/test_residual_init.py`, `tests/test_golden_reproduction.py` (new, `-m golden_repro`), `tests/conftest.py` |
| Eval | `diagnostics/residual_cube/eval_residual_cube.py` |
| Scripts | `scripts/run_cube_native_continue.sh`, `run_point_cube_pipeline.sh`, `run_residual_cube_pipeline.sh`, `run_residual_ablations.sh`, `m2_spotcheck.py` |

---

## 10. Interpretation guardrails

- **Split discipline (new, critical):** every headline comparison must be on the **same split**. The program uses **chronological**. The 0.416 golden is **random-split** — never compare it to chronological cube numbers. The chronological-split golden reference is **0.514** (`tests/test_golden_reproduction.py`).
- **Do not** compare residual to golden using residual-cache base inputs for golden or point — use fixed eval paths.
- **Do not** reuse pre-`DATA_REVISION=2` caches or pre-fix M6 JSON (point RMSE 7.25 era).
- **Do not** trust `residual_v1` (0.631) until retrained; the on-disk checkpoint is a pre-fix artifact (see §11.1).
- **S3a** is paired apples-to-apples on cube products; **S3b** is cross-pipeline diagnostic only — and its golden number (0.514) equals `eval_run.py` chronological golden, i.e. the evaluator is consistent.
- The program target on this chronological test split is a single scoreboard: beat golden's **0.514**, not two parallel scoreboards.
- Pickle-equivalence as M2 gate is **deprecated**; `point_cube` + `m2_spotcheck.py` replace it.

---

## 11. Provenance & eval audit (2026-07-05)

### 11.1 Anomaly 1 — `residual_v1` = 0.631 is a stale artifact (RESOLVED)
Checkpoint `saved/models/.../residual_cube/residual_v1/model_best.pth` mtime **2026-07-04 22:59**, which predates the new anchor (`point_cube` 07-05 11:18) and the residual cache `2ab55b15b14f` (07-05 11:20). Its **embedded config** records:
- `data_loader.cache_path = train_ready_7b9094413d13.pkl` (not `2ab55b15b14f`),
- `arch.warmstart_ckpt = .../argo16_scales/model_best.pth` (golden, not `point_cube`),
- `input_params.center_{sss,sst,ssh} = true` (the centering config removed this session).

Conclusion: never retrained. The 0.631 == prior-run 0.631 is the same model re-scored. **Action:** retrain with current `config_argo_residual_cube.json` (warmstart `point_cube`, no center flags, cache `2ab55b15b14f`) before any residual-vs-anchor claim.

### 11.2 Anomaly 3 — evaluator reproduces golden; 0.416 is a random-split number (RESOLVED)
Verified with the same `eval_run.py`, same checkpoint, same argo_v2 inputs:
| Checkpoint | Split | raw_profile_rmse T | S |
|---|---|---|---|
| `argo16_scales` | **random** (seed 42) | **0.4158** | 0.0722 |
| `argo16_scales` | **chronological** | **0.5143** | 0.0834 |

The historical "0.416" came from `notebooks/compare_outputs/results.json` (`argo_pca16`, `nb_configs.py` uses `split_mode: random`). Cross-check: `eval_residual_cube.py` S3b golden = 0.51430 == `eval_run.py` chronological golden 0.51430 → **the two evaluators agree**; there is no eval bug of the 7.25 class here. Confirmed non-issues (identical between golden argo_v2 cache and cube cache): PCA basis (`max|Δcomp|=0`), ground-truth profiles (`max|Δ|=0`), 1801-level depth grid, no `bottom_depth`/`clim_profiles` in either → no depth-mask/clim divergence.

**Regression guard:** `tests/test_golden_reproduction.py -m golden_repro` (3 tests): pins random→0.416, chronological→0.514, and that the split is the whole explanation.

### 11.3 Anomaly 2 — residual "worse than anchor" (EXPLAINED by 11.1)
It is not a real comparison: a golden-warmstarted, centering-config, old-cache checkpoint was scored against the new `point_cube`. This is the user's option (c). Re-open only after 11.1 retrain, then report **val-split** residual-vs-point plus the `gate_l1` trajectory; if residual wins val but loses test → chronological val→test shift; if it loses both → check checkpoint monitor (`val_loss` PC-MSE vs profile RMSE) and consider monitoring val `profile_rmse` and/or an L1 gate penalty.

### 11.5 Cube SST/SSH are double-decoded (correctness bug, but MODEL-NEUTRAL)
The on-disk cube stores **physically wrong SST and SSH**: SST ≈ 3 °C (std ~0.01) and SSH ≈ 5e-5 m (std ~2e-5) at all times; SSS is correct (35 ± 1.5). Cause: the cube chunks were written **2026-07-04 20:25–21:23**, *before* `build_cube.py`'s decode fix (mtime **07-05 10:52**) — the same "artifact not regenerated after the code fix" pattern as §11.1. The pre-fix build applied `scale_factor`/`add_offset` **twice**. SSS survived (`scale_factor=1`, no offset). Verified against raw archives:
- OSTIA `analysed_sst` `int16 scale=0.01 offset=273.15`: correct GoM plane = 23.3 °C (std 3.2); cube = `0.01·true + 2.73149`, **corr = 1.000000**.
- CMEMS `adt` `int32 scale=0.0001` (no offset): correct = 0.42 m (std 0.22); cube = `1e-4 · true`.
- The **current** `build_cube.py` produces correct planes (23.3 °C, 0.42 m) — only the stored cube is stale.

**Why it does not change any model number:** both corruptions are **affine** (`a·x + b`, `a>0`), every feature (`value`, `grad`, `laplacian`, `tendency`, `geo_uv`, `basin_*`, `value_centered`) is linear/affine in the field, and all are **train-split z-scored**. Since `z(a·x+b) ≡ z(x)`, the model inputs are identical to a correctly-decoded cube. So point_cube 0.577 and the branch numbers stand; this bug is orthogonal to the split issue (§11.2). Centered-SSH is likewise valid on the current cube (the per-sample basin subtraction survives z-scoring).

**Still must fix (correctness/provenance):** (1) rebuild the cube with current code and re-validate; (2) **harden A-V4** — its loose bounds (SST −3…35, SSH −1.5…1.5) passed the degenerate fields; add a spatial-variance floor / expected-mean check and a raw-vs-decoded spot check; (3) record `build_date`/`build_git_sha` (currently `None`); (4) §6's "M2 spotcheck +22 °C = Kelvin vs °C convention" is **misdiagnosed** — it is this double-decode, not a Kelvin/°C convention. Rebuild is not required for the z-scored experiments but is required before any physical-unit use or nonlinear feature.

### 11.4 Implication for the branch + centered-SSH result
On equal footing (chronological): golden **0.514** vs point_cube **0.577**, gap ~**0.063**.

**Centered-SSH `point_cube_v2` — RAN, did NOT close the gap.** Added a versioned `value_centered` operator (`ssh.value@local − basin_ssh`, cache `train_ready_affd326834f0.pkl`, config `config/argo/config_argo_point_cube_v2.json`) and trained (`-id point_cube_v2`). Result:

| Model | chronological test T | S |
|-------|----|----|
| golden (v2 pipeline) | 0.514 | 0.083 |
| point_cube (raw SSH) | 0.577 | 0.089 |
| **point_cube_v2 (centered SSH)** | **0.579** | 0.087 |

Centering is neutral (Δ ≈ +0.002 T, within noise; best val_loss 0.136 ≈ point_cube 0.135). **Interpretation:** the per-sample basin-SSH subtraction is largely redundant with the day-of-year harmonics (`timecos/timesin`) the point model already ingests — it can separate the basin-wide seasonal/steric component internally, so handing it `adt − basin` explicitly adds little. (Note: the earlier "z-scoring cannot recover the per-sample subtraction" reasoning is correct, but the recovered signal turns out not to matter here.) So the ~0.06 gap is **not** explained by SSH centering.

**Remaining candidates for the gap (chronological, equal-footing):** the mandated `point_cube` lr/bs sweep (not yet run); the residual head retrained against the correct anchor (§11.1/11.3); and only then, if a real gap persists, the SMAP/AVISO/MUR ingestion. Per-product **profile-location** imputation is negligible (residual cache `valid_mask`: sat value/grad/geo features 0% imputed at GoM profiles; the ~34% A-V5 ocean-NaN is basin land fraction, identical across products, not a CMEMS-SSS gap), so imputation is not the cause.
