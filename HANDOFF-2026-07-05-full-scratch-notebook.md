# HANDOFF — Full from-scratch notebook (all main models incl. cube-native)

**Date:** 2026-07-05 → 2026-07-06 (completed)
**Branch:** `residual_cube` (uncommitted working-tree changes, see §5)
**Status:** ✅ **COMPLETE** — from-scratch run finished (papermill EXIT=0), S0 anchoring bug found+fixed+retrained, eval-only rerun with S0 **PASS** finished 2026-07-06 00:26. Final results in §0. tmux session `scratch_nb` is idle (kept for inspection).
**Companion docs:** `NeSPReSO2_onTemplate/HANDOFF_cube-native_point_anchor.md` (corrected scoreboard, §11 audit), `NeSPReSO2_onTemplate/HANDOFF_residual-cube.md` (design spec)

---

## 0. FINAL RESULTS (chronological test, n=623, native-depth RMSE)

Authoritative artifact: `NeSPReSO2_onTemplate/notebooks/_executed_full_scratch_all_models_final.ipynb`
(eval-only rerun, all checkpoints pinned from the manifest; S0 PASS). The first executed notebook
`_executed_full_scratch_all_models.ipynb` is the from-scratch training record — its residual row
is the **unanchored** artifact (see §3b), all other rows identical.

| Row | T | S | vs reference |
|-----|---|---|--------------|
| Climatology | 1.657 | 0.216 | == 1.657/0.216 (split discipline confirmed) |
| Clim + SLA GEM | 1.076 | 0.155 | |
| Point (raw PCs, golden lineage) | **0.537** | 0.090 | ref 0.514/0.083 (run variance) |
| ANOM-point | 0.680 | 0.104 | ref 0.647 |
| ANOM-L4-patch | **0.933** | 0.137 | **was 1.857** — cache rebuild picked up the satellite gap downloads; now beats climatology, still no match for point models |
| Point-cube (anchor) | **0.577** | 0.088 | == 0.577/0.089 reference exactly |
| Cube residual (anchored, fixed) | **0.598** | 0.092 | anchor +0.021 T — **gate did not add test skill** |

Key takeaways:
1. **S0 now PASSES** (residual@init ≡ point_cube, 0.5770/0.0885) — first properly anchored residual train ever (7.3 min, best epoch 65, vs 41 min unanchored-from-scratch).
2. **Anchored residual finding:** with anchoring verified, the 32 differential surface features
   currently add no test skill (dT=+0.021; val-selected → §11.3 val→test shift pattern). S2 met
   (0.598 < ANOM-point 0.647), S3 not (golden 0.537/0.514). Per design §9 this is a *finding*,
   not a failure mode — next probes: gate_l1 trajectory, val-vs-test residual delta, L1 gate
   penalty, lr sweep, then M7 ablations on the fixed cache.
3. **ANOM-L4-patch recovered** from "loses to climatology" (1.857) to 0.933 after the from-scratch
   cache rebuild — the stale-satellite fix (2026-07-03 downloads) was the dominant issue there.
4. All five model rows passed the `eval_run.py` cross-check; fresh point_cube's early best epoch
   (27) was benign — it reproduces the 0.577 reference exactly.

---

## 1. What this session did

Built and launched `NeSPReSO2_onTemplate/notebooks/full_scratch_all_models.ipynb` — a
`build_anom_notebook.py`-style generated notebook that, with **`USE_TRAINED_MODEL = False`**:

1. **Rebuilds the GoM Zarr cube** (Component A) — the on-disk cube was the stale rev-2
   double-decoded store (cube-native handoff §11.5). `DATA_REVISION` was bumped **2 → 3** in
   `preproc/cube/cube_schema.py`; the notebook detected `rev 2 < 3`, moved the stale store to
   `data/cube/gom_cube_stale_rev2_0705_204716.zarr`, and is rebuilding from the NetCDF archives,
   then runs `--validate` plus an in-notebook physical-range assert on SST (guards the
   double-decode bug class).
2. **Force-rebuilds every train-ready cache**: v2 point (`argo_v2`), anomaly point, L4 anomaly
   patch (`argo_l4`, incl. DUACS SLA resampling — picks up the post-07-03 satellite gap
   downloads), and both cube caches (9-D point + 41-D residual).
3. **Trains all five main models from scratch**, sequentially, on chronological 70/15/15:
   | key | label | config | notes |
   |-----|-------|--------|-------|
   | `golden_point` | Point (raw PCs) | `config/argo/config_argo.json` | golden lineage, PCA source for cube caches |
   | `anom_point` | ANOM-point | `config_argo_anom.json` | |
   | `anom_patch_l4` | ANOM-L4-patch | `config_argo_patch_l4_anom.json` | full-batch (bs=0) |
   | `point_cube` | Point-cube | `config_argo_point_cube.json` | bs→256 (VRAM budget) |
   | `residual_cube` | Cube residual | `config_argo_residual_cube.json` | bs→128; **first real train of the current pipeline** |
   Runtime wiring (not in the JSONs): cube configs get `io.pca_ckpt` → fresh `golden_point`
   checkpoint; residual gets `arch.args.warmstart_ckpt` → fresh `point_cube` checkpoint. This is
   the retrain that cube-native handoff §11.1 said was mandatory before any residual-vs-anchor claim.
4. **S0 anchoring check** before residual training: untrained residual (gate=0) must reproduce
   point_cube test RMSE within 1e-3 (prints `*** FAIL ***` loudly but continues).
5. **Evaluates everything on the chronological test split** (n=623): climatology + GEM baselines
   (from the anom point cache) + 5 model rows, physical space, common 0–1800 m grid,
   `eval_run.py` cross-check per row, depth/scatter/residual/example/map plots, JSON export.

Checkpoints go to `saved/models/<exper>/scratch_0705_204716_<key>/model_best.pth`.

---

## 2. How to monitor / control the run

```bash
tmux attach -t scratch_nb                     # live papermill output (Ctrl-b d to detach)
tail -f NeSPReSO2_onTemplate/notebooks/scratch_outputs/papermill_run.log
# cell-by-cell progress + saved outputs (papermill saves after every cell):
ls -la NeSPReSO2_onTemplate/notebooks/_executed_full_scratch_all_models.ipynb
cat NeSPReSO2_onTemplate/notebooks/scratch_outputs/scratch_manifest.json   # ckpt+cache per trained model
```

- Runs on **GPU 0** (`CUDA_VISIBLE_DEVICES=0`; ~6.5 GB free next to an idle vLLM server; GPUs 1–3
  are saturated by another user). Batch overrides keep us inside the ~6 GB budget from the
  cube-native handoff §7.
- Expected wall time: cube ~5–10 min, caches ~minutes–1 h (DUACS resampling dominates), then five
  sequential trainings (8000 epochs, early-stop 500) — likely **several hours to overnight**.
- Relaunch after a crash: `tmux new -d -s scratch_nb "bash NeSPReSO2_onTemplate/scripts/run_full_scratch_notebook.sh"`.
  The cube is only rebuilt if its `data_revision` attr is stale, but caches/training rerun in full
  while `USE_TRAINED_MODEL=False`.
- **Eval-only rerun** (after training finished once): edit Section 0 of the notebook (or the
  builder) to `USE_TRAINED_MODEL = True` — checkpoints and cache paths are then pinned from
  `scratch_manifest.json`, nothing retrains.

---

## 3. Outputs to expect when it finishes

| Artifact | Path (under `NeSPReSO2_onTemplate/`) |
|----------|--------------------------------------|
| Executed notebook | `notebooks/_executed_full_scratch_all_models.ipynb` |
| Run log | `notebooks/scratch_outputs/papermill_run.log` |
| Manifest (ckpt/cache/epoch/wall per model) | `notebooks/scratch_outputs/scratch_manifest.json` |
| Results JSON | `notebooks/scratch_outputs/scratch_all_models_results.json` |
| Depth RMSE overlay | `notebooks/scratch_outputs/scratch_depth_rmse.png` |
| Rebuilt cube (rev 3) | `data/cube/gom_cube.zarr` (stale rev-2 kept as `gom_cube_stale_rev2_0705_204716.zarr`) |

**Reading the results (split discipline, cube-native handoff §10):** everything is chronological
test. Reference points: golden chrono **T=0.514/S=0.083**, point_cube **0.577/0.089**, clim
**1.657/0.216**. Cube-rebuild rev-3 is affine-equivalent to rev-2 after z-scoring, so
point_cube should land near 0.577 again. Watch for: (a) S0 PASS, (b) residual vs point_cube delta
(gate opening = signal in the differential features), (c) ANOM-L4-patch — memory says its test
window was 100 % stale satellite; the forced cache rebuild picks up the gap downloads, so this is
the first honest read on that model.

---

## 3b. ⚠️ S0 anchoring FAILED mid-run — root cause found and FIXED in code (2026-07-05 late)

The notebook's S0 gate fired: residual@init T RMSE **7.27** vs point_cube **0.577** (the §5.1
"~7.25" signature). Root cause — a real pipeline bug, present in **every residual run to date**:

- `build_point_cube_cache` **z-scores** sat cols 6–8 → point_cube trains on z-scored sats.
- `build_feature_cache` stored the residual point block **raw** (the literal C-I2 reading), and
  `PointAnchoredResidual.forward_base` feeds `x[:, :9]` straight through with **no internal
  scaling** → the frozen base sees raw SST/SSS/SSH where it expects z-scored → garbage anchor.
- `test_s0b_residual_init_matches_point_cube` never caught it because it feeds the *same*
  residual-cache inputs to both models — it proves weight-copy fidelity, not standardization
  consistency. The prior session's "s0b passes" was vacuous; **all previous residual_v1 numbers
  (0.631 etc.) were unanchored**.

Fixes applied (working tree):
1. `preproc/features/export_feature_cache.py` — residual point block now z-scored on the train
   split with the same recipe as the point cache; stats stored under
   `input_standardization.point_mean/point_std`; `cube_feature_hash` gained a
   `point_block_norm` marker so the fixed cache re-hashes (**`train_ready_76aa50b84810.pkl`**),
   leaving the point-cache hash untouched.
2. `tests/test_residual_init.py` — new `test_s0b_point_block_matches_point_cache` asserts the
   real invariant: residual cache cols `[:9]` ≡ point-cube cache inputs.
3. `scripts/fix_residual_retrain.py` — post-run driver: builds the fixed cache, retrains
   residual_cube (warmstart = fresh point_cube, pca = fresh golden, bs=128), updates the manifest.
4. Builder now reads `USE_TRAINED_MODEL` from the env (`USE_TRAINED_MODEL=1`) for a cheap
   eval-only notebook rerun with pinned checkpoints after the retrain.

**The residual_cube row in the first executed notebook is unanchored — ignore it.** All other
rows (baselines, golden, ANOM-point, ANOM-L4-patch, point_cube) are unaffected.

Also noteworthy from the run: fresh `point_cube` best epoch was **27** (vs 528 previously) —
check its eval row against the 0.577 reference before treating it as a healthy anchor.

## 4. Known caveats

- The rev-2 → rev-3 cube rebuild changes stored values to physically correct units, but every
  model input is train-split z-scored, so model metrics are expected to be statistically unchanged.
- Old cube caches (`308c9ec95f5a`, `2ab55b15b14f`, `affd326834f0`) are now orphaned by the
  `DATA_REVISION` bump — old checkpoints (`point_cube`, `point_cube_v2`, stale `residual_v1`)
  still pair with those pickles only.
- `anom_point` is known to need a loss-scale retune (memory note, 2026-07-03); it is trained
  as configured here, so expect that row to underperform its potential.
- Plot cells are wrapped in `safe()` — a plotting failure prints a traceback and continues, it
  does not kill the run. Training/cache failures **do** kill the run (by design).
- tmux launcher gotchas (already handled in `scripts/run_full_scratch_notebook.sh`): no `set -u`
  (conda gdal hook), explicit `PATH` prefix (tmux server env is bare), papermill was pip-installed
  into `nespreso` this session.

---

## 5. Working-tree changes (uncommitted, per repo rules)

| File | Change |
|------|--------|
| `NeSPReSO2_onTemplate/preproc/cube/cube_schema.py` | `DATA_REVISION` 2 → 3 (decode-fix rebuild) |
| `NeSPReSO2_onTemplate/notebooks/build_full_scratch_notebook.py` | **new** — notebook generator |
| `NeSPReSO2_onTemplate/notebooks/full_scratch_all_models.ipynb` | **new** — generated notebook (38 cells) |
| `NeSPReSO2_onTemplate/scripts/run_full_scratch_notebook.sh` | **new** — tmux/papermill launcher |
| `NeSPReSO2_onTemplate/preproc/features/export_feature_cache.py` | **S0 fix** — residual point block z-scored (point-cache recipe) + hash marker |
| `NeSPReSO2_onTemplate/tests/test_residual_init.py` | new `test_s0b_point_block_matches_point_cache` (real anchoring invariant) |
| `NeSPReSO2_onTemplate/scripts/fix_residual_retrain.py` | **new** — fixed-cache rebuild + anchored residual retrain driver |
| `HANDOFF-2026-07-05-full-scratch-notebook.md` | **new** — this file |
| `HANDOFF.md` | pointer updated |

## 6. Next steps

1. **Commit** the working-tree changes (user approval required): the S0 anchoring fix in
   `export_feature_cache.py` + strengthened test are correctness fixes that everything downstream
   depends on; the notebook/scripts/handoff ride along.
2. **Residual gate analysis** (why dT=+0.021): plot `gate_l1` over epochs from the fixed run's
   TensorBoard/log, compare residual-vs-anchor on **val** (if residual wins val but loses test →
   chronological shift; consider monitoring val profile_rmse and/or an L1 gate penalty per
   cube-native handoff §11.3).
3. **M7 ablations must be re-run on the fixed cache** (`76aa50b84810`) — all prior ablation JSONs
   inherit the unanchored-base bug. Also still outstanding: point_cube lr/bs sweep (§8.4 of
   cube-native handoff).
4. Feed the §0 numbers back into `HANDOFF_cube-native_point_anchor.md` and mark §11.1's
   "retrain residual_v1" item done (fixed variant: `scratch_0706_001728_residual_cube_fixed`).
5. Old cube caches (`308c9ec95f5a`, `2ab55b15b14f`, `affd326834f0`, `269d…`-era `e9680f670c7c`)
   and the stale rev-2 cube backup can be deleted once §0 is accepted.
