# Agent instructions — ISAS20_project

**Start here:** [`HANDOFF.md`](HANDOFF.md) (status, eval rules, next tasks).

This repo is **offline PyTorch batch ML** for NeSPReSO v2 — not a web app. Code lives in [`NeSPReSO2_onTemplate/`](NeSPReSO2_onTemplate/).

## Philosophy

**Ponytail (lazy senior dev)** is always on — see [`.cursor/rules/ponytail.mdc`](.cursor/rules/ponytail.mdc). YAGNI, stdlib first, deletion over addition, `ponytail:` comments on intentional shortcuts.

**Not lazy about:** trust-boundary validation (`validate_config`), pickle/checkpoint trust, numerical reproducibility (`seed`, split pins), cross-tag eval honesty, anything the user explicitly requested.

## Engineering principles (load on demand)

Read [`eng-principles-pack/INDEX.md`](eng-principles-pack/INDEX.md) first. Load **one hook + one or two skills** matching the task — do not ingest the whole pack.

| Task | Load |
|------|------|
| Loss / PCA / preproc / model math | `hooks/pre-merge-numerical.md` + `skills/numerical-code.md` |
| Tests / selfcheck | `skills/testing-strategy.md` |
| AI-written numerical code | `skills/ai-code-verification.md` |
| Before commit | `hooks/pre-commit-checklist.md` |

## Project skills

| Skill | When |
|-------|------|
| [`nespreso-train-eval`](.cursor/skills/nespreso-train-eval/SKILL.md) | Training, caches, eval, dual-run monitor |
| [`nespreso-numerical`](.cursor/skills/nespreso-numerical/SKILL.md) | v2 equivalence, tolerances, matched eval |

## Commands (HPC)

Always cap CPU scope:

```bash
cd NeSPReSO2_onTemplate
srun --ntasks=1 --cpus-per-task=8 python3 selfcheck.py
srun --ntasks=1 --cpus-per-task=8 --gres=gpu:1 python3 train.py -c config_argo.json
```

GPU training: add `--gres=gpu:1`. See [`NeSPReSO2_onTemplate/README.md`](NeSPReSO2_onTemplate/README.md).

## Hard rules

1. **Never mix checkpoint PCA with a different cache** — pair `-r` checkpoint with the cache it was trained on.
2. **Do not compare raw `eval_run.py` RMSE across `isas20` vs `argo_v2`** — different depth grids and truths; use `eval_matched.py`.
3. **Paths in config JSON**, not hardcoded in Python (except test fixtures in `selfcheck.py`).
4. **GoM perf:** do not enable `combo_phase4b_all`, bf16, or `torch.compile` without benchmark proof ≥10% full-step gain.
5. **Commits:** only when the user asks. **Push:** only when asked.

## Plans

| Doc | Purpose |
|-----|---------|
| [`HANDOFF.md`](HANDOFF.md) | Session handoff (read first) |
| [`PLAN-phase5.md`](PLAN-phase5.md) | AE decoder roadmap (closed) |
| [`PLAN-phase6.md`](PLAN-phase6.md) | Diagnostics, global regional notebook |
| [`PLAN-patch-arch-handoff.md`](PLAN-patch-arch-handoff.md) | Phases 1–4b detail |
