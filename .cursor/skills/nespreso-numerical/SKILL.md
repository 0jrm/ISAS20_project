---
name: nespreso-numerical
description: Numerical verification for NeSPReSO — v2 equivalence, tolerances, PCA/loss pins, reproducibility. Use when changing model/loss.py, preproc, PCA bases, or loss scales.
---

# NeSPReSO numerical checks

Sourced from [`eng-principles-pack/skills/numerical-code.md`](../../../eng-principles-pack/skills/numerical-code.md) + project pins.

## One command

```bash
cd NeSPReSO2_onTemplate
srun --ntasks=1 --cpus-per-task=8 python3 selfcheck.py
```

ponytail: single file, assert-based — no pytest framework.

## Tolerances

- v2 forward/loss equivalence: `TOL = 1e-6` in `selfcheck.py` (golden dict).
- Never `assert x == y` on floats; use `abs(x - y) < tol` or `torch.allclose`.
- Changing `outputs` PC count breaks v2 parity intentionally — update goldens and re-derive loss scales.

## Reproducibility

- `seed=42` + `torch.random_split(..., generator=manual_seed(42))` for train/val/test.
- Pin split in `selfcheck.py` if split logic changes.
- `cudnn_deterministic: true` by default in performance config.

## Loss scales

GoM defaults in `model/loss.py` are region-specific. After cache or PC count change:

```bash
python3 scripts/derive_loss_scales.py -c config_argo.json --update-config
```

## Pre-merge checklist (numerical changes)

From [`eng-principles-pack/hooks/pre-merge-numerical.md`](../../../eng-principles-pack/hooks/pre-merge-numerical.md):

- [ ] `selfcheck.py` passes
- [ ] Tolerances explicit (not exact float equality)
- [ ] Seeds documented if stochastic
- [ ] Eval RMSE on test split if loss/objective changed
- [ ] Cross-tag claims use `eval_matched.py` only

## AI-generated numerical code

Verify algorithm choice explicitly. AI often gets loop bounds and PCA layout wrong. Run `selfcheck.py` and spot-check one forward pass against v2 if touching `PredictionModel` / `CombinedPCALoss`.
