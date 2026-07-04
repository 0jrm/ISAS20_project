# AI Agent Design Specification: Point-Anchored Residual Patch Model

## 1. Overall Objective

### Goal

Design an architecture that **cannot underperform the point model in principle**, while allowing spatial-temporal patch information to improve predictions where additional context is informative.

### Core Principle

The patch input is a strict superset of the point model inputs.

Architecture must preserve this property.

---

# 2. High-Level Architecture

```
                 Surface observations

         ┌─────────────────────────────────┐
         │                                 │
         │  Center Pixel                   │
         │  (SSH SST SSS)                  │
         │                                 │
         └────────────┬────────────────────┘
                      │
               Point Feature Builder
                      │
      time harmonics + spatial harmonics
      + bathymetry
      + center SSH/SST/SSS
                      │
               Point Encoder
          (warm-start / frozen option)
                      │
                  Base PCs
                      │
                      ├──────────────┐
                      │              │
                      │          Gate g
                      │              │
                      ▼              │
             Residual Patch Encoder  │
                 ΔPC                 │
                      │              │
                      └──────┬───────┘
                             ▼

                 Output = Base + g·ΔPC
```

---

# 3. Module Specifications

---

## Module A — Dataset Builder

### Responsibilities

Construct training examples.

### Inputs

Point features

* time harmonics
* spatial harmonics
* bathymetry
* center SSH
* center SST
* center SSS

Patch features

* 5×5×7 SSH
* 5×5×7 SST
* 5×5×7 SSS

Targets

* 32 PC coefficients

---

### Preprocessing

Mandatory:

* z-score every continuous feature
* statistics computed on training split only
* identical transform applied to validation/test

Store:

```
mean
std
feature names
normalization version
```

---

### Derived Patch Features

Generate:

* center-relative SSH
* center-relative SST
* center-relative SSS

Optional:

* SSH gradients
* SSH Laplacian
* temporal tendency
* SST gradients

Keep original channels if desired.

---

### Tests

### Test A1

Training split statistics

Expected:

```
mean ≈ 0
std ≈ 1
```

Tolerance:

```
|mean| < 0.02

0.98 < std < 1.02
```

---

### Test A2

No train/test leakage.

Verify normalization statistics originate only from train split.

---

### Test A3

Center-relative transform

For every sample

```
center pixel == 0
```

Must hold exactly.

---

---

# Module B — Point Encoder

Purpose

Reproduce (or exceed) existing point model.

---

### Inputs

Exactly:

```
time harmonics

space harmonics

bathymetry

center SSH

center SST

center SSS
```

No omissions.

---

### Output

32 PCs

---

### Initialization

Preferred

Warm-start from trained point model.

Fallback

Train independently.

---

### Optional modes

Frozen

Fine-tuned

Joint training

---

### Tests

### Test B1

Input completeness

Verify all 9 predictive variables exist.

Fail if local SSH/SST/SSS absent.

---

### Test B2

Warm-start integrity

Initial predictions should match saved point model.

Tolerance:

```
RMSE difference < 1e-6
```

(before training)

---

### Test B3

Frozen mode

No parameter updates.

Gradient norm

```
== 0
```

---

---

# Module C — Patch Encoder

Purpose

Extract only spatial-temporal residual information.

---

### Inputs

Center-relative patches.

Never raw-only patches.

---

### Design constraints

Small capacity.

Avoid aggressive pooling.

Avoid losing spatial localization.

Possible backbone:

```
Conv

Residual block

Attention

Global pooling

MLP
```

Avoid:

```
Conv

Conv

AdaptiveAvgPool
```

that destroys center information.

---

### Regularization

weight decay

dropout

small hidden dimension

early stopping

---

### Output

```
ΔPC
```

32 values.

---

### Tests

### Test C1

Localization

Perturb center neighbor.

Encoder output must change.

If insensitive:

FAIL

---

### Test C2

Center invariance

Adding constant offset to whole patch

(after center-relative transform)

should leave output unchanged.

---

### Test C3

Gradient sensitivity

Synthetic linear SSH gradient

Encoder output must differ from flat field.

---

---

# Module D — Residual Combiner

Formula

```
Output = Base + g * ΔPC
```

---

### Initialization

Gate

```
g = 0
```

or

Last residual layer

weights = 0

---

### Desired property

Initialization exactly reproduces point model.

---

### Tests

### Test D1

Immediately after initialization

```
Output == Base
```

Numerical tolerance

```
1e-8
```

---

### Test D2

Residual norm

Initially

```
||Δ||

≈0
```

---

### Test D3

Gate learning

After training

```
g

≠0
```

for at least some validation samples.

---

---

# Module E — Trainer

Training stages

---

### Stage 1

Train point model.

Save checkpoint.

---

### Stage 2

Initialize residual model.

Residual branch disabled.

Verify baseline.

---

### Stage 3

Train residual only.

Optional:

freeze point encoder.

---

### Stage 4

Optional joint fine-tuning.

Small learning rate.

---

### Loss

Primary

PC regression loss.

Optional

weighted PCs.

---

### Early stopping

Validation RMSE.

Never training loss.

---

### Tests

### Test E1

Epoch 0 validation

Residual model

≈ point model.

Difference

<0.1%

---

### Test E2

Training stability

No exploding gradients.

Gradient norm monitored.

---

### Test E3

Convergence

Validation improves after residual enabled.

---

---

# Module F — Evaluation

Metrics

Primary

Temperature RMSE

Target:

```
Baseline

0.416

Goal

<0.416
```

---

Secondary

PC RMSE

Salinity RMSE

Depth-wise RMSE

---

Regional analysis

Evaluate separately:

eddy core

eddy edge

background

rapid evolution

---

### Statistical testing

Use paired tests.

Per-profile differences.

Recommended:

paired t-test

Wilcoxon signed-rank

bootstrap confidence interval

---

### Tests

### Test F1

Overall improvement

Mean RMSE

```
Residual

<

Point
```

---

### Test F2

Statistical significance

95% CI excludes zero.

or

```
p < 0.05
```

---

### Test F3

No catastrophic regressions

Profiles with worse prediction

must not increase dramatically.

Monitor distribution.

---

---

# Module G — Diagnostics

Every run automatically generates

Training curves

Validation curves

Residual magnitude

Gate values

Feature normalization report

Gradient norms

Residual contribution histogram

Per-PC errors

Depth-wise reconstruction error

Regional breakdown

---

### Tests

### Test G1

Normalization report

No feature

```
std < 0.05

or

std > 5
```

after normalization.

---

### Test G2

Residual utilization

Residual branch not permanently inactive.

Mean residual norm

> threshold.

---

### Test G3

Branch dependence

Ablation:

```
Point only

Point + residual
```

Residual should produce measurable prediction differences.

---

# 4. Experimental Roadmap

## Phase 1

Implement train-split z-scoring.

Success criterion:

stable optimization.

---

## Phase 2

Restore local SSH/SST/SSS to scalar encoder.

Expectation:

largest immediate improvement.

---

## Phase 3

Implement residual architecture.

Zero-initialized.

Must exactly reproduce baseline at epoch 0.

---

## Phase 4

Switch patch input to center-relative representation.

---

## Phase 5

Add engineered spatial features.

Priority:

1. SSH gradients
2. SSH Laplacian
3. temporal tendency
4. SST fronts

---

## Phase 6

Optional self-supervised patch pretraining.

Use abundant satellite archive.

Fine-tune on profile prediction.

---

# 5. Design Choices and Trade-offs

| Decision                          | Rationale                                                         | Trade-off                                          |
| --------------------------------- | ----------------------------------------------------------------- | -------------------------------------------------- |
| Residual architecture             | Preserves strong baseline by construction                         | Limits residual branch influence early in training |
| Include center scalars explicitly | Restores superset guarantee                                       | Slight feature redundancy with patch               |
| Center-relative patches           | Forces learning of spatial structure rather than local values     | Absolute magnitude must come from point branch     |
| Small patch encoder               | Better sample efficiency (~2900 profiles vs. 525+ patch features) | May miss highly complex interactions               |
| Zero initialization               | Guarantees baseline-equivalent initialization                     | Slower initial residual learning                   |
| Warm-start point encoder          | Reuses validated representation                                   | Requires checkpoint compatibility                  |
| Freeze then fine-tune             | Stable optimization                                               | Slightly longer training pipeline                  |
| Gradient/Laplacian features       | Encodes physically meaningful eddy structure                      | Additional preprocessing complexity                |

---

# 6. Known Limitations

* Theoretical guarantees apply only at the Bayes-optimal level; finite-sample optimization can still degrade performance if the residual branch is poorly trained or overfits.
* Spatial context likely contributes only second-order information because local SSH/SST/SSS already explain much of the recoverable subsurface variance.
* With ~2,900 training profiles and a high-dimensional patch input, sample efficiency remains the primary bottleneck; careful regularization is essential.
* Improvements are expected to be concentrated in dynamically complex regions (e.g., eddy edges and rapidly evolving systems) rather than uniformly across all samples.
* Expected gain is modest (approximately 5–15% RMSE reduction, from ~0.416 toward ~0.35–0.40), making rigorous paired statistical evaluation necessary to distinguish real improvements from sampling variability.
