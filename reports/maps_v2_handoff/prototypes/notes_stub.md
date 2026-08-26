# Companion notes (stub)

These notes live in a **separate file** from the slide HTML. Production will keep that split.

## Audience

Computational science graduate students (some oceanography) and an academic advisor.

## Spine (display names)

| Code / cell | Display name |
|-------------|--------------|
| `PatchConvMLP` | SatEncoder |
| A×CRPS / AxCRPS probabilistic head | Uncertainty Head |
| `HeaveResidualFast` | Heave Residual |
| Export contract into data assimilation | DA handoff |

## Acronyms (first-use expansions for the deck)

- **NeSPReSO.** Neural Sparse Profiling of the Subsurface Ocean (working expansion used in this project; confirm preferred long form with the advisor if needed).
- **ARGO.** Array for Real-time Geostrophic Oceanography (global profiling float network).
- **PCA.** Principal Component Analysis.
- **CRPS.** Continuous Ranked Probability Score.
- **MLD.** Mixed Layer Depth.
- **D26.** Depth of the 26 °C isotherm.
- **DA.** Data Assimilation.
- **SLA.** Sea Level Anomaly.
- **SST / SSS / SSH.** Sea Surface Temperature / Salinity / Height.

## Architecture facts pulled from code (prototype memory)

- SatEncoder (`PatchConvMLP`) embeds scalar encodings + satellite channels; optional patch Conv2d trunk; probabilistic mode emits `[μ, σ]` via `mu_out` / `sigma_out` + softplus.
- Uncertainty Head trains with closed-form Gaussian CRPS (`gaussian_crps_torch` in `evalphys/calibration.py`).
- Heave Residual predicts warp `(MLD, D26, stretch)` plus residual T/S PCs on a canonical depth grid, then unwarps to physical profiles (`model/heave.py`, `model/warp.py`, `HeaveResidualFast` batched warp).

## Open for production notes

Fill speaker asides, caveats, and pointer paths here. Do not put this text into the slide stage.
