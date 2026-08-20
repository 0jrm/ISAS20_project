# OSSE claim language (rewrite — do not change frozen `osse_results.md` numbers)

Frozen cast-column v1 (2021, n=1101): E3 T RMSE **0.545** vs E2 ISOP **0.541**; E4 localized full Σ **0.616**. Those numbers stay in [`osse_results.md`](osse_results.md).

## What the next OSSE may claim

Score **native 1 m hydrography vs nature**, not depth-pooled T after `layer_sample` / `xa2inc`.

Allowed headlines:

1. **50–200 m T vs nature** (thermocline band), not 0–2000 m pooled RMSE.
2. **Thermocline-depth error**: D20, D26, and max-N² depth RMSE vs nature.
3. **Analysis not farther from the synthetic than xb** on those metrics (increment does not degrade the column).
4. Optional: **24 h KE** if the HYCOM cycle is run later — not this sprint.

Depth-pooled T RMSE is a **footnote**, not the trophy. SLA/Cooper–Haines already owns heave; leftover skill is residual hydrography (spice, plume, double thermocline, MLD independent of SSH).

Steric vs ADT in the Loop Current box (24–28°N, 88–84°W) is a **ship gate (~2 cm RMS)**, not an RC-2 claim. Fail the gate → do not promote.

Diagonal R in (η, warped-residual) space. Dense localized R in z already hurt (E4 vs diag-control).

**Out of scope this sprint:** TSIS insertion, `xa2inc`, competing with SLA on heave.

CPC **ONI** and **RONI** are optional surface inputs (spliced after the six calendar harmonics; files under `data/indices/`).
