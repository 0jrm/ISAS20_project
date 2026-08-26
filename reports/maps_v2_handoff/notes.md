# NeSPReSO v2 → DA · companion notes

**How to open.** Open [`index.html`](index.html) in a browser (local file is fine; KaTeX needs network once for the CDN). Use **JSON editor** to show/hide the editable deck. Change **Slide layout** (A/B/C) per slide; **Default** applies when a slide omits `layout` (production default is **C**). Persist edits with **Save HTML**. These notes are a **separate file** on purpose.

**How to make JSON edits permanent.** Click **Save HTML**. Prefer Chromium/Chrome: pick the existing deck file to overwrite (same dialog reads + writes; later saves in that tab reuse it). If the browser has no file-system picker (some embedded viewers), you will be asked to choose the existing `.html`, then a baked download appears — replace the on-disk file with that download. Reload to confirm. Same for [`da_update.html`](da_update.html).

**Audience.** Computational science graduate students (some oceanography) and an academic advisor.

**Color accents (match the slides).**

- <span style="color:#0B6E4F;font-weight:700">Green · SatEncoder</span>
- <span style="color:#C81D25;font-weight:700">Red · Uncertainty Head (A×CRPS)</span>
- <span style="color:#3D348B;font-weight:700">Indigo · Heave Residual</span>
- <span style="color:#E09F3E;font-weight:700">Amber · DA handoff</span>

**Sources.** Live code under `NeSPReSO2_onTemplate/` (`model/model.py`, `model/prob_head.py`, `model/heave.py`, `model/heave_fast.py`, `model/warp.py`, `model/loss.py`, `evalphys/calibration.py`, `scripts/export_sigma_o.py`) and `reports/heave_da_serve_spec.json`. No invented OSSE scoreboard numbers here.

---

## Slide 1 · Cover

**NeSPReSO** means Neural Sparse Profiling of the Subsurface Ocean in this project’s working language. The product for **Data Assimilation (DA)** is a subsurface Temperature/Salinity (**T/S**) profile (and a disciplined observation-error story), not a research plot alone.

Two heads share the same trunk idea:

1. **Uncertainty Head** — operational **A×CRPS** cell (PCA-coefficient probabilistic head).
2. **Heave Residual** — warp landmarks + residual PCs on a canonical depth grid.

Now let’s walk the trunk, then each head, then what DA is allowed to ingest.

---

## Slide 2 · Shared trunk: SatEncoder

<span style="color:#0B6E4F;font-weight:700">SatEncoder</span> is the display name for code class **`PatchConvMLP`**.

**What it does.** Takes scalar encodings plus satellite channels and builds a latent vector \(h\).

- **Point mode** (`patch_shape=None`): linear embed of three satellite scalars.
- **Patch mode**: reshape the flat satellite block to \((B,C,T,H,W)\), run a small **Conv2d** trunk, project, then an **MLP** (Multi-Layer Perceptron) head stack.

Fusion is additive in feature space:

\[
h = \mathrm{EncProj}(e) + \mathrm{SatProj}(\mathrm{sat})
\]

Both A×CRPS and Heave Residual instantiate this backbone (Heave wraps it inside `HeaveResidual`).

---

## Slide 3 · Inputs & fusion

**Uncertainty Head (A×CRPS)** — **9-D** input:

| Block | Fields |
|-------|--------|
| Harmonics (6) | time cos/sin, lat cos/sin, lon cos/sin |
| Satellite (3) | **SSS**, **SST**, **SSH** (Sea Surface Salinity / Temperature / Height) |

`n_enc=6`, `n_sat=3`, `input_dim=9`, `output_dim=32` (16 T + 16 S PCA scores).

**Heave Residual** — **11-D** input: the same six harmonics, then **ONI** (Oceanic Niño Index) and **RONI** (Relative Oceanic Niño Index), then SSS, SST, SSH.

`n_enc=8`, `n_sat=3`, `input_dim=11`. ENSO columns splice **after** harmonics and **before** satellite columns (`preproc.enso.inject_enso_columns`).

Same SatEncoder *class*, different input width and head layout.

---

## Slide 4 · Uncertainty Head architecture

<span style="color:#C81D25;font-weight:700">Uncertainty Head</span> = operational **PCA-coeff A×CRPS**.

After the trunk, two linear maps share \(h\):

- \(\mu = \texttt{mu_out}(h)\) — predicted PCA coefficients.
- \(\sigma = \mathrm{softplus}(\texttt{sigma_out}(h)) + \sigma_{\min}\) — positive scales (`model/prob_head.py`).

When `probabilistic=true`, forward returns \([\mu \| \sigma]\) with width \(2d\). Stage training can freeze \(\sigma\) (`set_sigma_trainable`); the serve path still treats \(\sigma\) as research-grade unless gates pass (see DA slide).

---

## Slide 5 · CRPS equation

**CRPS** = Continuous Ranked Probability Score. It scores an entire predictive distribution against a scalar observation.

For a Gaussian forecast \(\mathcal{N}(\mu,\sigma^2)\) and observation \(y\), NeSPReSO uses the closed form in `evalphys.calibration.gaussian_crps_torch`:

\[
\mathrm{CRPS}(\mu,\sigma,y)=\sigma\Big[z\big(2\Phi(z)-1\big)+2\varphi(z)-\tfrac{1}{\sqrt{\pi}}\Big],\quad z=\tfrac{y-\mu}{\sigma}
\]

with \(\varphi,\Phi\) the standard normal density and CDF. Training takes the mean over PCA coefficient dimensions (`PCAHeteroLoss` / density–spice variants elsewhere). **A×CRPS** here means the matrix-A PCA-coefficient CRPS cell that is frozen for serve.

---

## Slide 6 · From PCA \(\sigma\) to T/S, spice, density, DA

This is the slide advisors ask about. Pause here.

### What PCA-space spread *is*

Scores live in a reduced basis \(V\) (PCA loadings). A diagonal score covariance induces a physical-space covariance

\[
\Sigma = V\,\mathrm{diag}(\sigma^2)\,V^{\mathsf T}.
\]

Mean reconstruction is the usual \(\mu_z = V\mu_{\mathrm{pc}} + \mathrm{mean}\).

### What it is good for

| Use | Verdict |
|-----|---------|
| **Research calibration** | Marginal CRPS on scores; decode and look at depth-band CRPS / **ENCE** (Expected Normalized Calibration Error), PIT, spread–skill. |
| **T and S profiles** | \(\mu\) decoded to physical T/S is the operational product. |
| **Spice / density** | Not separate A×CRPS outputs on this serve cell. Derive from decoded T/S (or use a dedicated density–spice representation elsewhere). |
| **DA observation error \(R\)** | **Do not** write CRPS-head \(\sigma\) into NetCDF err/\(R\) today. Ingest **Dai \(\sigma_o\)** (floored). Prefer **diagonal** \(R\). |

### Why diag preference

\(\Sigma = V\,\mathrm{diag}(\sigma^2)\,V^{\mathsf T}\) has rank at most equal to the number of PCs. Over a deep vertical grid that is near-null in most directions. Fed raw into column **OI** (Optimal Interpolation), that structure can hurt. Localization can stabilize algebraically, but off-diagonals induced by a shared \(V\) are not a learned joint error model. Serve contract: **μ only**; **\(R \approx \mathrm{diag}(\sigma_o^{\mathrm{Dai}})^2\)**.

Hard rule from `heave_da_serve_spec.json`: never mix a checkpoint with a different cache (PCA lives in the cache pickle).

---

## Slide 7 · Heave Residual: output layout

<span style="color:#3D348B;font-weight:700">Heave Residual</span> = `HeaveResidual` / `HeaveResidualFast`.

μ layout (width 35):

\[
\mu = [\underbrace{w_{0:3}}_{\text{warp logits}},\; \underbrace{z_T}_{16},\; \underbrace{z_S}_{16}]
\]

Residual PCs are **on the canonical depth grid**, not native-z PCA scores used by A×CRPS. Probabilistic mode doubles width with σ.

---

## Slide 8 · Warp mechanism (no metaphor — just the map)

Forget Residual-in-Residual picture books. The mechanism is **landmark registration**.

### Knots

**Canonical** (fixed references in `model/warp.py`):

\[
(0,\; \mathrm{MLD}_0=50\,\mathrm{m},\; \mathrm{D26}_0=120\,\mathrm{m},\; z_{\mathrm{bot}})
\]

**Physical** (per profile):

\[
(0,\; \mathrm{MLD},\; \mathrm{D26},\; z_{\mathrm{bot}})
\]

**MLD** = Mixed Layer Depth. **D26** = depth of the 26 °C isotherm. \(z_{\mathrm{bot}}\) is the bottom of the profile grid (serve caches use 1800 m).

### `decode_warp` (`model/heave.py`)

With gap constant \(\mathrm{GAP}_0 = 120-50-5 = 65\) and minimum layer 5 m:

\[
\begin{aligned}
\mathrm{MLD} &= 50\,e^{w_0},\\
\mathrm{D26} &= \mathrm{MLD} + 5 + 65\,e^{w_1},\\
\mathrm{stretch} &= 1 + 0.3\,\tanh(w_2).
\end{aligned}
\]

Raw zeros map to MLD 50 m and D26 120 m. Depths are clamped to keep layers ordered.

### Important honesty

**`stretch` is predicted and returned by `decode_warp`, but it is unused in reconstruct.** `physical_ts` drops it (`_stretch`); only MLD and D26 enter warp/unwarp. Say that out loud; it prevents a wild goose chase.

### Interp geometry (the actual map)

Let the knot vectors be

\[
\mathbf{c}=(0,50,120,z_{\mathrm{bot}}),\qquad
\mathbf{p}=(0,\mathrm{MLD},\mathrm{D26},z_{\mathrm{bot}}).
\]

Define the piecewise-linear maps (same as `np.interp` / torch `searchsorted` + lerp):

\[
z_{\mathrm{phys}}(z_{\mathrm{c}})
=\mathrm{interp}\bigl(z_{\mathrm{c}};\;\mathbf{c}\to\mathbf{p}\bigr),
\qquad
z_{\mathrm{c}}(z_{\mathrm{phys}})
=\mathrm{interp}\bigl(z_{\mathrm{phys}};\;\mathbf{p}\to\mathbf{c}\bigr).
\]

**Warp** (physical profile → values on the numeric canonical grid \(z\)):

\[
T_{\mathrm{canon}}(z)
=T_{\mathrm{phys}}\bigl(z_{\mathrm{phys}}(z)\bigr).
\]

In code: for each grid node \(z\), look up the physical depth that landmark-aligns to it, then sample \(T_{\mathrm{phys}}\) there (`phys_from_canon` then `interp` along \(z\)).

**Unwarp** (canonical profile → values on physical \(z\)):

\[
T_{\mathrm{phys}}(z)
=T_{\mathrm{canon}}\bigl(z_{\mathrm{c}}(z)\bigr).
\]

Inverse knot map, then sample. Same \(z\) array numerically; the *meaning* of “50 m” changes between frames.

Intuition: a shallow MLD compresses the mixed-layer segment and stretches the thermocline segment (or the reverse for a deep MLD), so residual Principal Components (PCs) always see the thermocline in a roughly fixed place on the canonical axis.

---

## Slide 9 · Residual on canonical → physical T/S

Decode path in `HeaveResidualLoss.physical_ts` / `decode_ts`:

1. \(\mathrm{MLD},\mathrm{D26} \leftarrow \texttt{decode_warp}(w)\).
2. \(T_{\mathrm{res}} = z_T V_T + \mu_T\) (canonical grid). Same for \(S\).
3. **Warp** climatology onto canonical landmarks: \(T_{\mathrm{prior}} = \mathrm{Warp}(T_{\mathrm{clim}};\mathrm{MLD},\mathrm{D26})\).
4. Add residual: \(T_{\mathrm{canon}} = T_{\mathrm{prior}} + T_{\mathrm{res}}\).
5. **Unwarp** to physical depth: \(T_{\mathrm{phys}} = \mathrm{Unwarp}(T_{\mathrm{canon}};\mathrm{MLD},\mathrm{D26})\).

Compact form:

\[
T_{\mathrm{phys}}
=\mathrm{Unwarp}\Bigl(
\mathrm{Warp}(T_{\mathrm{clim}};\mathrm{MLD},\mathrm{D26})
+ V_T z_T + \mu_T
\Bigr).
\]

In words: move the prior so its mixed layer and 26 °C depth sit on the fixed canonical knots, learn a residual *in that aligned frame*, then push the sum back to physical depth.

**Climatology fallback.** Some caches have no `clim_profiles`. Loss fallback is a **basin-mean** of training true profiles broadcast to every row (`heave_da_serve_spec.json`). Reproduce that; do not silently nearest-neighbor Argo casts.

HeaveResidualFast (same weights, batched warp) is an engineering detail, not a separate science slide — mention only if asked.

---

## Slide 10 · DA handoff contract

<span style="color:#E09F3E;font-weight:700">Contract pins</span> (from `reports/heave_da_serve_spec.json`):

1. **Serve μ only.** If forward width is \(2d\), take `[:, :d]`. Depth grid from cache `PRES` (1801 levels, 0…1800 m).
2. **Checkpoint ↔ cache pairing** is mandatory (PCA in cache).
3. Do not serve ablations listed under `do_not_serve` (conv3, bathy, …) as the production ingest.

### Define R, H, and Dai σₒ (say these out loud)

| Symbol | Meaning |
|--------|---------|
| **R** | Observation-error covariance in DA / OI. Today: **diagonal** \(R \approx \mathrm{diag}((\sigma_o)^2)\). Not a dense PCA-induced Σ. |
| **H** | TSIS observation operator. Averages the **1 m** T/S profile into the **41 HYCOM layers** whose interfaces come from the background column thickness (`thknss`). Interfaces are cycle-dependent — H is not a fixed z table. Apply H in TSIS, **not** in the ML export path. |
| **Dai σₒ** | Per-layer observation-error scale = RMSE of the served model vs Argo **after H**, then floored (0.05 °C / 0.02 psu). Export: `reports/sigma_o_hycom.csv` via `scripts/export_sigma_o.py`. **Not** 1 m RMSE. **Not** CRPS-head σ. |

CRPS-as-σₒ stays deferred until ENCE-by-band gates. Do not invent a gate number in the talk; point to the export script and current reports.

---

## Slide 11 · Architectural tradeoffs

Speak these as decisions, not slogans.

1. **Marginal CRPS vs joint \(R\).** Training fits per-score spreads. Induced \(\Sigma\) is basis-shaped, not a full error model. Diagonal Dai \(R\) is the safer DA ingest today.
2. **Native PCA vs heave.** A×CRPS: PCA on physical \(z\). Heave: landmarks + residual on canonical \(z\), then unwarp. Trade geometry prior for a more complex decode and a clim-prior dependency.
3. **Research σ vs ingest σ.** Keep both stories honest in the room: CRPS σ for calibration science; Dai \(\sigma_o\) for TSIS/OI until gates say otherwise.

---

## Acronym sheet (first-use expansions)

| Short | Full |
|-------|------|
| NeSPReSO | Neural Sparse Profiling of the Subsurface Ocean |
| DA | Data Assimilation |
| ARGO | Array for Real-time Geostrophic Oceanography (profiling floats) |
| PCA | Principal Component Analysis |
| CRPS | Continuous Ranked Probability Score |
| A×CRPS | Matrix-A PCA-coefficient CRPS cell (operational Uncertainty Head) |
| MLP | Multi-Layer Perceptron |
| SST / SSS / SSH | Sea Surface Temperature / Salinity / Height |
| ONI / RONI | Oceanic Niño Index / Relative Oceanic Niño Index |
| MLD | Mixed Layer Depth |
| D26 | Depth of the 26 °C isotherm |
| ENCE | Expected Normalized Calibration Error |
| OI | Optimal Interpolation |
| HYCOM | HYbrid Coordinate Ocean Model |
| H | Observation operator (1 m → HYCOM layers) |
| R | Observation-error covariance |
| Dai σₒ | Floored per-layer RMSE after H (ingest scale) |
| TSIS | DA / TSIS ingest cycle |
| CDF | Cumulative Distribution Function |
| SLA | Sea Level Anomaly (steric context elsewhere) |

---

## Edit checklist for authors

1. Edit slide text in the JSON panel, then **Save HTML** to bake into the file.
2. Set `"layout": "a"|"b"|"c"` per slide; omit to use `defaultLayout` (**c**).
3. Keep equations as KaTeX strings in `"equation"`.
4. Keep diagrams keyed by `"diagram"` id (`cover`, `encoder`, `fusion`, `mu_sigma`, `crps`, `pca_map`, `heave_head`, `warp_knots`, `unwarp`, `da`, `tradeoffs`). Put `{{diagram}}` in `"body"` to sandwich the figure in the text flow.
5. Update this `notes.md` in the same PR/session as content changes.
