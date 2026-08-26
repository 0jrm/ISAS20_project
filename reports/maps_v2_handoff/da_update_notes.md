# MAPS lab update · NeSPReSO DA loops · companion notes

**How to open.** Open [`da_update.html`](da_update.html) in a browser (KaTeX needs network once for the CDN). Use **JSON editor** to edit the deck live. Default layout is **A** (bracket map). Movies and stills live under [`media/`](media/). ML architecture deck (present after this one): [`index.html`](index.html).

**How to make JSON edits permanent.** Click **Save HTML**. Prefer Chromium/Chrome: pick the existing deck file to overwrite (same dialog reads + writes; later saves in that tab reuse it). If the browser has no file-system picker (some embedded viewers), you will be asked to choose the existing `.html`, then a baked download appears — replace the on-disk file with that download. Reload to confirm.

**Audience.** Lab peers (grad CS, mixed backgrounds) and advisor. ~15 minutes. Progress update, not an OSSE skill claim.

**Color legend (fixed for the whole deck).** Same mapping on every slide, code bracket, and diagram.

| Color | Accent key | Means |
|-------|------------|--------|
| Green | `encoder` | NeSPReSO observation path (API → masked `tsis_obs`) |
| Amber | `da` | TSIS / DA cycle / experiment design |
| Indigo | `heave` | HYCOM dynamics (freerun, `incflg` apply, forecast, 05.3 ownership) |
| Red | `uncertainty` | Scores / Argo RMSE / verification |

When the speaker says “highlighted by the green text,” that always means NeSPReSO obs. Amber always means TSIS/DA. Indigo always means HYCOM. Red always means scores.

**Sources (numbers only from these).**

- [`MAPS-handoff-drift-loop.html`](../../MAPS-handoff-drift-loop.html) verification + mission sections (`da_loop_20240106`)
- Aug 20 packet `speaker-notes.md` / `README.txt` inside `HYCOM-TSIS-NeSPReSO-update-20260820.zip`
- Media under `media/` (unpacked from that zip)

No invented OSSE thresholds. Do not call this an OSSE.

---

## Slide 1 · cover

This is a lab progress update on Aim 2’s assimilation loop. Leave knowing (1) the pipe is owned, (2) traditional TSIS on a drifted Gulf beats a free run, (3) profile-only NeSPReSO did not. Open questions at the end are real asks for the room. Present the ML deck (`index.html`) after this talk if you want SatEncoder / A×CRPS / Heave.

---

## Slide 2 · bottom_line

Three facts, color-coded:

1. **Indigo · pipe.** Stage restart → TSIS → xa2inc → `incflg=-1` → week forecast. Apply logs on drifted DA members: 97 `finished incupdate`, 0 `neg dp`.
2. **Amber · ops wins.** Drifted analysis-time Argo T RMSE: null 0.912 → ops 0.496 °C. Ops still ahead at +168 h (0.748 vs null 0.853). Source: drift-loop verification table.
3. **Red · NeSPReSO loses.** Same table: nespreso_argo 1.414, nespreso_tracks 1.815 at analysis. Both worse than null.

Extra ownership fact (not DA skill): V08 HYCOM matches abozec expt 05.3 on the first archive byte-for-byte.

**Nuance.** Speaker-notes slide 17 once wrote tracks analysis RMSE as 1.420. Drift-loop mission + verification table both say **1.815**. This deck pins **1.815**.

---

## Slide 3 · three_machines

Analogy for mixed-CS audience:

- **Green NeSPReSO** estimates profile-like T/S where floats are missing (observation operator role).
- **Amber TSIS** is the statistical analysis (xprep → xgmrf → xa2inc). It writes an increment, not a new ocean.
- **Indigo HYCOM** is the dynamical model that must absorb the patch. Messy increments show up as spin-up the next day.

If any one of the three is wrong, the loop fails for a different reason. Today’s failure mode for NeSPReSO is on the observation side, not apply.

---

## Slide 4 · daily_loop

Read nested brackets outside-in: indigo HYCOM background, amber TSIS chain inside, indigo apply.

Order on the diagram: **bg → xprep → xgmrf → xa2inc → apply**. Those arrows are warranted; that is the owned daily chain. Critical switch: `incflg=-1` means “apply difference archive.” `incflg=+1` treated the file as a target state and produced `neg dp` (negative layer thickness) on early attempts. xa2inc files are small-amplitude diffs (a few degrees), not full T fields.

---

## Slide 5 · campaigns

| Campaign | Date story | Point |
|----------|------------|--------|
| 2 Jan twins | Analysis +6 h after reanalysis restart | Pipe works; ops ≺ null ≺ NeSPReSO; too close to IC |
| 6 Jan drifted | 4 d freerun, then TSIS, then 7 d forecast | Same ranking on a wandering background |
| 05.3 canary | V08 vs abozec 05.3 | Config ownership; first `archm` bit-identical |

**Drifted members.** `null` (no increment), `ops` (cargo Argo + L3/L4 SLA + L4 SST via xprep), `nespreso_argo` (9 NeSPReSO columns at 6 Jan GDAC sites), `nespreso_tracks` (NeSPReSO on 0.25° thinning of 4–5 Jan L3 SLA tracks). Shared `restart_day04`. Innovations use the drifted column.

**Explicit non-goals (drift).** No L3 SST. No velocity increments (even ops). No L4 SST/SSH skill score. Not an OSSE. Does not overwrite `da_loop_20240102`.

---

## Slide 6 · obs_path

**Green path (NeSPReSO).** `build_obs.py` reads drifted archive, hits profile/grid APIs, layer-averages onto HYCOM interfaces, masks satellites to `SPVAL`, profiles-only gates. Skips xprep.

**Amber path (ops).** No 6 Jan qcobs file → xprep from `ocn_obs`. Cargo window ±4 d for profiles/SLA; L4 SST analysis day only.

**Mechanical fact.** NeSPReSO increment SSH and velocity are identically zero. Ops SSH increment RMS = 0.075 (drift handoff / speaker-notes Backup A).

Still shown: `media/inc_nespreso_argo_tem.png` (2 Jan T increment; SSH maps blank on purpose).

---

## Slide 7 · headlines

Speak the ranking out loud. Profile counts 8–11/day. Do not overclaim percent OSSE thresholds from that. Ranking is enough.

**Drifted analysis (°C):** ops 0.496 · null 0.912 · nespreso_argo 1.414 · nespreso_tracks 1.815.

**2 Jan contrast (°C, speaker-notes slide 14):** null 1.03 · ops 0.64 · nespreso_argo 2.34 · dense 4.36.

Figure `media/argo_rmse_T.png` is the 2 Jan bar chart from the packet.

---

## Slide 8 · rmse_table

Full drifted table (MAPS-handoff-drift-loop verification):

| Member | anl | +24 h | +48 h | +168 h | incupd lines |
|--------|-----|-------|-------|--------|--------------|
| ops | 0.496 | 0.667 | 0.510 | 0.748 | 97 |
| null | 0.912 | 0.840 | 0.783 | 0.853 | 0 |
| nespreso_argo | 1.414 | 0.850 | 0.771 | 0.849 | 97 |
| nespreso_tracks | 1.815 | 1.784 | 1.681 | 1.553 | 97 |

Counts 9 / 11 / 8 / 10. Analysis-time T bias vs Argo: null −0.26, ops −0.18, argo −0.44, tracks −0.78 °C.

**In-sample caveat.** 6 Jan Argo is in-sample for nespreso_argo (those 9 sites) and for ops (cargo ±4 d). Only +168 h (12 Jan) is outside ops `p_twin_end=4`.

---

## Slide 9 · movie

Play `media/drift_vs_null.mp4` (264 hourly frames). Rows SST/SSS/SSH. Absolute: null, ops, argo, tracks. Diffs: member − ops. First 96 h shared freerun (diffs zero). Apply from 5 Jan 19Z.

**Inference callout.** tracks−ops SSH can look basin-scale wrong even though `δSSH_inc ≡ 0`. That is steric/thickness response, not a written SSH patch. Label it inference, not a TSIS SSH increment.

Stills also in `media/still_*.png` if the room prefers posters over video.

---

## Slide 10 · next

**Learned.** Traditional DA works on this drifted ocean. Profile-only NeSPReSO does not beat null. More columns (dense / tracks) made Argo worse. Apply is healthy. Bottleneck is obs quality (cold bias, error model, missing SSH).

**Ask.**

1. Error model. Native NeSPReSO vs those nine 6 Jan GDAC profiles was not recomputed in the drifted campaign.
2. SSH. Write SSH, keep satellites on, or stay profile-only until T/S is honest?
3. Ingest. Direct NetCDF surgery is a test hook; production Class 3 still wants xprep to write profiles.
4. OSSE gate. No nature run until obs quality is why we win or lose.

**Next.** Score native NeSPReSO vs GDAC on drifted dates · error model / vertical remap (not another dense grid) · optional SSH/hybrid once T/S stops hurting floats · keep masterB (DA, 92 ranks) and V08 (05.3, 189) separate · OSSE only after beating ops or a documented reason not to.

---

## Backup · how big was the correction?

Plain-language takeaway: ops actually wrote a surface-height fix plus a modest T patch. NeSPReSO wrote T/S only (SSH and velocity zero). The patches applied cleanly; the science loss is not “HYCOM rejected the increment.”

| Member | SSH rms | T rms | reading |
|--------|---------|-------|---------|
| ops | 0.075 | 0.070 | has SSH + T |
| nespreso_argo | 0 | 0.047 | T/S only |
| nespreso_tracks | 0 | 0.167 | T/S only, larger T |

Thickness max abs ~3e6 pressure units; `onem ≈ 9806` ≈ 1 m seawater. Treat thickness as untrusted until wet-only stats.

---

## Backup · backup_053

V08 free-run canary. First 6 h `archm` bit-identical to 05.3. Month-3 interior SST RMS 0.17 K, SSH 2.8 cm is expected mesoscale wander. Movie: `media/3mo_fromic_owned_vs_053.mp4` (SSH in meters as `srfhgt/9.806`, Δ ±5 cm). Frame 1 difference should look empty. A southern nest stripe on frame 1 is the old Montgomery/namelist bug, not mesoscale.

Do not mix masterB and V08. Do not strip `hybthn` to force masterB onto 05.3 `blkdat`.

---

## Acronym sheet (first-use expansions)

| Short | Full |
|-------|------|
| NeSPReSO | Neural Sparse Profiling of the Subsurface Ocean |
| DA | Data Assimilation |
| TSIS | analysis system (xprep / xgmrf / xa2inc path here) |
| HYCOM | HYbrid Coordinate Ocean Model |
| ARGO / GDAC | profiling floats / Global Data Assembly Centre |
| RMSE | Root Mean Square Error |
| SLA / SST / SSS / SSH | Sea Level Anomaly / Sea Surface Temperature / Salinity / Height |
| OSSE | Observing System Simulation Experiment |
| IC | Initial Condition |
| ops | traditional cargo-obs analysis member |

---

## Edit checklist

1. Edit slide text in the JSON panel, then **Save HTML** to bake into the file.
2. Keep `"layout": "a"|"b"|"c"`; omit to use `defaultLayout` (**a**).
3. For RIR-style nested brackets set `"codeHtml": true` and put `.bkt.bkt-*` HTML in `"code"`.
4. Equations are KaTeX strings in `"equation"`.
5. Media: `"media": { "type": "video"|"img", "src": "media/...", "caption": "..." }` (takes the side panel when set).
6. Update this file in the same session as content changes.
