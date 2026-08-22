# Where H comes from (GOMb0.04 TSIS ingest)

Pick **C**, not B. Do not stub (D) unless this zip cannot land in the v2 tree.

## What H is

TSIS does not see NeSPReSO's 1801 x 1 m levels. Ingest already remaps with `layer_sample` in

`hycom/science/tsis_gom_da/nespreso_loop/hycom_io.py`

copied here as `h_operator.py`.

```
dp_m, p_ifc = interfaces_m(thknss[:, j, i])   # 42 interfaces, metres
T_k = layer_sample(z_1m, T_1m, p_ifc)        # 41 layer means
```

`thknss` is the HYCOM background archive field (pressure thickness). `ONEM = 9806`. Interfaces are `cumsum(thknss/ONEM)` from the surface. They move in x, y, and time. That is the hybrid coordinate.

This is the H that built `tsis_obs_*_nespreso_argo.nc` on 6 Jan. It is also why 50-200 m died: a sharp 1 m thermocline averaged across a thick layer is not the profile in the paper.

## The options

**A.** `blkdat.input` is not an interface file. It has `kdm=41`, `nsigma=14`, and 41 target densities (`sigma`). Those densities define the hybrid vertical coordinate. They are not z. Putting `blkdat` in `io.hycom_interfaces` would teach the scorecard the wrong H.

**B.** There is no documented GOMb0.04 41-level z table that matches ingest. A fixed z grid is a different operator than TSIS. Using one for Dai RMSE will not predict the 4 °C thermocline failure.

**C.** Use this operator. For live DA, `p_ifc` comes from the background `archv` `thknss` column at the cast. For the v2 scorecard without HYCOM I/O, point `io.hycom_interfaces` at `interfaces_20240105_18Z.json` in this zip. That file is the actual 6 Jan drifted-background interfaces at the 9 GDAC sites, plus a mean `p_ifc` for a single-column scorecard. Label it reference-H, not climatology-H.

**D.** Only if the zip cannot be committed. Then keep skipping 41-layer means. Do not export Dai σ_o on 1 m or on a 60-level OSSE grid and call it TSIS R.

## v2 wiring

```yaml
io:
  hycom_interfaces: path/to/interfaces_20240105_18Z.json
```

Scorecard should call `h_operator.layer_sample(z, T, p_ifc)` per column. If a cast has its own `p_ifc` in the JSON, use that. Else use `scorecard_reference_p_ifc`. Dai σ_o for TSIS is RMSE of those 41 layer means versus Argo remapped with the same H, floored at 0.05 °C / 0.02 psu.

Do not treat `scorecard_reference_p_ifc` as the DA operator for a Loop Current OSSE. Re-extract `thknss` from the nature-run / background archive at each virtual-cast column.
