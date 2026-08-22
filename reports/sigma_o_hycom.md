# 41-layer Dai σ_o after H

A×CRPS is the frozen Phase 6 cell. HeaveFast is the v2 challenger. conv3, ops, bathy, and bathy_wind use the same H, floors, and chrono test on their own caches. They are ablations, not ingest, unless they beat HeaveFast on thermocline σ_T. The ingest file is this 41-layer table, not 1 m RMSE and not dense Σ. Markdown shows σ_T. σ_S is in the csv/json.

H is reference-H from the 2024-01-05 18Z drifted GOMb0.04 background (9 GDAC columns plus mean p_ifc). It is not live thknss. Label is `h_kind=reference`.

Floors are 0.05 °C and 0.02 psu (Argo analysis limits). σ = max(layer RMSE, floor).

CRPS-as-σ_o is deferred until ENCE < 0.20 by band. A physical ENCE(T) = 0.236. HeaveFast ENCE(σ_D26) = 0.52.

Interfaces: `/unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project/NeSPReSO2_onTemplate/data/hycom/interfaces_20240105_18Z.json`.

## Thermocline layers (zmid 50–200 m)

| k | zmid_m | A_CRPS mean σ_T | HeaveFast σ_T | conv3 σ_T | ops σ_T | bathy σ_T | bathy_wind σ_T |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 56.3 | 1.309 | 1.255 | 1.258 | 1.208 | 1.238 | 1.231 |
| 11 | 64.3 | 1.321 | 1.255 | 1.288 | 1.232 | 1.286 | 1.257 |
| 12 | 72.3 | 1.332 | 1.259 | 1.308 | 1.256 | 1.322 | 1.289 |
| 13 | 80.3 | 1.335 | 1.268 | 1.319 | 1.269 | 1.350 | 1.315 |
| 14 | 88.3 | 1.314 | 1.252 | 1.305 | 1.264 | 1.346 | 1.311 |
| 15 | 96.3 | 1.297 | 1.239 | 1.293 | 1.264 | 1.338 | 1.303 |
| 16 | 104.3 | 1.273 | 1.223 | 1.272 | 1.252 | 1.322 | 1.287 |
| 17 | 112.4 | 1.255 | 1.213 | 1.256 | 1.236 | 1.310 | 1.273 |
| 18 | 124.0 | 1.210 | 1.178 | 1.218 | 1.200 | 1.276 | 1.230 |
| 19 | 138.3 | 1.147 | 1.134 | 1.171 | 1.158 | 1.221 | 1.163 |
| 20 | 157.2 | 1.095 | 1.098 | 1.138 | 1.117 | 1.183 | 1.113 |
| 21 | 198.6 | 1.014 | 1.004 | 1.054 | 1.022 | 1.098 | 1.022 |

## Loop Current vs complement (zmid 50–200 m)

LC is the evalphys box 24–28°N, 88–84°W. Complement is the rest of the chrono test. Shelf is omitted. This cache has no `bottom_depth`.

- A_CRPS s42 lc n=240: thermocline mean σ_T 1.072 °C (native 50–200 T 1.096 °C)
- A_CRPS s42 complement n=383: thermocline mean σ_T 1.347 °C (native 50–200 T 1.285 °C)
- A_CRPS s43 lc n=240: thermocline mean σ_T 1.047 °C (native 50–200 T 1.075 °C)
- A_CRPS s43 complement n=383: thermocline mean σ_T 1.313 °C (native 50–200 T 1.256 °C)
- A_CRPS s44 lc n=240: thermocline mean σ_T 1.064 °C (native 50–200 T 1.093 °C)
- A_CRPS s44 complement n=383: thermocline mean σ_T 1.348 °C (native 50–200 T 1.286 °C)
- HeaveFast s42 lc n=240: thermocline mean σ_T 1.051 °C (native 50–200 T 1.075 °C)
- HeaveFast s42 complement n=383: thermocline mean σ_T 1.277 °C (native 50–200 T 1.232 °C)
- conv3 s42 lc n=240: thermocline mean σ_T 1.109 °C (native 50–200 T 1.101 °C)
- conv3 s42 complement n=383: thermocline mean σ_T 1.315 °C (native 50–200 T 1.279 °C)
- ops s42 lc n=240: thermocline mean σ_T 1.024 °C (native 50–200 T 1.051 °C)
- ops s42 complement n=383: thermocline mean σ_T 1.304 °C (native 50–200 T 1.260 °C)
- bathy s42 lc n=240: thermocline mean σ_T 1.200 °C (native 50–200 T 1.197 °C)
- bathy s42 complement n=383: thermocline mean σ_T 1.317 °C (native 50–200 T 1.284 °C)
- bathy_wind s42 lc n=240: thermocline mean σ_T 1.127 °C (native 50–200 T 1.132 °C)
- bathy_wind s42 complement n=383: thermocline mean σ_T 1.291 °C (native 50–200 T 1.245 °C)

## Full 41-layer σ_o

| k | zmid_m | A_CRPS mean σ_T | HeaveFast σ_T | conv3 σ_T | ops σ_T | bathy σ_T | bathy_wind σ_T |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.5 | 1.054 | 1.036 | 1.042 | 1.051 | 1.057 | 1.046 |
| 1 | 1.9 | 1.053 | 1.035 | 1.041 | 1.049 | 1.055 | 1.045 |
| 2 | 4.4 | 1.044 | 1.024 | 1.032 | 1.034 | 1.042 | 1.036 |
| 3 | 8.4 | 1.048 | 1.022 | 1.035 | 1.028 | 1.039 | 1.039 |
| 4 | 13.2 | 1.054 | 1.022 | 1.050 | 1.034 | 1.045 | 1.052 |
| 5 | 18.6 | 1.066 | 1.035 | 1.080 | 1.055 | 1.063 | 1.089 |
| 6 | 24.9 | 1.108 | 1.077 | 1.131 | 1.103 | 1.095 | 1.146 |
| 7 | 32.3 | 1.190 | 1.160 | 1.208 | 1.171 | 1.158 | 1.216 |
| 8 | 40.3 | 1.275 | 1.236 | 1.242 | 1.217 | 1.192 | 1.257 |
| 9 | 48.3 | 1.300 | 1.255 | 1.228 | 1.202 | 1.197 | 1.234 |
| 10 | 56.3 | 1.309 | 1.255 | 1.258 | 1.208 | 1.238 | 1.231 |
| 11 | 64.3 | 1.321 | 1.255 | 1.288 | 1.232 | 1.286 | 1.257 |
| 12 | 72.3 | 1.332 | 1.259 | 1.308 | 1.256 | 1.322 | 1.289 |
| 13 | 80.3 | 1.335 | 1.268 | 1.319 | 1.269 | 1.350 | 1.315 |
| 14 | 88.3 | 1.314 | 1.252 | 1.305 | 1.264 | 1.346 | 1.311 |
| 15 | 96.3 | 1.297 | 1.239 | 1.293 | 1.264 | 1.338 | 1.303 |
| 16 | 104.3 | 1.273 | 1.223 | 1.272 | 1.252 | 1.322 | 1.287 |
| 17 | 112.4 | 1.255 | 1.213 | 1.256 | 1.236 | 1.310 | 1.273 |
| 18 | 124.0 | 1.210 | 1.178 | 1.218 | 1.200 | 1.276 | 1.230 |
| 19 | 138.3 | 1.147 | 1.134 | 1.171 | 1.158 | 1.221 | 1.163 |
| 20 | 157.2 | 1.095 | 1.098 | 1.138 | 1.117 | 1.183 | 1.113 |
| 21 | 198.6 | 1.014 | 1.004 | 1.054 | 1.022 | 1.098 | 1.022 |
| 22 | 256.9 | 0.891 | 0.905 | 0.971 | 0.924 | 0.993 | 0.931 |
| 23 | 323.7 | 0.792 | 0.831 | 0.893 | 0.836 | 0.907 | 0.858 |
| 24 | 420.3 | 0.690 | 0.710 | 0.744 | 0.698 | 0.771 | 0.725 |
| 25 | 527.2 | 0.582 | 0.567 | 0.575 | 0.543 | 0.598 | 0.565 |
| 26 | 625.2 | 0.456 | 0.435 | 0.439 | 0.415 | 0.448 | 0.424 |
| 27 | 742.0 | 0.339 | 0.318 | 0.315 | 0.302 | 0.317 | 0.304 |
| 28 | 861.8 | 0.250 | 0.236 | 0.234 | 0.224 | 0.241 | 0.231 |
| 29 | 962.5 | 0.208 | 0.200 | 0.200 | 0.191 | 0.208 | 0.198 |
| 30 | 1061.8 | 0.188 | 0.184 | 0.181 | 0.175 | 0.189 | 0.181 |
| 31 | 1192.2 | 0.150 | 0.147 | 0.144 | 0.140 | 0.152 | 0.145 |
| 32 | 1963.3 | 0.050 | 0.050 | 0.050 | 0.050 | 0.050 | 0.050 |
| 33 | 2791.5 | n/a | n/a | n/a | n/a | n/a | n/a |
| 34 | 2927.8 | n/a | n/a | n/a | n/a | n/a | n/a |
| 35 | 2927.8 | n/a | n/a | n/a | n/a | n/a | n/a |
| 36 | 2927.8 | n/a | n/a | n/a | n/a | n/a | n/a |
| 37 | 2927.8 | n/a | n/a | n/a | n/a | n/a | n/a |
| 38 | 2927.8 | n/a | n/a | n/a | n/a | n/a | n/a |
| 39 | 2927.8 | n/a | n/a | n/a | n/a | n/a | n/a |
| 40 | 2927.8 | n/a | n/a | n/a | n/a | n/a | n/a |

## Native 1 m hydrography (not R)

These numbers compare hydrography to the DA σ_o table. They are not the ingest file.

- A_CRPS s42: 50–200 m T RMSE 1.215 °C, D26 RMSE 19.05 m, n=623
- A_CRPS s43: 50–200 m T RMSE 1.189 °C, D26 RMSE 18.91 m, n=623
- A_CRPS s44: 50–200 m T RMSE 1.216 °C, D26 RMSE 18.77 m, n=623
- HeaveFast s42: 50–200 m T RMSE 1.174 °C, D26 RMSE 18.46 m, n=623
- conv3 s42: 50–200 m T RMSE 1.214 °C, D26 RMSE 18.64 m, n=623
- ops s42: 50–200 m T RMSE 1.184 °C, D26 RMSE 18.27 m, n=623
- bathy s42: 50–200 m T RMSE 1.251 °C, D26 RMSE 20.80 m, n=623
- bathy_wind s42: 50–200 m T RMSE 1.203 °C, D26 RMSE 19.84 m, n=623

## Deep layers

- A_CRPS deep (zmid>800) σ_T is at the 0.05 °C floor on at least one layer. It is not the random-split v1 0.013.
- HeaveFast deep (zmid>800) σ_T is at the 0.05 °C floor on at least one layer.
- conv3 deep (zmid>800) σ_T is at the 0.05 °C floor on at least one layer.
- ops deep (zmid>800) σ_T is at the 0.05 °C floor on at least one layer.
- bathy deep (zmid>800) σ_T is at the 0.05 °C floor on at least one layer.
- bathy_wind deep (zmid>800) σ_T is at the 0.05 °C floor on at least one layer.

