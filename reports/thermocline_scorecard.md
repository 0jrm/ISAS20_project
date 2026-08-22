# Thermocline scorecard

evalphys **1.2.0**. Source: `../data/cache/train_ready_3adcff404b0b.pkl`. test n=623.

LC steric gate: 2 cm RMS in 24–28°N, 88–84°W. HYCOM 41-layer means use reference-H from the 2024-01-05 18Z drifted GOMb0.04 interfaces (`/unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project/NeSPReSO2_onTemplate/data/hycom/interfaces_20240105_18Z.json`), not live thknss.

## Models

- **A_CRPS**: D26 RMSE 19.05 m; 50–200 T RMSE 1.215 (heave-aligned 1.102, heave fraction 0.18); T RMSE 0–50/50–200/200–800 1.14669857964303/1.2153574495267272/0.6581226342343112; N² profile viol@1e-8 0.3852327447833066; LC steric RMS 181116.36598454986 cm pass=False
- **HeaveFast**: D26 RMSE 18.46 m; 50–200 T RMSE 1.174 (heave-aligned 1.034, heave fraction 0.22); T RMSE 0–50/50–200/200–800 1.1216622034137511/1.174108216909109/0.6534044639007366; N² profile viol@1e-8 0.9919743178170144; LC steric RMS 181118.5147300038 cm pass=False
- **Heave**: D26 RMSE 19.85 m; 50–200 T RMSE 1.205 (heave-aligned 1.066, heave fraction 0.22); T RMSE 0–50/50–200/200–800 1.1342323763593638/1.2054541857096392/0.7038554276607999; N² profile viol@1e-8 0.9935794542536116; LC steric RMS 181119.71879737708 cm pass=False
- **Latent**: D26 RMSE 19.50 m; 50–200 T RMSE 1.170 (heave-aligned 1.095, heave fraction 0.12); T RMSE 0–50/50–200/200–800 1.0698770034320244/1.1700979040523187/0.6519700108225647; N² profile viol@1e-8 1.0; LC steric RMS 181116.16018403517 cm pass=False
- **Direct**: D26 RMSE 19.03 m; 50–200 T RMSE 1.189 (heave-aligned 1.112, heave fraction 0.13); T RMSE 0–50/50–200/200–800 1.1108872970197492/1.1893363390762697/0.6174098596708294; N² profile viol@1e-8 1.0; LC steric RMS 181140.97239345082 cm pass=False
- **ISOP**: skipped (no joint_eof_meta)

## HYCOM 41-layer RMSE after H (vs Argo)

- **A_CRPS**: 50–200 m zmid mean RMSE T 1.252 °C, S 0.170 psu; zmid>800 T 0.172 °C. Full 41-layer vectors are in the JSON.
- **HeaveFast**: 50–200 m zmid mean RMSE T 1.198 °C, S 0.176 psu; zmid>800 T 0.163 °C. Full 41-layer vectors are in the JSON.
- **Heave**: 50–200 m zmid mean RMSE T 1.230 °C, S 0.183 psu; zmid>800 T 0.169 °C. Full 41-layer vectors are in the JSON.
- **Latent**: 50–200 m zmid mean RMSE T 1.208 °C, S 0.197 psu; zmid>800 T 0.159 °C. Full 41-layer vectors are in the JSON.
- **Direct**: 50–200 m zmid mean RMSE T 1.226 °C, S 0.371 psu; zmid>800 T 0.171 °C. Full 41-layer vectors are in the JSON.

## T1 reconstruction ceilings (truth through the representation)

- **pca16**: D26 RMSE 8.33 m; 50–200 T RMSE 0.116; heave fraction 0.00
- **warp_clim_true_landmarks**: D26 RMSE 61.00 m; 50–200 T RMSE 3.438; heave fraction 0.55
- **gem_sla**: D26 RMSE 26.82 m; 50–200 T RMSE 1.396; heave fraction 0.26

If warp-clim ceiling << PCA-16 in 50–200 m, landmark heave is the missing degree of freedom.

