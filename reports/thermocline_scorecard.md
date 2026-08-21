# Thermocline scorecard

evalphys **1.2.0**. Source: `../data/cache/train_ready_3adcff404b0b.pkl`. test n=623.

LC steric gate: 2 cm RMS in 24–28°N, 88–84°W. HYCOM 41-layer means: skipped (no interface file).

## Models

- **A_CRPS**: D26 RMSE 19.05 m; 50–200 T RMSE 1.215 (heave-aligned 1.102, heave fraction 0.18); T RMSE 0–50/50–200/200–800 1.14669857964303/1.2153574495267272/0.6581226342343112; N² profile viol@1e-8 0.3852327447833066; LC steric RMS 181116.36598454986 cm pass=False
- **Heave_best**: D26 RMSE 19.16 m; 50–200 T RMSE 1.248 (heave-aligned 1.255, heave fraction 0.00); T RMSE 0–50/50–200/200–800 1.8113029720696108/1.2483135952712876/0.6362750288642824; N² profile viol@1e-8 0.9983948635634029; LC steric RMS 181115.01183696432 cm pass=False
- **Heave_1**: D26 RMSE 18.98 m; 50–200 T RMSE 1.241 (heave-aligned 1.190, heave fraction 0.08); T RMSE 0–50/50–200/200–800 1.7043801169600674/1.241023685167563/0.644367302013291; N² profile viol@1e-8 0.9919743178170144; LC steric RMS 181116.2505883726 cm pass=False
- **ISOP**: skipped (no joint_eof_meta)

## T1 reconstruction ceilings (truth through the representation)

- **pca16**: D26 RMSE 8.33 m; 50–200 T RMSE 0.116; heave fraction 0.00
- **warp_clim_true_landmarks**: D26 RMSE 61.00 m; 50–200 T RMSE 3.438; heave fraction 0.55
- **gem_sla**: D26 RMSE 26.82 m; 50–200 T RMSE 1.396; heave fraction 0.26

If warp-clim ceiling << PCA-16 in 50–200 m, landmark heave is the missing degree of freedom.

