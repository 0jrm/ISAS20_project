# Thermocline scorecard

evalphys **1.2.0**. Source: `../data/cache/train_ready_3adcff404b0b.pkl`. test n=623.

LC steric gate: 2 cm RMS in 24–28°N, 88–84°W. HYCOM 41-layer means: skipped (no interface file).

## Models

- **A_CRPS**: D26 RMSE 19.05 m; 50–200 T RMSE 1.215 (heave-aligned 1.102, heave fraction 0.18); LC steric RMS 181116.36598454986 cm pass=False
- **ISOP**: skipped (no joint_eof_meta)

## T1 reconstruction ceilings (truth through the representation)

- **pca16**: D26 RMSE 8.33 m; 50–200 T RMSE 0.116; heave fraction 0.00
- **warp_clim_true_landmarks**: D26 RMSE 61.00 m; 50–200 T RMSE 3.438; heave fraction 0.55
- **gem_sla**: D26 RMSE 26.82 m; 50–200 T RMSE 1.396; heave fraction 0.26

If warp-clim ceiling << PCA-16 in 50–200 m, landmark heave is the missing degree of freedom.

