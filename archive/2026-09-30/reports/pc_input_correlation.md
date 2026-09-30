Historical. Not instructions.

# PC–Input Correlation Report

N = 4,145 samples · 32 inputs · 64 PCs (32 T + 32 S)
Bonferroni threshold: α=0.01/(32×64) = 4.88e-06

## Top |r| ≥ 0.3 (Bonferroni-significant)

| Input | PC | r | p |
|-------|-----|-----|-----|
| ssh | T_PC1 | +0.968 | 0.0e+00 |
| ssh | S_PC1 | +0.960 | 0.0e+00 |
| sst | T_PC2 | +0.845 | 0.0e+00 |
| timecos | T_PC2 | +0.829 | 0.0e+00 |
| sss | S_PC2 | +0.741 | 0.0e+00 |
| ssh.laplacian@1deg | S_PC1 | -0.718 | 0.0e+00 |
| ssh.laplacian@1deg | T_PC1 | -0.715 | 0.0e+00 |
| loncos | S_PC1 | +0.355 | 4.4e-123 |
| loncos | T_PC1 | +0.348 | 1.7e-118 |
| timecos | T_PC3 | -0.340 | 9.4e-113 |
| sst | T_PC3 | -0.321 | 6.5e-100 |
| sst | S_PC4 | +0.310 | 4.8e-93 |
| timesin | T_PC4 | +0.306 | 1.3e-90 |

## Strong correlates per input (|r| ≥ 0.5, p < Bonferroni)

- **timecos**: T_PC2(r=+0.83)
- **sss**: S_PC2(r=+0.74)
- **sst**: T_PC2(r=+0.84)
- **ssh**: T_PC1(r=+0.97), S_PC1(r=+0.96)
- **ssh.laplacian@1deg**: S_PC1(r=-0.72), T_PC1(r=-0.72)

## All 32 T PCs — top-3 inputs (no |r| cutoff)

| PC | max|r| | best input | 2nd | 3rd |
|----|--------|------------|-----|-----|
| T_PC1 | 0.968 | ssh(+0.968) | ssh.laplacian@1deg(-0.715) | loncos(+0.348) |
| T_PC2 | 0.845 | sst(+0.845) | timecos(+0.829) | sss(-0.284) |
| T_PC3 | 0.340 | timecos(-0.340) | sst(-0.321) | ssh.geo_u@local(+0.193) |
| T_PC4 | 0.306 | timesin(+0.306) | lonsin(+0.176) | latsin(-0.167) |
| T_PC5 | 0.262 | timesin(+0.262) | latcos(-0.195) | latsin(+0.193) |
| T_PC6 | 0.161 | sst.grad_y@1deg(-0.161) | sss.grad_y@1deg(-0.150) | lonsin(-0.149) |
| T_PC7 | 0.151 | timesin(+0.151) | sst.tendency@7d(-0.097) | sss.grad_y@1deg(+0.082) |
| T_PC8 | 0.124 | bathy(+0.124) | wind_speed(+0.099) | ssh.laplacian@1deg(-0.090) |
| T_PC9 | 0.129 | bathy(-0.129) | wind_speed(+0.084) | ssh.geo_v@1deg(+0.057) |
| T_PC10 | 0.101 | lonsin(-0.101) | sst.grad_y@1deg(-0.076) | ssh.geo_u@1deg(-0.061) |
| T_PC11 | 0.103 | bathy(+0.103) | lonsin(-0.086) | sst.tendency@7d(+0.068) |
| T_PC12 | 0.095 | wind_speed(-0.095) | sss(-0.055) | timesin(-0.054) |
| T_PC13 | 0.072 | bathy(+0.072) | sst.tendency@7d(+0.048) | timesin(-0.038) |
| T_PC14 | 0.066 | sst.tendency@7d(+0.066) | loncos(+0.055) | timesin(-0.054) |
| T_PC15 | 0.091 | lonsin(-0.091) | bathy(+0.088) | sst.grad_y@1deg(-0.066) |
| T_PC16 | 0.083 | bathy(-0.083) | wind_u(+0.056) | wind_speed(-0.050) |
| T_PC17 | 0.067 | bathy(-0.067) | lonsin(+0.064) | sst.grad_x@1deg(+0.059) |
| T_PC18 | 0.070 | bathy(-0.070) | sst.tendency@7d(+0.063) | wind_speed(-0.060) |
| T_PC19 | 0.070 | wind_speed(+0.070) | sst.grad_x@1deg(+0.061) | lonsin(+0.059) |
| T_PC20 | 0.069 | sst.grad_x@1deg(-0.069) | lonsin(-0.056) | sss.grad_x@local(+0.038) |
| T_PC21 | 0.069 | wind_speed(-0.069) | lonsin(-0.062) | latsin(+0.044) |
| T_PC22 | 0.112 | sst.grad_x@1deg(-0.112) | bathy(+0.079) | lonsin(-0.077) |
| T_PC23 | 0.068 | wind_speed(-0.068) | sst.tendency@7d(+0.047) | latsin(+0.036) |
| T_PC24 | 0.086 | lonsin(-0.086) | bathy(+0.080) | sst.grad_x@1deg(-0.070) |
| T_PC25 | 0.068 | wind_speed(+0.068) | latsin(-0.058) | latcos(+0.057) |
| T_PC26 | 0.067 | sst.grad_y@local(+0.067) | bathy(-0.062) | lonsin(+0.056) |
| T_PC27 | 0.059 | latsin(-0.059) | latcos(+0.058) | loncos(-0.047) |
| T_PC28 | 0.056 | ssh.grad_y@1deg(-0.056) | ssh.grad_y@local(-0.056) | ssh.geo_u@1deg(+0.055) |
| T_PC29 | 0.059 | latsin(+0.059) | latcos(-0.058) | loncos(+0.056) |
| T_PC30 | 0.067 | wind_speed(-0.067) | ssh.grad_y@local(+0.043) | ssh.geo_u@local(-0.042) |
| T_PC31 | 0.088 | latsin(-0.088) | latcos(+0.086) | lonsin(+0.072) |
| T_PC32 | 0.063 | latsin(+0.063) | latcos(-0.062) | loncos(+0.060) |

## All 32 S PCs — top-3 inputs (no |r| cutoff)

| PC | max|r| | best input | 2nd | 3rd |
|----|--------|------------|-----|-----|
| S_PC1 | 0.960 | ssh(+0.960) | ssh.laplacian@1deg(-0.718) | loncos(+0.355) |
| S_PC2 | 0.741 | sss(+0.741) | timecos(-0.243) | sss.grad_y@local(+0.216) |
| S_PC3 | 0.219 | ssh.geo_u@local(+0.219) | ssh.grad_y@local(-0.218) | ssh.geo_u@1deg(+0.216) |
| S_PC4 | 0.310 | sst(+0.310) | timecos(+0.239) | sss(-0.170) |
| S_PC5 | 0.233 | timesin(+0.233) | ssh.geo_v@local(+0.150) | ssh.grad_x@local(+0.150) |
| S_PC6 | 0.180 | latsin(+0.180) | latcos(-0.175) | timesin(+0.117) |
| S_PC7 | 0.203 | latcos(+0.203) | latsin(-0.194) | loncos(-0.129) |
| S_PC8 | 0.110 | timesin(-0.110) | sss(+0.068) | timecos(-0.064) |
| S_PC9 | 0.111 | loncos(+0.111) | sst(+0.067) | ssh.geo_u@local(+0.065) |
| S_PC10 | 0.081 | lonsin(-0.081) | sss(-0.063) | sss.grad_y@1deg(-0.058) |
| S_PC11 | 0.070 | sst.grad_y@1deg(-0.070) | sss.grad_y@local(-0.050) | latcos(-0.044) |
| S_PC12 | 0.097 | sst.grad_y@1deg(+0.097) | loncos(+0.066) | sss.grad_y@1deg(+0.043) |
| S_PC13 | 0.050 | sst.grad_y@1deg(-0.050) | sst.tendency@7d(+0.037) | sss.grad_y@1deg(-0.033) |
| S_PC14 | 0.055 | sss.grad_y@local(-0.055) | sst(-0.042) | timesin(+0.042) |
| S_PC15 | 0.042 | sst.tendency@7d(-0.042) | latsin(-0.037) | latcos(+0.035) |
| S_PC16 | 0.052 | sss.grad_y@local(-0.052) | timesin(-0.038) | wind_speed(-0.030) |
| S_PC17 | 0.083 | lonsin(-0.083) | bathy(+0.047) | wind_v(+0.047) |
| S_PC18 | 0.061 | lonsin(+0.061) | ssh.geo_u@local(+0.050) | ssh.geo_u@1deg(+0.049) |
| S_PC19 | 0.069 | sss.grad_y@1deg(-0.069) | sss.grad_x@local(+0.052) | sss.grad_y@local(-0.046) |
| S_PC20 | 0.060 | lonsin(+0.060) | bathy(-0.042) | sss.grad_y@1deg(+0.036) |
| S_PC21 | 0.029 | timesin(-0.029) | bathy(+0.024) | sss.grad_y@local(+0.023) |
| S_PC22 | 0.046 | wind_v(+0.046) | ssh.grad_y@local(-0.027) | ssh.geo_u@local(+0.025) |
| S_PC23 | 0.042 | sss.grad_x@local(+0.042) | wind_speed(-0.034) | sss.grad_y@1deg(-0.033) |
| S_PC24 | 0.061 | sss.grad_x@local(+0.061) | bathy(-0.045) | lonsin(+0.033) |
| S_PC25 | 0.058 | lonsin(-0.058) | ssh.geo_u@local(-0.042) | ssh.geo_u@1deg(-0.042) |
| S_PC26 | 0.037 | wind_v(+0.037) | lonsin(+0.033) | sst.grad_y@local(+0.033) |
| S_PC27 | 0.038 | ssh.tendency@7d(+0.038) | bathy(+0.030) | sss.grad_y@1deg(-0.027) |
| S_PC28 | 0.019 | wind_u(-0.019) | wind_speed(+0.018) | sss(-0.018) |
| S_PC29 | 0.052 | wind_speed(-0.052) | lonsin(-0.045) | loncos(-0.042) |
| S_PC30 | 0.030 | bathy(-0.030) | sst.grad_x@local(+0.029) | sst.grad_y@1deg(+0.024) |
| S_PC31 | 0.033 | ssh.geo_u@1deg(+0.033) | ssh.grad_y@1deg(-0.033) | ssh.geo_u@local(+0.031) |
| S_PC32 | 0.036 | bathy(-0.036) | ssh.geo_u@1deg(+0.031) | ssh.grad_y@1deg(-0.031) |

## Summary: max |r| per input across all PCs

| Input | max|r| | best PC |
|-------|--------|---------|
| timecos | 0.829 | T_PC2(r=+0.829) |
| timesin | 0.306 | T_PC4(r=+0.306) |
| latcos | 0.203 | S_PC7(r=+0.203) |
| latsin | 0.201 | S_PC2(r=-0.201) |
| loncos | 0.355 | S_PC1(r=+0.355) |
| lonsin | 0.246 | T_PC1(r=+0.246) |
| sss | 0.741 | S_PC2(r=+0.741) |
| sst | 0.845 | T_PC2(r=+0.845) |
| ssh | 0.968 | T_PC1(r=+0.968) |
| sst.grad_x@local | 0.142 | T_PC1(r=-0.142) |
| sst.grad_y@local | 0.096 | T_PC1(r=+0.096) |
| sst.grad_x@1deg | 0.179 | T_PC1(r=-0.179) |
| sst.grad_y@1deg | 0.161 | T_PC6(r=-0.161) |
| sss.grad_x@local | 0.149 | T_PC1(r=-0.149) |
| sss.grad_y@local | 0.216 | S_PC2(r=+0.216) |
| sss.grad_x@1deg | 0.207 | T_PC1(r=-0.207) |
| sss.grad_y@1deg | 0.200 | S_PC1(r=+0.200) |
| ssh.grad_x@local | 0.162 | T_PC1(r=-0.162) |
| ssh.grad_y@local | 0.218 | S_PC3(r=-0.218) |
| ssh.grad_x@1deg | 0.165 | T_PC1(r=-0.165) |
| ssh.grad_y@1deg | 0.215 | S_PC3(r=-0.215) |
| ssh.laplacian@1deg | 0.718 | S_PC1(r=-0.718) |
| sst.tendency@7d | 0.126 | T_PC6(r=-0.126) |
| ssh.tendency@7d | 0.102 | S_PC1(r=+0.102) |
| ssh.geo_u@local | 0.219 | S_PC3(r=+0.219) |
| ssh.geo_v@local | 0.168 | T_PC1(r=-0.168) |
| ssh.geo_u@1deg | 0.216 | S_PC3(r=+0.216) |
| ssh.geo_v@1deg | 0.171 | T_PC1(r=-0.171) |
| bathy | 0.129 | T_PC9(r=-0.129) |
| wind_u | 0.068 | T_PC8(r=-0.068) |
| wind_v | 0.092 | T_PC4(r=-0.092) |
| wind_speed | 0.262 | T_PC2(r=-0.262) |
