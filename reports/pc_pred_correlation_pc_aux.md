# Predicted PC–input correlation

Checkpoint `saved/pc_aux/models/NeSPReSO2_ARGO_GoM_pc_aux_pc_aux_s2/pc_aux_s2/model_best.pth`. Chronological `test`, n=623.
r_pred: inputs vs model μ PCs. Residual: truth − μ. Skill: r(μ, truth) per PC.

## Per-PC skill r(μ, truth)

| PC | r | truth best input | pred best input | residual max\|r\| |
|----|--:|------------------|-----------------|------------------:|
| T_PC1 | +0.971 | ssh(+0.965) | ssh(+0.993) | 0.144 |
| T_PC2 | +0.800 | timecos(+0.697) | timecos(+0.811) | 0.170 |
| T_PC3 | +0.636 | ssh.geo_u@local(+0.408) | sst(-0.456) | 0.353 |
| T_PC4 | +0.386 | lonsin(+0.227) | roni(-0.592) | 0.203 |
| T_PC5 | +0.343 | oni(-0.268) | oni(-0.662) | 0.138 |
| T_PC6 | +0.395 | latsin(+0.284) | oni(-0.538) | 0.226 |
| T_PC7 | +0.202 | oni(-0.164) | timesin(+0.442) | 0.102 |
| T_PC8 | +0.035 | timecos(-0.280) | ssh(-0.683) | 0.287 |
| T_PC9 | +0.026 | sss(+0.098) | ssh.grad_x@local(+0.525) | 0.127 |
| T_PC10 | +0.022 | lonsin(-0.130) | sst.grad_y@1.0deg(-0.513) | 0.151 |
| T_PC11 | -0.004 | sst.grad_x@1.0deg(-0.104) | ssh(-0.453) | 0.140 |
| T_PC12 | +0.064 | ssh.tendency@7d(+0.093) | ssh(+0.629) | 0.119 |
| T_PC13 | +0.040 | ssh.geo_v@1.0deg(+0.087) | sss(-0.528) | 0.106 |
| T_PC14 | +0.036 | timecos(-0.090) | sst.tendency@7d(+0.554) | 0.130 |
| T_PC15 | -0.122 | ssh(+0.184) | ssh.geo_v@1.0deg(+0.640) | 0.260 |
| T_PC16 | +0.041 | sss.grad_y@1.0deg(-0.118) | sst.tendency@7d(+0.724) | 0.107 |
| T_PC17 | -0.003 | sss.grad_x@1.0deg(+0.089) | sst.tendency@7d(-0.546) | 0.098 |
| T_PC18 | +0.080 | ssh.geo_u@local(+0.133) | sst.tendency@7d(+0.670) | 0.135 |
| T_PC19 | +0.149 | ssh(-0.143) | sst.tendency@7d(-0.412) | 0.112 |
| T_PC20 | +0.101 | ssh(+0.122) | ssh.grad_y@1.0deg(+0.563) | 0.165 |
| T_PC21 | -0.031 | ssh(+0.129) | roni(+0.531) | 0.184 |
| T_PC22 | -0.038 | ssh(+0.144) | ssh(-0.586) | 0.190 |
| T_PC23 | -0.032 | sss.grad_x@1.0deg(-0.094) | sss.grad_x@1.0deg(+0.539) | 0.168 |
| T_PC24 | -0.042 | sss.grad_y@1.0deg(+0.093) | ssh(-0.474) | 0.129 |
| T_PC25 | +0.018 | ssh.geo_u@local(+0.155) | sst.tendency@7d(-0.680) | 0.146 |
| T_PC26 | +0.046 | sst.grad_x@local(+0.074) | sst.grad_y@1.0deg(+0.500) | 0.090 |
| T_PC27 | -0.047 | ssh.tendency@7d(+0.148) | ssh(+0.562) | 0.158 |
| T_PC28 | +0.123 | sst.tendency@7d(+0.096) | ssh.grad_y@1.0deg(-0.691) | 0.094 |
| T_PC29 | +0.050 | ssh.grad_x@1.0deg(-0.126) | ssh.grad_x@1.0deg(-0.471) | 0.125 |
| T_PC30 | -0.063 | sst.grad_y@1.0deg(+0.073) | ssh.geo_u@local(-0.542) | 0.120 |
| T_PC31 | -0.008 | ssh.tendency@7d(+0.103) | timecos(-0.322) | 0.107 |
| T_PC32 | -0.075 | ssh.geo_u@1.0deg(-0.072) | ssh.grad_x@local(-0.535) | 0.104 |

### Salinity

| PC | r | truth best input | pred best input | residual max\|r\| |
|----|--:|------------------|-----------------|------------------:|
| S_PC1 | +0.966 | ssh(+0.957) | ssh(+0.993) | 0.178 |
| S_PC2 | +0.523 | sss(+0.619) | sss(+0.681) | 0.269 |
| S_PC3 | +0.764 | ssh.geo_u@local(+0.315) | ssh(-0.446) | 0.220 |
| S_PC4 | +0.379 | lonsin(-0.359) | sss(-0.611) | 0.321 |
| S_PC5 | +0.279 | sss(+0.214) | sss(+0.528) | 0.187 |
| S_PC6 | +0.288 | sss(+0.341) | sss(+0.409) | 0.315 |
| S_PC7 | +0.369 | sss(+0.397) | ssh(-0.347) | 0.331 |
| S_PC8 | +0.312 | sss(+0.234) | oni(+0.468) | 0.216 |
| S_PC9 | +0.184 | ssh(-0.184) | ssh(-0.617) | 0.144 |
| S_PC10 | +0.087 | lonsin(-0.243) | ssh(-0.664) | 0.219 |
| S_PC11 | -0.108 | sss(-0.211) | lonsin(-0.525) | 0.208 |
| S_PC12 | +0.105 | latcos(+0.134) | ssh(-0.515) | 0.156 |
| S_PC13 | -0.020 | sss(-0.168) | ssh(+0.713) | 0.274 |
| S_PC14 | +0.061 | sss(-0.137) | sst(-0.691) | 0.181 |
| S_PC15 | -0.077 | lonsin(+0.112) | loncos(-0.430) | 0.159 |
| S_PC16 | +0.077 | timesin(-0.088) | ssh(-0.704) | 0.129 |
| S_PC17 | +0.020 | lonsin(-0.129) | ssh(-0.630) | 0.218 |
| S_PC18 | +0.115 | ssh(-0.101) | timecos(-0.407) | 0.082 |
| S_PC19 | -0.077 | ssh.tendency@7d(+0.121) | ssh(-0.714) | 0.200 |
| S_PC20 | -0.000 | sss(+0.075) | ssh(-0.805) | 0.209 |
| S_PC21 | +0.002 | sss(+0.077) | sss(-0.415) | 0.154 |
| S_PC22 | +0.069 | sss(+0.084) | ssh(-0.849) | 0.242 |
| S_PC23 | +0.005 | sss(+0.116) | sst(+0.403) | 0.141 |
| S_PC24 | +0.102 | sst.grad_y@1.0deg(+0.137) | sst(+0.623) | 0.125 |
| S_PC25 | -0.059 | lonsin(-0.139) | sss.grad_x@local(-0.488) | 0.139 |
| S_PC26 | +0.036 | sst.grad_y@local(+0.130) | ssh.geo_u@1.0deg(+0.682) | 0.128 |
| S_PC27 | -0.076 | ssh(+0.115) | ssh(-0.511) | 0.218 |
| S_PC28 | -0.085 | sst(-0.093) | ssh(+0.905) | 0.283 |
| S_PC29 | +0.078 | ssh(+0.154) | sss(-0.672) | 0.222 |
| S_PC30 | -0.004 | sss.grad_x@1.0deg(+0.094) | ssh(-0.831) | 0.243 |
| S_PC31 | +0.031 | sss.grad_y@local(-0.104) | timecos(-0.664) | 0.162 |
| S_PC32 | -0.042 | sst.grad_x@local(-0.059) | ssh.grad_y@1.0deg(-0.520) | 0.150 |

## Predicted μ — all 32 T PCs, top-3 inputs

| PC | max|r| | best input | 2nd | 3rd |
|----|--------|------------|-----|-----|
| T_PC1 | 0.993 | ssh(+0.993) | ssh.laplacian@1.0deg(-0.741) | loncos(+0.468) |
| T_PC2 | 0.811 | timecos(+0.811) | sst(+0.790) | sss(-0.457) |
| T_PC3 | 0.456 | sst(-0.456) | timecos(-0.434) | lonsin(+0.275) |
| T_PC4 | 0.592 | roni(-0.592) | oni(-0.590) | timesin(+0.581) |
| T_PC5 | 0.662 | oni(-0.662) | roni(-0.647) | timesin(+0.647) |
| T_PC6 | 0.538 | oni(-0.538) | timesin(+0.530) | roni(-0.517) |
| T_PC7 | 0.442 | timesin(+0.442) | oni(-0.407) | sst.tendency@7d(-0.403) |
| T_PC8 | 0.683 | ssh(-0.683) | ssh.laplacian@1.0deg(+0.537) | sst.tendency@7d(-0.372) |
| T_PC9 | 0.525 | ssh.grad_x@local(+0.525) | ssh.geo_v@local(+0.524) | ssh.grad_x@1.0deg(+0.521) |
| T_PC10 | 0.513 | sst.grad_y@1.0deg(-0.513) | ssh.geo_u@1.0deg(-0.432) | ssh.geo_u@local(-0.428) |
| T_PC11 | 0.453 | ssh(-0.453) | sss.grad_y@local(-0.450) | sss(-0.417) |
| T_PC12 | 0.629 | ssh(+0.629) | sst(+0.404) | ssh.laplacian@1.0deg(-0.379) |
| T_PC13 | 0.528 | sss(-0.528) | sst.tendency@7d(+0.480) | ssh.grad_y@local(-0.364) |
| T_PC14 | 0.554 | sst.tendency@7d(+0.554) | ssh.grad_y@1.0deg(-0.453) | ssh.grad_y@local(-0.448) |
| T_PC15 | 0.640 | ssh.geo_v@1.0deg(+0.640) | ssh.geo_v@local(+0.633) | ssh.grad_x@1.0deg(+0.631) |
| T_PC16 | 0.724 | sst.tendency@7d(+0.724) | ssh.grad_y@1.0deg(-0.305) | ssh.tendency@7d(+0.305) |
| T_PC17 | 0.546 | sst.tendency@7d(-0.546) | ssh(-0.537) | ssh.laplacian@1.0deg(+0.344) |
| T_PC18 | 0.670 | sst.tendency@7d(+0.670) | sss.grad_y@local(-0.437) | roni(+0.375) |
| T_PC19 | 0.412 | sst.tendency@7d(-0.412) | ssh.grad_x@local(-0.381) | ssh.geo_v@local(-0.379) |
| T_PC20 | 0.563 | ssh.grad_y@1.0deg(+0.563) | ssh.grad_y@local(+0.561) | ssh.geo_u@1.0deg(-0.559) |
| T_PC21 | 0.531 | roni(+0.531) | oni(+0.509) | timesin(-0.486) |
| T_PC22 | 0.586 | ssh(-0.586) | ssh.laplacian@1.0deg(+0.537) | ssh.tendency@7d(-0.485) |
| T_PC23 | 0.539 | sss.grad_x@1.0deg(+0.539) | ssh(-0.458) | sss.grad_x@local(+0.442) |
| T_PC24 | 0.474 | ssh(-0.474) | sss.grad_y@local(+0.473) | ssh.laplacian@1.0deg(+0.365) |
| T_PC25 | 0.680 | sst.tendency@7d(-0.680) | sst.grad_y@1.0deg(+0.341) | lonsin(+0.292) |
| T_PC26 | 0.500 | sst.grad_y@1.0deg(+0.500) | sss.grad_y@1.0deg(+0.415) | lonsin(+0.398) |
| T_PC27 | 0.562 | ssh(+0.562) | sst(+0.415) | ssh.laplacian@1.0deg(-0.365) |
| T_PC28 | 0.691 | ssh.grad_y@1.0deg(-0.691) | ssh.grad_y@local(-0.689) | ssh.geo_u@1.0deg(+0.680) |
| T_PC29 | 0.471 | ssh.grad_x@1.0deg(-0.471) | ssh.grad_x@local(-0.469) | ssh.geo_v@1.0deg(-0.462) |
| T_PC30 | 0.542 | ssh.geo_u@local(-0.542) | ssh.geo_u@1.0deg(-0.542) | ssh.grad_y@1.0deg(+0.530) |
| T_PC31 | 0.322 | timecos(-0.322) | sss.grad_y@1.0deg(+0.292) | sss.grad_y@local(+0.249) |
| T_PC32 | 0.535 | ssh.grad_x@local(-0.535) | ssh.grad_x@1.0deg(-0.532) | ssh.geo_v@local(-0.530) |

## Predicted μ — all 32 S PCs, top-3 inputs

| PC | max|r| | best input | 2nd | 3rd |
|----|--------|------------|-----|-----|
| S_PC1 | 0.993 | ssh(+0.993) | ssh.laplacian@1.0deg(-0.745) | loncos(+0.458) |
| S_PC2 | 0.681 | sss(+0.681) | sst(-0.415) | timecos(-0.355) |
| S_PC3 | 0.446 | ssh(-0.446) | ssh.grad_y@local(-0.435) | ssh.geo_u@local(+0.432) |
| S_PC4 | 0.611 | sss(-0.611) | sst(+0.380) | oni(+0.376) |
| S_PC5 | 0.528 | sss(+0.528) | oni(-0.495) | roni(-0.472) |
| S_PC6 | 0.409 | sss(+0.409) | ssh(+0.247) | ssh.laplacian@1.0deg(-0.232) |
| S_PC7 | 0.347 | ssh(-0.347) | sst.grad_y@1.0deg(+0.317) | sss(+0.317) |
| S_PC8 | 0.468 | oni(+0.468) | timesin(-0.462) | roni(+0.453) |
| S_PC9 | 0.617 | ssh(-0.617) | ssh.laplacian@1.0deg(+0.375) | sst(-0.374) |
| S_PC10 | 0.664 | ssh(-0.664) | ssh.laplacian@1.0deg(+0.447) | sss.grad_x@1.0deg(+0.327) |
| S_PC11 | 0.525 | lonsin(-0.525) | ssh.geo_u@1.0deg(-0.476) | ssh.geo_u@local(-0.466) |
| S_PC12 | 0.515 | ssh(-0.515) | sss(+0.384) | sst(-0.379) |
| S_PC13 | 0.713 | ssh(+0.713) | sss(+0.596) | ssh.laplacian@1.0deg(-0.588) |
| S_PC14 | 0.691 | sst(-0.691) | timecos(-0.522) | roni(-0.436) |
| S_PC15 | 0.430 | loncos(-0.430) | ssh.geo_v@local(+0.422) | ssh.grad_y@1.0deg(+0.421) |
| S_PC16 | 0.704 | ssh(-0.704) | ssh.laplacian@1.0deg(+0.535) | sss.grad_y@local(-0.476) |
| S_PC17 | 0.630 | ssh(-0.630) | loncos(-0.548) | ssh.laplacian@1.0deg(+0.471) |
| S_PC18 | 0.407 | timecos(-0.407) | lonsin(+0.404) | sst(-0.347) |
| S_PC19 | 0.714 | ssh(-0.714) | sss.grad_x@local(+0.573) | ssh.laplacian@1.0deg(+0.558) |
| S_PC20 | 0.805 | ssh(-0.805) | sst(-0.613) | ssh.laplacian@1.0deg(+0.516) |
| S_PC21 | 0.415 | sss(-0.415) | sst(+0.384) | timecos(+0.339) |
| S_PC22 | 0.849 | ssh(-0.849) | ssh.laplacian@1.0deg(+0.562) | sst(-0.537) |
| S_PC23 | 0.403 | sst(+0.403) | ssh(+0.395) | roni(+0.338) |
| S_PC24 | 0.623 | sst(+0.623) | timecos(+0.412) | sss.grad_x@local(+0.340) |
| S_PC25 | 0.488 | sss.grad_x@local(-0.488) | ssh(+0.385) | sst.tendency@7d(+0.351) |
| S_PC26 | 0.682 | ssh.geo_u@1.0deg(+0.682) | ssh.grad_y@1.0deg(-0.677) | ssh.geo_u@local(+0.671) |
| S_PC27 | 0.511 | ssh(-0.511) | sss(-0.496) | lonsin(-0.459) |
| S_PC28 | 0.905 | ssh(+0.905) | ssh.laplacian@1.0deg(-0.626) | sst(+0.451) |
| S_PC29 | 0.672 | sss(-0.672) | sst(+0.647) | timecos(+0.515) |
| S_PC30 | 0.831 | ssh(-0.831) | ssh.laplacian@1.0deg(+0.569) | sst(-0.320) |
| S_PC31 | 0.664 | timecos(-0.664) | sst(-0.570) | roni(-0.399) |
| S_PC32 | 0.520 | ssh.grad_y@1.0deg(-0.520) | ssh.geo_u@1.0deg(+0.514) | ssh.grad_y@local(-0.512) |

## Residual (truth−μ) — all 32 T PCs

| PC | max|r| | best input | 2nd | 3rd |
|----|--------|------------|-----|-----|
| T_PC1 | 0.144 | sss(+0.144) | sst.grad_y@1.0deg(+0.130) | sst.grad_x@local(-0.113) |
| T_PC2 | 0.170 | sss.grad_x@1.0deg(-0.170) | ssh(+0.170) | ssh.grad_y@local(-0.127) |
| T_PC3 | 0.353 | ssh.geo_u@local(+0.353) | ssh.geo_u@1.0deg(+0.352) | ssh.grad_y@local(-0.351) |
| T_PC4 | 0.203 | sss(-0.203) | loncos(+0.172) | oni(+0.159) |
| T_PC5 | 0.138 | sst.grad_y@1.0deg(+0.138) | sss.grad_y@1.0deg(+0.124) | sss.grad_y@local(+0.116) |
| T_PC6 | 0.226 | latcos(-0.226) | latsin(+0.226) | timesin(-0.154) |
| T_PC7 | 0.102 | sst.tendency@7d(+0.102) | timecos(-0.090) | sst.grad_y@1.0deg(+0.083) |
| T_PC8 | 0.287 | timecos(-0.287) | loncos(+0.231) | ssh.laplacian@1.0deg(-0.191) |
| T_PC9 | 0.127 | ssh.tendency@7d(-0.127) | ssh.geo_v@1.0deg(-0.112) | ssh.grad_x@1.0deg(-0.110) |
| T_PC10 | 0.151 | ssh(+0.151) | sss.grad_x@1.0deg(-0.112) | timecos(+0.095) |
| T_PC11 | 0.140 | timecos(+0.140) | sst(+0.127) | ssh.grad_y@1.0deg(+0.124) |
| T_PC12 | 0.119 | sst.grad_y@local(-0.119) | ssh(-0.105) | ssh.tendency@7d(+0.096) |
| T_PC13 | 0.106 | ssh.grad_x@1.0deg(+0.106) | ssh.geo_v@1.0deg(+0.105) | ssh(+0.104) |
| T_PC14 | 0.130 | timecos(-0.130) | sst(-0.110) | ssh.geo_v@local(+0.075) |
| T_PC15 | 0.260 | ssh(+0.260) | ssh.laplacian@1.0deg(-0.210) | ssh.geo_v@local(-0.149) |
| T_PC16 | 0.107 | sss.grad_y@1.0deg(-0.107) | sst.tendency@7d(-0.097) | loncos(-0.090) |
| T_PC17 | 0.098 | ssh.geo_u@1.0deg(-0.098) | ssh.grad_y@1.0deg(+0.096) | ssh.geo_u@local(-0.095) |
| T_PC18 | 0.135 | lonsin(+0.135) | sst.grad_y@1.0deg(+0.128) | ssh.geo_u@local(+0.119) |
| T_PC19 | 0.112 | ssh(-0.112) | roni(-0.107) | oni(-0.099) |
| T_PC20 | 0.165 | ssh(+0.165) | sst(+0.115) | ssh.tendency@7d(+0.083) |
| T_PC21 | 0.184 | ssh(+0.184) | ssh.geo_v@local(-0.124) | ssh.grad_x@local(-0.120) |
| T_PC22 | 0.190 | ssh(+0.190) | sss.grad_y@local(+0.154) | sst(+0.127) |
| T_PC23 | 0.168 | sss.grad_x@1.0deg(-0.168) | lonsin(+0.124) | sss.grad_x@local(-0.116) |
| T_PC24 | 0.129 | ssh(+0.129) | sss.grad_x@local(-0.104) | sss.grad_x@1.0deg(-0.103) |
| T_PC25 | 0.146 | ssh.geo_u@local(+0.146) | ssh.grad_y@local(-0.146) | ssh.geo_u@1.0deg(+0.144) |
| T_PC26 | 0.090 | sst.grad_x@local(+0.090) | sss(-0.079) | ssh.grad_x@local(+0.074) |
| T_PC27 | 0.158 | ssh.tendency@7d(+0.158) | sst(-0.098) | ssh(-0.079) |
| T_PC28 | 0.094 | sss.grad_x@1.0deg(+0.094) | sss.grad_x@local(+0.071) | ssh.geo_v@1.0deg(+0.068) |
| T_PC29 | 0.125 | ssh(+0.125) | sss.grad_x@1.0deg(-0.115) | sst.grad_y@local(+0.080) |
| T_PC30 | 0.120 | sst.grad_y@1.0deg(+0.120) | lonsin(+0.085) | latcos(+0.080) |
| T_PC31 | 0.107 | ssh.tendency@7d(+0.107) | roni(-0.093) | oni(-0.091) |
| T_PC32 | 0.104 | ssh.geo_u@1.0deg(-0.104) | ssh.grad_y@1.0deg(+0.103) | ssh.geo_u@local(-0.102) |

## Residual (truth−μ) — all 32 S PCs

| PC | max|r| | best input | 2nd | 3rd |
|----|--------|------------|-----|-----|
| S_PC1 | 0.178 | sss(+0.178) | loncos(-0.157) | ssh.grad_y@1.0deg(+0.150) |
| S_PC2 | 0.269 | sss.grad_y@local(+0.269) | sss(+0.267) | lonsin(-0.200) |
| S_PC3 | 0.220 | ssh.laplacian@1.0deg(-0.220) | sss(+0.166) | sss.grad_y@local(+0.157) |
| S_PC4 | 0.321 | lonsin(-0.321) | ssh.geo_u@1.0deg(-0.281) | ssh.grad_y@1.0deg(+0.276) |
| S_PC5 | 0.187 | latsin(+0.187) | latcos(-0.185) | sst.grad_x@local(+0.147) |
| S_PC6 | 0.315 | oni(-0.315) | roni(-0.299) | timesin(+0.293) |
| S_PC7 | 0.331 | sss(+0.331) | sss.grad_y@local(+0.239) | latcos(+0.231) |
| S_PC8 | 0.216 | sss(+0.216) | ssh(-0.134) | sss.grad_x@local(+0.130) |
| S_PC9 | 0.144 | sss(-0.144) | sst(+0.134) | sst.tendency@7d(+0.129) |
| S_PC10 | 0.219 | lonsin(-0.219) | timecos(+0.192) | sst(+0.190) |
| S_PC11 | 0.208 | sss(-0.208) | lonsin(+0.206) | loncos(+0.199) |
| S_PC12 | 0.156 | sst(+0.156) | sst.grad_x@local(+0.097) | lonsin(-0.092) |
| S_PC13 | 0.274 | sss(-0.274) | sss.grad_y@local(-0.186) | latcos(-0.174) |
| S_PC14 | 0.181 | sss(-0.181) | timecos(+0.145) | sst(+0.090) |
| S_PC15 | 0.159 | loncos(+0.159) | lonsin(+0.146) | ssh.grad_y@1.0deg(-0.143) |
| S_PC16 | 0.129 | ssh(+0.129) | sst.grad_y@local(+0.084) | sss.grad_y@local(+0.083) |
| S_PC17 | 0.218 | ssh(+0.218) | sst(+0.187) | ssh.laplacian@1.0deg(-0.166) |
| S_PC18 | 0.082 | sss.grad_y@1.0deg(-0.082) | sss(-0.070) | sss.grad_x@1.0deg(+0.067) |
| S_PC19 | 0.200 | sss.grad_x@1.0deg(-0.200) | ssh(+0.199) | ssh.tendency@7d(+0.166) |
| S_PC20 | 0.209 | ssh(+0.209) | ssh.laplacian@1.0deg(-0.162) | sst(+0.159) |
| S_PC21 | 0.154 | sss(+0.154) | sst(-0.137) | timecos(-0.114) |
| S_PC22 | 0.242 | ssh(+0.242) | ssh.laplacian@1.0deg(-0.189) | sst(+0.151) |
| S_PC23 | 0.141 | sss(+0.141) | sst(-0.138) | roni(-0.130) |
| S_PC24 | 0.125 | sss(+0.125) | sst(-0.125) | sst.grad_y@local(-0.122) |
| S_PC25 | 0.139 | lonsin(-0.139) | sss.grad_x@local(+0.107) | sst.grad_y@1.0deg(-0.106) |
| S_PC26 | 0.128 | ssh(-0.128) | sst.grad_y@local(+0.119) | sss.grad_x@1.0deg(+0.090) |
| S_PC27 | 0.218 | ssh(+0.218) | ssh.laplacian@1.0deg(-0.152) | ssh.geo_v@1.0deg(-0.140) |
| S_PC28 | 0.283 | ssh(-0.283) | sst(-0.202) | ssh.laplacian@1.0deg(+0.169) |
| S_PC29 | 0.222 | sss(+0.222) | timesin(+0.134) | sss.grad_y@local(+0.129) |
| S_PC30 | 0.243 | ssh(+0.243) | ssh.laplacian@1.0deg(-0.176) | ssh.geo_u@local(-0.095) |
| S_PC31 | 0.162 | sss.grad_y@local(-0.162) | timecos(+0.128) | ssh(-0.123) |
| S_PC32 | 0.150 | ssh(-0.150) | loncos(-0.140) | lonsin(-0.116) |

## Unused channels vs residual PCs (bathy/wind, not in this net)

| channel | max\|r\| | best residual PC |
|---------|--------:|------------------|
| bathy | 0.216 | S_PC7(r=+0.216) |
| wind_u | 0.103 | S_PC2(r=+0.103) |
| wind_v | 0.120 | S_PC4(r=+0.120) |
| wind_speed | 0.132 | T_PC28(r=-0.132) |

## Summary

Mean r(μ, truth) on easy PCs (T1–4, S1–2) = 0.714; on the other 58 PCs = 0.071.
Wrote `/unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project/NeSPReSO2_onTemplate/../reports/pc_pred_correlation_pc_aux.json`.

