"""Frozen evalphys constants — bump VERSION + manifest on semantic change."""

import numpy as np

VERSION = "1.1.0"
N2_TOL = 1.0e-8  # s^-2; headline static-stability tolerance
N2_TOL_SWEEP = (0.0, 1.0e-9, 1.0e-8, 1.0e-7)
DEPTH_BANDS = ((0.0, 50.0), (50.0, 200.0), (200.0, 800.0), (800.0, np.inf))
DEPTH_BAND_LABELS = ("0-50", "50-200", "200-800", ">800")
MLD_DSIGMA_THRESHOLD = 0.03  # kg/m^3 vs 10 m reference (de Boyer Montégut)
MLD_Z_REF_M = 10.0
RHO0_KGM3 = 1025.0
ENCE_MAX = 0.20
SIGMA_MIN_DEFAULT = 1.0e-3
# σ₀ monotonicity: violation iff Δσ₀ < -SIGMA0_TOL (kg/m³) with depth increasing
SIGMA0_TOL = 0.0
GSW_BACKEND_HEADLINE = "gsw"
