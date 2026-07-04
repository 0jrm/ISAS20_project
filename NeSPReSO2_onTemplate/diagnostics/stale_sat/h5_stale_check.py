#!/usr/bin/env python3
"""Fingerprint stale satellite data in the L4 HDF5: time-constant patches."""
from datetime import date

import h5py
import numpy as np

P = "/unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project/data/NeSPReSO_v2_ARGO_GoM_sat/satellite_NeSPReSO_v2_ARGO_GoM.h5"

with h5py.File(P, "r") as f:
    jd = f["stations"]["julian_date"][:]  # astropy JD
    sst = f["ostia"]["analysed_sst"][:]   # (N,7,5,5)
    ssh = f["ssh"]["adt"][:]
    sss = f["sss"]["sos"][:]

# astropy JD -> gregorian ordinal: JD 2440587.5 = 1970-01-01
days70 = jd - 2440587.5
ords = (days70 + date(1970, 1, 1).toordinal()).astype(int)

def const_frac(arr):
    """Fraction of stations whose 7-day patch is identical at every time step."""
    a = np.nan_to_num(arr, nan=-999.0)
    return np.all(np.abs(a - a[:, :1]) < 1e-6, axis=(1, 2, 3))

cs, ch, cl = const_frac(sst), const_frac(ssh), const_frac(sss)

print("month  n  const_SST  const_SSH  const_SSS")
key = ords // 30
for k in np.unique(key):
    m = key == k
    if m.sum() < 15:
        continue
    d = date.fromordinal(int(ords[m].min()))
    print(f"{d}  n={m.sum():4d}  {cs[m].mean():.2f}  {ch[m].mean():.2f}  {cl[m].mean():.2f}")
