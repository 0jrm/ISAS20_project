"""Permutation nulls must collapse pattern correlation when identities are shuffled."""

from __future__ import annotations

import numpy as np
import pandas as pd

from evalphys.permutation_null import cell_observables, permutation_cache


def _tiny_tables(n: int = 40, nz: int = 8):
    z = np.linspace(50.0, 190.0, nz)
    rng = np.random.default_rng(0)
    t_argo = 20.0 + rng.normal(scale=0.5, size=(n, nz))
    t_xb = t_argo + 1.0 + rng.normal(scale=0.2, size=(n, nz))
    t_nes = t_argo + 0.1 + rng.normal(scale=0.05, size=(n, nz))
    z20_argo = 100.0 + rng.normal(scale=5.0, size=n)
    z20_nes = z20_argo + rng.normal(scale=2.0, size=n)
    z20_xb = z20_argo + 20.0
    casts = pd.DataFrame(
        {
            "cast_id": np.arange(n),
            "model": "A_CRPS",
            "era": "2025",
            "in_bbox": True,
            "ood": False,
            "lc": False,
            "argo_z20_m": z20_argo,
            "xb_z20_m": z20_xb,
            "nes_z20_m": z20_nes,
        }
    )
    levels = pd.DataFrame(
        {
            "cast_id": np.repeat(np.arange(n), nz),
            "model": "A_CRPS",
            "z": np.tile(z, n),
            "t_argo": t_argo.reshape(-1),
            "t_xb": t_xb.reshape(-1),
            "t_nes": t_nes.reshape(-1),
            "valid_t": True,
        }
    )
    return casts, levels


def test_paired_pattern_beats_permutation_mean():
    casts, levels = _tiny_tables()
    obs = cell_observables(casts, levels)
    cache = permutation_cache(casts, levels, n_perm=40, seed=1)
    row = cache[(cache["slice"] == "in_bbox") & (cache["era"] == "2025")].iloc[0]
    assert obs["murphy_pattern"] > 0.7
    assert obs["murphy_xb_pattern"] > 0.7
    assert row["null_pattern_mean"] < obs["murphy_pattern"] - 0.3
    assert row["z_pattern"] > 2.0
