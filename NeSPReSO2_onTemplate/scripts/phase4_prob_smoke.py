#!/usr/bin/env python3
"""Phase 4 acceptance: smoke-train crps / nll / quantile (2 epochs each) + dacov selfcheck."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_CFG = _ROOT / "config/argo/config_argo_densityspice_prob_smoke.json"


def _run_mode(mode: str) -> None:
    base = json.loads(_CFG.read_text())
    base["loss_config"]["prob_mode"] = mode
    base["arch"]["args"]["probabilistic"] = True
    if mode == "quantile":
        base["arch"]["args"]["n_quantiles"] = 9
    else:
        base["arch"]["args"]["n_quantiles"] = 0
    base["trainer"]["save_dir"] = f"saved/smoke_prob_{mode}/"
    base["name"] = f"smoke_prob_{mode}"
    with tempfile.NamedTemporaryFile("w", suffix=f"_{mode}.json", delete=False) as f:
        json.dump(base, f, indent=2)
        cfg_path = f.name
    cmd = [sys.executable, str(_ROOT / "train.py"), "-c", cfg_path, "-id", f"smoke_{mode}"]
    print(" ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(_ROOT))


def _dacov_check() -> None:
    import numpy as np
    from sklearn.decomposition import PCA

    sys.path.insert(0, str(_ROOT))
    from dacov import assert_psd, mc_vs_diag_agreement, spice_covariance, density_ctrl_covariance

    rng = np.random.default_rng(0)
    n_z, n_comp = 40, 8
    X = rng.normal(size=(200, n_z))
    pca = PCA(n_components=n_comp).fit(X)
    sd = np.ones(n_z)
    sz = np.abs(rng.normal(size=n_comp)) + 0.1
    cov = spice_covariance(sz, pca, sd)
    assert_psd(cov)
    ag = mc_vs_diag_agreement(sz, pca, sd, n_draw=2000, seed=0, rtol=0.15)
    assert ag["pass"], ag
    dz = np.ones(16)
    ca = density_ctrl_covariance(np.abs(rng.normal(size=16)) + 0.05, dz)
    assert_psd(ca)
    print("dacov selfcheck: OK", ag)


def main() -> int:
    _dacov_check()
    for mode in ("crps", "nll", "quantile"):
        _run_mode(mode)
    print("Phase 4 smoke: all three prob_modes trained")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
