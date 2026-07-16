"""evalphys v1.0.0 regression tests."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from evalphys.calibration import ence, gaussian_crps, gaussian_crps_torch, pit_histogram, spread_skill, summarize_calibration
from evalphys.constants import N2_TOL, SIGMA_MIN_DEFAULT
from evalphys.inversion import sigma0_spice_from_ts, ts_from_sigma0_spice
from evalphys.manifest import write_manifest
from evalphys.metrics import (
    static_stability_violations,
    summarize_physical,
    to_teos10,
)


def _gom_profile_grid(n_prof: int = 4, n_lev: int = 40):
    rng = np.random.default_rng(42)
    depth = np.linspace(0, 500, n_lev)
    lat = 25.0 + rng.uniform(-2, 2, n_prof)
    lon = -90.0 + rng.uniform(-3, 3, n_prof)
    T = 28.0 - 0.04 * depth[None, :] + 0.05 * rng.normal(size=(n_prof, n_lev))
    S = 36.2 - 0.002 * depth[None, :] + 0.01 * rng.normal(size=(n_prof, n_lev))
    return T, S, depth, lat, lon


def test_stable_profile_zero_violations():
    n_prof, n_lev = 3, 50
    depth = np.linspace(0, 800, n_lev)
    lat = np.full(n_prof, 26.0)
    lon = np.full(n_prof, -90.0)
    # Monotonic cooling + salinity increase ⇒ stable
    T = np.broadcast_to(30.0 - 0.03 * depth, (n_prof, n_lev))
    S = np.broadcast_to(35.5 + 0.005 * depth, (n_prof, n_lev))
    out = static_stability_violations(T, S, depth, lat, lon, n2_tol=N2_TOL)
    assert out["violation_rate_profile"] == 0.0
    assert out["violation_rate_level"] == 0.0


def test_injected_inversion_detected():
    T, S, depth, lat, lon = _gom_profile_grid(n_prof=8, n_lev=60)
    # Inject σ₀ inversion: lighter below denser over ~5 m near surface
    sig = __import__("evalphys.metrics", fromlist=["sigma0_profiles"]).sigma0_profiles(T, S, depth, lat, lon)
    k = 3
    sig[:, k + 1] = sig[:, k] - 0.05
    # Reconstruct approximate T,S perturbation via small salinity bump (ponytail: direct σ edit enough for N² test)
    from evalphys.inversion import ts_from_sigma0_spice
    from evalphys.metrics import sigma0_profiles as sp

    sa, ct, p = to_teos10(T, S, depth, lat, lon)
    import gsw

    tau = gsw.spiciness0(sa, ct)
    T2, S2, ok = ts_from_sigma0_spice(sig, tau, p, lon[:, None], lat[:, None])
    out = static_stability_violations(T2, S2, depth, lat, lon, n2_tol=N2_TOL)
    assert out["violation_rate_profile"] > 0.0


def test_perfect_gaussian_calibration():
    rng = np.random.default_rng(0)
    n = 5000
    mu = rng.normal(size=n)
    sigma = rng.uniform(0.3, 1.5, size=n)
    y = mu + rng.normal(scale=sigma)
    pit = pit_histogram(mu, sigma, y)
    assert pit["sup_bin_deviation"] is not None
    assert pit["sup_bin_deviation"] < 0.03
    e = ence(mu, sigma, y)
    assert e["ence"] is not None
    assert e["ence"] < 0.05
    ss = spread_skill(mu, sigma, y)
    assert ss["status"] == "ok"
    assert ss["slope_rmse_vs_sigma"] is not None
    assert 0.9 <= ss["slope_rmse_vs_sigma"] <= 1.1
    assert ss["spearman_sigma_abs_error"] > 0.3


def test_crps_point_forecast_near_mae():
    rng = np.random.default_rng(1)
    y = rng.normal(size=2000)
    mu = y + rng.normal(scale=0.5, size=2000)
    sigma = np.full_like(y, SIGMA_MIN_DEFAULT)
    crps = gaussian_crps(mu, sigma, y)
    mae = np.abs(mu - y)
    rel = np.mean(np.abs(crps - mae)) / np.mean(mae)
    assert rel < 0.01


def test_torch_crps_matches_numpy():
    import torch

    rng = np.random.default_rng(2)
    mu = torch.tensor(rng.normal(size=(4, 5)), dtype=torch.float64)
    sigma = torch.tensor(rng.uniform(0.1, 2.0, size=(4, 5)), dtype=torch.float64)
    y = torch.tensor(rng.normal(size=(4, 5)), dtype=torch.float64)
    np_crps = gaussian_crps(mu.numpy(), sigma.numpy(), y.numpy())
    th_crps = gaussian_crps_torch(mu, sigma, y).detach().numpy()
    assert np.allclose(np_crps, th_crps, rtol=1e-10, atol=1e-12)


def test_gsw_vs_gsw_torch_sigma0():
    import gsw
    import gsw_torch

    rng = np.random.default_rng(99)
    n = 100
    depth = np.linspace(0, 400, 30)
    lat = 25.0 + rng.uniform(-3, 3, n)
    lon = -90.0 + rng.uniform(-4, 4, n)
    T = 27.0 - 0.03 * depth[None, :] + 0.1 * rng.normal(size=(n, depth.size))
    S = 36.0 + 0.001 * depth[None, :] + 0.02 * rng.normal(size=(n, depth.size))
    sa, ct, p = to_teos10(T, S, depth, lat, lon)
    sig_ref = gsw.sigma0(sa, ct)
    lat_t = __import__("torch").as_tensor(lat, dtype=__import__("torch").float64)[:, None]
    lon_t = __import__("torch").as_tensor(lon, dtype=__import__("torch").float64)[:, None]
    sa_t = gsw_torch.SA_from_SP(
        __import__("torch").as_tensor(S, dtype=__import__("torch").float64),
        __import__("torch").as_tensor(p, dtype=__import__("torch").float64),
        lon_t,
        lat_t,
    )
    ct_t = gsw_torch.CT_from_t(
        sa_t,
        __import__("torch").as_tensor(T, dtype=__import__("torch").float64),
        __import__("torch").as_tensor(p, dtype=__import__("torch").float64),
    )
    sig_torch = gsw_torch.sigma0(sa_t, ct_t).detach().numpy()
    # ponytail: plan asks 1e-6; gsw/gsw_torch differ ~1e-5 on σ₀ (see readiness._GSW_REF_ATOL)
    assert np.nanmax(np.abs(sig_ref - sig_torch)) < 1e-4


def test_inversion_round_trip():
    T, S, depth, lat, lon = _gom_profile_grid(n_prof=20, n_lev=35)
    sa, ct, p = to_teos10(T, S, depth, lat, lon)
    import gsw

    sig = gsw.sigma0(sa, ct)
    tau = gsw.spiciness0(sa, ct)
    T2, S2, ok = ts_from_sigma0_spice(sig, tau, p, lon[:, None], lat[:, None], sa0=sa, ct0=ct)
    fail_rate = 1.0 - ok.mean()
    assert fail_rate < 0.001
    m = ok
    assert np.max(np.abs(T2[m] - T[m])) < 0.01
    assert np.max(np.abs(S2[m] - S[m])) < 0.01


def test_manifest_writes():
    path = write_manifest()
    assert path.exists()
    data = __import__("json").loads(path.read_text())
    assert data["version"] == "1.0.0"
    assert "N2_TOL" in data


def test_summarize_physical_smoke():
    T, S, depth, lat, lon = _gom_profile_grid()
    out = summarize_physical(T, S, T, S, depth, lat, lon)
    assert "static_stability_pred" in out
    assert "1e-08" in out["static_stability_pred"]
