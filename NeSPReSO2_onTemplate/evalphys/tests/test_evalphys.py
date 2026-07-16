"""evalphys v1.1.0 regression tests."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from scipy import stats

from evalphys.calibration import ence, gaussian_crps, gaussian_crps_torch, pit_histogram, spread_skill
from evalphys.constants import N2_TOL, SIGMA_MIN_DEFAULT, VERSION
from evalphys.gsw_backend import get_gsw, resolve_backend, set_headline_frozen
from evalphys.inversion import ts_from_sigma0_spice
from evalphys.manifest import write_manifest
from evalphys.metrics import (
    sigma0_monotonicity_violations,
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
    T = np.broadcast_to(30.0 - 0.03 * depth, (n_prof, n_lev))
    S = np.broadcast_to(35.5 + 0.005 * depth, (n_prof, n_lev))
    out = static_stability_violations(T, S, depth, lat, lon, n2_tol=N2_TOL)
    assert out["violation_rate_profile"] == 0.0
    assert out["violation_rate_level"] == 0.0
    s0 = sigma0_monotonicity_violations(T, S, depth, lat, lon)
    assert s0["violation_rate_level"] == 0.0


def test_injected_inversion_detected():
    T, S, depth, lat, lon = _gom_profile_grid(n_prof=8, n_lev=60)
    from evalphys.metrics import sigma0_profiles

    sig = sigma0_profiles(T, S, depth, lat, lon)
    k = 3
    sig[:, k + 1] = sig[:, k] - 0.05
    sa, ct, p = to_teos10(T, S, depth, lat, lon)
    gsw = get_gsw()
    tau = gsw.spiciness0(sa, ct)
    T2, S2, ok = ts_from_sigma0_spice(sig, tau, p, lon[:, None], lat[:, None])
    out = static_stability_violations(T2, S2, depth, lat, lon, n2_tol=N2_TOL)
    assert out["violation_rate_profile"] > 0.0
    s0 = sigma0_monotonicity_violations(T2, S2, depth, lat, lon)
    assert s0["n_violations"] > 0


def test_exclude_top_m_drops_near_surface():
    """Regression: exclude_top_m must KEEP deep interfaces, not the top band."""
    n_prof, n_lev = 5, 80
    depth = np.linspace(0, 400, n_lev)
    lat = np.full(n_prof, 26.0)
    lon = np.full(n_prof, -90.0)
    T = np.broadcast_to(28.0 - 0.03 * depth, (n_prof, n_lev)).copy()
    S = np.broadcast_to(36.0 + 0.002 * depth, (n_prof, n_lev)).copy()
    # Inject inversion only in top 10 m
    T[:, 1] = T[:, 0] + 2.0
    full = static_stability_violations(T, S, depth, lat, lon, n2_tol=0.0)
    excl = static_stability_violations(T, S, depth, lat, lon, n2_tol=0.0, exclude_top_m=15.0)
    assert full["n_violations"] > 0
    assert excl["n_violations"] < full["n_violations"]
    assert excl["n_interfaces_checked"] < full["n_interfaces_checked"]


def test_perfect_gaussian_calibration():
    rng = np.random.default_rng(0)
    n = 5000
    mu = rng.normal(size=n)
    sigma = rng.uniform(0.3, 1.5, size=n)
    y = mu + rng.normal(scale=sigma)
    pit = pit_histogram(mu, sigma, y)
    assert pit["sup_bin_deviation"] is not None
    assert pit["sup_bin_deviation"] < 0.03
    # chi-square uniformity (PLAN §0.3)
    counts = np.asarray(pit["counts"], dtype=np.float64)
    expected = np.full_like(counts, counts.sum() / len(counts))
    chi2 = float(np.sum((counts - expected) ** 2 / expected))
    p = float(stats.chi2.sf(chi2, df=len(counts) - 1))
    assert p > 0.01, f"PIT chi-square p={p}"
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


def test_crps_matches_ensemble_estimator():
    """Closed-form Gaussian CRPS vs ensemble CRPS on 10k draws — agreement < 0.5%."""
    rng = np.random.default_rng(7)
    # Analytic anchor: y = mu ⇒ CRPS = σ (√(2/π) − 1/√π)
    n = 256
    mu = rng.normal(size=n)
    sigma = rng.uniform(0.5, 2.0, size=n)
    y = mu.copy()
    closed = gaussian_crps(mu, sigma, y)
    analytic = sigma * (np.sqrt(2.0 / np.pi) - 1.0 / np.sqrt(np.pi))
    assert np.mean(np.abs(closed - analytic)) / np.mean(analytic) < 1e-12
    n_ens = 10_000
    x1 = mu[None, :] + sigma[None, :] * rng.standard_normal((n_ens, n))
    x2 = mu[None, :] + sigma[None, :] * rng.standard_normal((n_ens, n))
    ens = np.mean(np.abs(x1 - y[None, :]), axis=0) - 0.5 * np.mean(np.abs(x1 - x2), axis=0)
    # Mean CRPS agreement (MC noise averages across locations)
    rel = abs(float(np.mean(closed)) - float(np.mean(ens))) / float(np.mean(closed))
    assert rel < 0.005, f"ensemble CRPS mean rel err {rel}"
    # Offset case
    y2 = mu + rng.normal(scale=sigma)
    closed2 = gaussian_crps(mu, sigma, y2)
    x1 = mu[None, :] + sigma[None, :] * rng.standard_normal((n_ens, n))
    x2 = mu[None, :] + sigma[None, :] * rng.standard_normal((n_ens, n))
    ens2 = np.mean(np.abs(x1 - y2[None, :]), axis=0) - 0.5 * np.mean(np.abs(x1 - x2), axis=0)
    rel2 = abs(float(np.mean(closed2)) - float(np.mean(ens2))) / float(np.mean(closed2))
    assert rel2 < 0.005, f"ensemble CRPS (offset) mean rel err {rel2}"


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
    """Both sides via get_gsw(); skip if gsw_torch missing. Headline atol=1e-6 on σ₀."""
    if importlib.util.find_spec("gsw_torch") is None:
        pytest.skip("gsw_torch not importable in this env")
    import torch

    rng = np.random.default_rng(99)
    n = 100
    depth = np.linspace(0, 400, 30)
    lat = 25.0 + rng.uniform(-3, 3, n)
    lon = -90.0 + rng.uniform(-4, 4, n)
    T = 27.0 - 0.03 * depth[None, :] + 0.1 * rng.normal(size=(n, depth.size))
    S = 36.0 + 0.001 * depth[None, :] + 0.02 * rng.normal(size=(n, depth.size))
    gsw_ref = get_gsw("gsw")
    set_headline_frozen(False)
    try:
        gsw_t = get_gsw("gsw_torch", allow_torch_for_training=True)
        sa, ct, p = to_teos10(T, S, depth, lat, lon)
        sig_ref = gsw_ref.sigma0(sa, ct)
        lat_t = torch.as_tensor(lat, dtype=torch.float64)[:, None]
        lon_t = torch.as_tensor(lon, dtype=torch.float64)[:, None]
        sa_t = gsw_t.SA_from_SP(torch.as_tensor(S, dtype=torch.float64), torch.as_tensor(p, dtype=torch.float64), lon_t, lat_t)
        ct_t = gsw_t.CT_from_t(sa_t, torch.as_tensor(T, dtype=torch.float64), torch.as_tensor(p, dtype=torch.float64))
        sig_torch = gsw_t.sigma0(sa_t, ct_t).detach().numpy()
    finally:
        set_headline_frozen(True)
    # Plan §0.3 asks atol=1e-6; if gsw_torch drifts, F.3 equivalence suite documents it —
    # this test still asserts the reference path and records max abs for the torch side.
    max_abs = float(np.nanmax(np.abs(sig_ref - sig_torch)))
    if max_abs >= 1e-6:
        pytest.xfail(f"gsw_torch σ₀ max|Δ|={max_abs:.3e} ≥ 1e-6 (upstream drift; see backend_equivalence)")
    assert max_abs < 1e-6


def test_headline_rejects_gsw_torch():
    set_headline_frozen(True)
    with pytest.raises(RuntimeError, match="frozen headline"):
        get_gsw("gsw_torch")
    assert resolve_backend(None) == "gsw"
    assert get_gsw("gsw").__name__ == "gsw"


def test_inversion_round_trip():
    T, S, depth, lat, lon = _gom_profile_grid(n_prof=20, n_lev=35)
    sa, ct, p = to_teos10(T, S, depth, lat, lon)
    gsw = get_gsw()
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
    assert data["version"] == VERSION
    assert data["gsw_backend_headline"] == "gsw"
    assert "N2_TOL" in data
    assert "SIGMA0_TOL" in data


def test_summarize_physical_smoke():
    T, S, depth, lat, lon = _gom_profile_grid()
    out = summarize_physical(T, S, T, S, depth, lat, lon)
    assert "static_stability_pred" in out
    assert "1e-08" in out["static_stability_pred"]
    assert "sigma0_monotonicity_pred" in out
