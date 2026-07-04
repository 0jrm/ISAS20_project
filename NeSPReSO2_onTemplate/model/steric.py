"""Steric height consistency (Phase B)."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import torch
import torch.nn as nn

import gsw_torch as gsw

_GSW_DTYPE = torch.float64
_G = 9.81
_REF_PRES_DBAR = 1800.0


def _to_depth_major(
    temp: torch.Tensor,
    sal: torch.Tensor,
    pres_1d: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ensure (n_depth, batch) layout."""
    if pres_1d is not None and pres_1d.ndim == 1:
        nz = int(pres_1d.shape[0])
        if temp.shape[0] == nz:
            return temp, sal
        if temp.shape[1] == nz:
            return temp.T, sal.T
    if temp.shape[0] >= temp.shape[1]:
        return temp, sal
    return temp.T, sal.T


def steric_height_anomaly(
    temp: torch.Tensor,
    sal: torch.Tensor,
    pres_dbar: torch.Tensor,
    lat: torch.Tensor,
    lon: torch.Tensor,
    *,
    subsample_dz: int = 1,
    ref_pres_dbar: float = _REF_PRES_DBAR,
) -> torch.Tensor:
    """
    Dynamic height anomaly relative to ``ref_pres_dbar`` (m).

    ``temp``, ``sal``: (batch, n_depth) or (n_depth, batch).
    Returns shape ``(batch,)``.
    """
    pres_hint = pres_dbar if pres_dbar.ndim == 1 else None
    temp, sal = _to_depth_major(temp, sal, pres_hint)
    n_depth, batch = temp.shape
    device = temp.device

    if pres_dbar.ndim == 1:
        pres = pres_dbar.to(device=device, dtype=_GSW_DTYPE)
        if pres.shape[0] != n_depth:
            raise ValueError(f"pres length {pres.shape[0]} != n_depth {n_depth}")
        pres = pres.unsqueeze(1).expand(n_depth, batch)
    else:
        pres = pres_dbar.to(device=device, dtype=_GSW_DTYPE)
        pres, _ = _to_depth_major(pres, pres, pres_hint)

    lat_t = lat.to(device=device, dtype=_GSW_DTYPE).reshape(1, -1).expand(n_depth, -1)
    lon_t = lon.to(device=device, dtype=_GSW_DTYPE).reshape(1, -1).expand(n_depth, -1)
    temp_t = temp.to(dtype=_GSW_DTYPE)
    sal_t = sal.to(dtype=_GSW_DTYPE)

    step = max(1, int(subsample_dz))
    idx = torch.arange(0, n_depth, step, device=device)
    if idx[-1] != n_depth - 1:
        idx = torch.cat([idx, torch.tensor([n_depth - 1], device=device)])

    t_sub = temp_t[idx]
    s_sub = sal_t[idx]
    p_sub = pres[idx]
    lat_sub = lat_t[idx]
    lon_sub = lon_t[idx]

    sa = gsw.SA_from_SP(s_sub, p_sub, lon_sub, lat_sub)
    ct = gsw.CT_from_t(sa, t_sub, p_sub)
    alpha = gsw.specvol_anom_standard(sa, ct, p_sub)

    p_pa = p_sub * 1.0e4
    dp = torch.diff(p_pa, dim=0)
    alpha_mid = 0.5 * (alpha[:-1] + alpha[1:])
    integral = (alpha_mid * dp).sum(dim=0)
    return (integral / _G).to(dtype=temp.dtype)


def compute_clim_steric(
    profiles: Mapping[str, np.ndarray],
    pres: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    *,
    subsample_dz: int = 5,
) -> np.ndarray:
    """Steric height (m) for each sample from depth-major T/S profiles."""
    temp = torch.as_tensor(profiles["temperature"], dtype=torch.float32)
    sal = torch.as_tensor(profiles["salinity"], dtype=torch.float32)
    pres_t = torch.as_tensor(pres, dtype=torch.float32)
    lat_t = torch.as_tensor(lat, dtype=torch.float32)
    lon_t = torch.as_tensor(lon, dtype=torch.float32)
    with torch.no_grad():
        h = steric_height_anomaly(temp, sal, pres_t, lat_t, lon_t, subsample_dz=subsample_dz)
    return h.cpu().numpy().astype(np.float32)


def fit_steric_calibration(
    steric_true: np.ndarray,
    clim_steric: np.ndarray,
    ssh_sla: np.ndarray,
) -> dict[str, float]:
    """Affine fit: ssh_sla ≈ alpha * (steric_true - clim_steric) + beta on train."""
    x = np.asarray(steric_true - clim_steric, dtype=np.float64)
    y = np.asarray(ssh_sla, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 10:
        return {"alpha": 1.0, "beta": 0.0, "r_train": float("nan")}
    xv, yv = x[valid], y[valid]
    A = np.column_stack([xv, np.ones_like(xv)])
    coef, _, _, _ = np.linalg.lstsq(A, yv, rcond=None)
    alpha, beta = float(coef[0]), float(coef[1])
    pred = alpha * xv + beta
    r = float(np.corrcoef(pred, yv)[0, 1]) if yv.size > 1 else float("nan")
    return {"alpha": alpha, "beta": beta, "r_train": r}


class StericConstraint(nn.Module):
    """Calibrated steric-height vs observed SLA penalty."""

    def __init__(self, dataset, device, config):
        super().__init__()
        self.scale = float(config.get("scale", 0.01))
        self.subsample_dz = int(config.get("subsample_dz", 5))
        dev = device or torch.device("cpu")

        self.register_buffer(
            "ssh_obs_sla",
            torch.tensor(np.asarray(dataset.ssh_obs_sla, dtype=np.float32), device=dev),
        )
        self.register_buffer(
            "clim_steric",
            torch.tensor(np.asarray(dataset.clim_steric, dtype=np.float32), device=dev),
        )
        self.register_buffer(
            "lat",
            torch.tensor(np.asarray(dataset.LAT, dtype=np.float32), device=dev),
        )
        self.register_buffer(
            "lon",
            torch.tensor(np.asarray(dataset.LON, dtype=np.float32), device=dev),
        )
        pres = np.asarray(dataset.PRES, dtype=np.float32).ravel()
        self.register_buffer("pres_levels", torch.tensor(pres, device=dev))

        cal = getattr(dataset, "steric_calibration", None) or {}
        self.register_buffer("cal_alpha", torch.tensor(float(cal.get("alpha", 1.0)), device=dev))
        self.register_buffer("cal_beta", torch.tensor(float(cal.get("beta", 0.0)), device=dev))

    def forward(self, temp_profiles, sal_profiles, indices) -> torch.Tensor:
        if indices is None or self.scale <= 0:
            return temp_profiles.new_tensor(0.0)
        idx = torch.as_tensor(indices, dtype=torch.long, device=temp_profiles.device)
        h_pred = steric_height_anomaly(
            temp_profiles,
            sal_profiles,
            self.pres_levels,
            self.lat[idx],
            self.lon[idx],
            subsample_dz=self.subsample_dz,
        )
        steric_anom = h_pred - self.clim_steric[idx]
        pred_sla = self.cal_alpha * steric_anom + self.cal_beta
        resid = pred_sla - self.ssh_obs_sla[idx]
        valid = torch.isfinite(resid)
        if not valid.any():
            return temp_profiles.new_tensor(0.0)
        return self.scale * (resid[valid] ** 2).mean()
