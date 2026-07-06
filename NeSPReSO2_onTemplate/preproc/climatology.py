"""Harmonic climatology + anomaly-target cache augmentation (Phase A)."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
from sklearn.decomposition import PCA

from base.split_utils import build_split_indices, sample_dates
from model.loss import get_pca_weights
from preproc.ssh_obs import sample_ssh_obs


@dataclass
class Climatology:
    coef: dict[str, np.ndarray]  # var -> (30, n_z)
    pres: np.ndarray
    norm: dict[str, Any]
    meta: dict[str, Any] = field(default_factory=dict)


def _normalize_coords(lat: np.ndarray, lon: np.ndarray, basin: Mapping[str, float]) -> tuple[np.ndarray, np.ndarray]:
    lat_n = 2.0 * (lat - basin["min_lat"]) / (basin["max_lat"] - basin["min_lat"]) - 1.0
    lon_n = 2.0 * (lon - basin["min_lon"]) / (basin["max_lon"] - basin["min_lon"]) - 1.0
    return lat_n.astype(np.float64), lon_n.astype(np.float64)


def _day_of_year_accurate(juld: np.ndarray, *, dataset_tag: str, v2_src: str | None) -> np.ndarray:
    dates = sample_dates(juld, dataset_tag=dataset_tag, v2_src=v2_src)
    years = dates.astype("datetime64[Y]")
    return (dates - years).astype("timedelta64[D]").astype(np.float64) + 1.0


def design_matrix(
    lat: np.ndarray,
    lon: np.ndarray,
    juld: np.ndarray,
    basin: Mapping[str, float],
    *,
    dataset_tag: str,
    v2_src: str | None = None,
    doy: np.ndarray | None = None,
) -> np.ndarray:
    """Tensor-product basis: 5 time × 6 space = 30 columns. ``doy`` overrides JULD decoding."""
    n = len(lat)
    if doy is None:
        doy = _day_of_year_accurate(juld, dataset_tag=dataset_tag, v2_src=v2_src)
    doy = np.asarray(doy, dtype=np.float64)
    ang = 2.0 * np.pi * doy / 365.25
    time_terms = np.column_stack(
        [
            np.ones(n),
            np.cos(ang),
            np.sin(ang),
            np.cos(2.0 * ang),
            np.sin(2.0 * ang),
        ]
    )
    lat_n, lon_n = _normalize_coords(lat, lon, basin)
    space_terms = np.column_stack(
        [
            np.ones(n),
            lat_n,
            lon_n,
            lat_n ** 2,
            lon_n ** 2,
            lat_n * lon_n,
        ]
    )
    return (time_terms[:, :, None] * space_terms[:, None, :]).reshape(n, 30)


def _ridge_solve_multi_depth(X: np.ndarray, Y: np.ndarray, alpha: float) -> np.ndarray:
    """Ridge coefficients for all depth levels, grouping identical valid-masks.

    Depth levels that share an identical NaN pattern across training samples
    (common — e.g. a fixed set of stations shallower than a given depth) share
    one ``Xv``/``XtX`` factorization and are solved as one multi-RHS
    ``np.linalg.solve`` call instead of one solve per depth. Falls back to a
    singleton group (one solve) per depth when every mask is unique, so this is
    never worse than the original per-depth loop, only faster when masks repeat.
    """
    n_z = Y.shape[1]
    beta = np.zeros((X.shape[1], n_z), dtype=np.float64)
    groups: dict[bytes, list[int]] = {}
    for iz in range(n_z):
        key = np.packbits(np.isfinite(Y[:, iz])).tobytes()
        groups.setdefault(key, []).append(iz)

    for idxs in groups.values():
        valid = np.isfinite(Y[:, idxs[0]])
        if valid.sum() < X.shape[1] + 1:
            continue  # beta columns already zero-initialized
        Xv = X[valid]
        Yv = Y[np.ix_(valid, idxs)]
        XtX = Xv.T @ Xv
        reg = alpha * np.eye(XtX.shape[0], dtype=np.float64)
        beta[:, idxs] = np.linalg.solve(XtX + reg, Xv.T @ Yv)
    return beta


def _smooth_coef_vertical(coef: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return coef
    pad = window // 2
    out = np.empty_like(coef)
    for i in range(coef.shape[0]):
        row = coef[i]
        padded = np.pad(row, (pad, pad), mode="edge")
        kernel = np.ones(window, dtype=np.float64) / window
        out[i] = np.convolve(padded, kernel, mode="valid")
    return out


def fit_climatology(
    profiles: Mapping[str, np.ndarray],
    lat: np.ndarray,
    lon: np.ndarray,
    juld: np.ndarray,
    pres: np.ndarray,
    train_idx: np.ndarray | list[int],
    cfg: Mapping[str, Any],
) -> Climatology:
    """Fit per-variable ridge regression on train profiles only."""
    basin = cfg.get("basin") or {
        "min_lat": 18.0,
        "max_lat": 31.0,
        "min_lon": -98.0,
        "max_lon": -81.0,
    }
    dataset_tag = cfg.get("dataset_tag", "argo_v2")
    v2_src = cfg.get("v2_src")
    alpha = float(cfg.get("ridge_alpha", 1.0))
    smooth_window = int(cfg.get("coef_smooth_window", 0))

    X_full = design_matrix(lat, lon, juld, basin, dataset_tag=dataset_tag, v2_src=v2_src)
    X_train = X_full[np.asarray(train_idx, dtype=int)]
    pres = np.asarray(pres, dtype=np.float32).ravel()
    train_idx = np.asarray(train_idx, dtype=int)
    coef: dict[str, np.ndarray] = {}

    for var, prof in profiles.items():
        arr = np.asarray(prof, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"profiles[{var}] must be 2-D depth-major, got {arr.shape}")
        beta = _ridge_solve_multi_depth(X_train, arr[:, train_idx].T, alpha)
        if smooth_window > 1:
            beta = _smooth_coef_vertical(beta.T, smooth_window).T
        coef[var] = beta.astype(np.float32)

    return Climatology(
        coef=coef,
        pres=pres,
        norm={"basin": dict(basin)},
        meta={"dataset_tag": dataset_tag, "ridge_alpha": alpha},
    )


def eval_climatology(
    clim: Climatology,
    lat: np.ndarray,
    lon: np.ndarray,
    juld: np.ndarray,
) -> dict[str, np.ndarray]:
    """Evaluate climatology; returns var -> (n_z, N)."""
    basin = clim.norm["basin"]
    dataset_tag = clim.meta.get("dataset_tag", "argo_v2")
    v2_src = clim.meta.get("v2_src")
    X = design_matrix(lat, lon, juld, basin, dataset_tag=dataset_tag, v2_src=v2_src)
    out: dict[str, np.ndarray] = {}
    for var, beta in clim.coef.items():
        out[var] = (X @ beta).T.astype(np.float32)
    return out


def _profiles_depth_major(
    profiles: Mapping[str, np.ndarray], n: int, pres: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    """Return depth-major profiles (n_z, N). Accepts (N, n_z) or (n_z, N)."""
    n_z = int(np.asarray(pres).ravel().shape[0]) if pres is not None else None
    out = {}
    for name, arr in profiles.items():
        a = np.asarray(arr, dtype=np.float32)
        if a.ndim != 2:
            raise ValueError(f"profile {name} must be 2-D, got {a.shape}")
        if a.shape == (n_z, n):
            out[name] = a
        elif a.shape == (n, n_z):
            out[name] = a.T
        elif a.shape[1] == n and a.shape[0] != n:
            out[name] = a
        elif a.shape[0] == n:
            out[name] = a.T
        else:
            raise ValueError(f"cannot infer depth-major layout for {name}: {a.shape}, n={n}, n_z={n_z}")
    return out


def build_anomaly_targets_block(
    profiles: Mapping[str, np.ndarray],
    lat: np.ndarray,
    lon: np.ndarray,
    juld: np.ndarray,
    pres: np.ndarray,
    outputs: OrderedDict,
    config: dict,
) -> dict[str, Any]:
    """
    Shared cache augmentation when ``io.anomaly_targets`` is enabled.

    Split (train-only clim + PCA), SSH obs, optional steric calibration.
    """
    io_cfg = config.get("io", {}) or {}
    dl_cfg = config.get("data_loader", {}).get("args", {}) or {}
    n = len(lat)
    prof_dm = _profiles_depth_major(profiles, n, pres)

    split_indices = build_split_indices(
        n,
        juld,
        dl_cfg,
        dataset_tag=io_cfg.get("dataset_tag", "unknown"),
        v2_src=io_cfg.get("v2_src"),
    )
    train_idx = np.asarray(split_indices["train"], dtype=int)

    clim_cfg = {
        "basin": io_cfg.get("basin")
        or {"min_lat": 18.0, "max_lat": 31.0, "min_lon": -98.0, "max_lon": -81.0},
        "dataset_tag": io_cfg.get("dataset_tag", "argo_v2"),
        "v2_src": io_cfg.get("v2_src"),
        "ridge_alpha": io_cfg.get("clim_ridge_alpha", 1.0),
        "coef_smooth_window": io_cfg.get("clim_coef_smooth", 0),
    }
    clim = fit_climatology(prof_dm, lat, lon, juld, pres, train_idx, clim_cfg)
    clim_profiles = eval_climatology(clim, lat, lon, juld)

    anomalies = {}
    for name in outputs:
        raw = np.nan_to_num(prof_dm[name], nan=0.0)
        anomalies[name] = raw - clim_profiles[name]

    pca_models = {}
    pcs_by_name = {}
    target_cols = []
    for name, n_comp in outputs.items():
        anom_train = anomalies[name][:, train_idx].T
        fit_x = np.nan_to_num(anom_train, nan=0.0)
        pca = PCA(n_components=n_comp).fit(fit_x)
        pcs_full = pca.transform(np.nan_to_num(anomalies[name].T, nan=0.0)).astype(np.float32).T
        pca_models[name] = pca
        pcs_by_name[name] = pcs_full
        target_cols.append(pcs_full.T.astype(np.float32))
        ev = pca.explained_variance_ratio_.sum()
        print(f"  anomaly PCA {name}: {n_comp} PCs, explained variance={ev:.3f}")

    targets = np.hstack(target_cols).astype(np.float32)
    weights = get_pca_weights(pca_models, pcs_by_name, list(outputs.keys()))

    ssh_adt, ssh_sla = sample_ssh_obs(
        lat,
        lon,
        juld,
        dataset_tag=io_cfg.get("dataset_tag", "argo_v2"),
        v2_src=io_cfg.get("v2_src"),
        max_batch_size=int(io_cfg.get("ssh_batch_size", 200)),
    )

    block: dict[str, Any] = {
        "targets": targets,
        "pca_models": pca_models,
        "pcs_by_name": pcs_by_name,
        "weights": weights,
        "climatology": clim,
        "clim_profiles": clim_profiles,
        "anomaly_targets": True,
        "ssh_obs_adt": ssh_adt,
        "ssh_obs_sla": ssh_sla,
        "split_indices": split_indices,
    }

    if io_cfg.get("steric_at_build", True):
        from model.steric import compute_clim_steric, fit_steric_calibration

        clim_steric = compute_clim_steric(clim_profiles, pres, lat, lon)
        steric_true = compute_clim_steric(
            {k: np.nan_to_num(prof_dm[k], nan=0.0) for k in outputs},
            pres,
            lat,
            lon,
        )
        cal = fit_steric_calibration(
            steric_true[train_idx],
            clim_steric[train_idx],
            ssh_sla[train_idx],
        )
        block["clim_steric"] = clim_steric
        block["steric_calibration"] = cal

    return block
