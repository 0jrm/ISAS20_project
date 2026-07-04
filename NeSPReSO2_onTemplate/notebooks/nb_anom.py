"""Anomaly point-vs-patch notebook helpers: baselines, skill, classical diagnostics.

Row contract (shared with ``nb_metrics`` plotting): every model/baseline row is a dict
with ``key``, ``label``, ``group``, ``metrics`` (from ``_profile_metrics_from_pred``),
``avg_common_rmse`` and ``metrics["inference"]["indices"|"cache"]`` for spatial plots.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from nb_checkpoints import checkpoint_epoch, discover_checkpoint, _checkpoint_from_config
from nb_metrics import (
    COMMON_DEPTH_M,
    _profile_metrics_from_pred,
    align_profiles_to_depth,
    avg_common_rmse,
    bin_map_scalar_rmse,
    common_depth_mask,
    split_indices,
)

VAR_UNITS = {"temperature": "°C", "salinity": "PSU"}
MODEL_COLORS = {
    "clim": "#7f7f7f",
    "gem": "#bcbd22",
    "argo_anom_point": "#1f77b4",
    "argo_anom_patch_l4": "#d62728",
}


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


def resolve_or_load_production(key, cfg, *, train_fn, force_train=False, template_root=None):
    """Load the last trained checkpoint for a production key, or train as configured.

    Unlike ``resolve_or_train`` this accepts early-stopped checkpoints as final —
    production configs rely on ``early_stop``, not a fixed epoch target.
    """
    found = None if force_train else discover_checkpoint(key, cfg, template_root=template_root)
    if found is not None:
        return found, "found"
    cfg.resume = None
    print(f"{key}: no checkpoint found — training as configured "
          f"({cfg.config['trainer']['epochs']} epochs, early_stop={cfg.config['trainer'].get('early_stop')}) …")
    train_fn(cfg)
    ckpt = discover_checkpoint(key, cfg, template_root=template_root) or _checkpoint_from_config(cfg)
    if ckpt is None or not Path(ckpt).is_file():
        raise FileNotFoundError(f"{key}: training finished but no checkpoint under {cfg.save_dir}")
    return Path(ckpt), "trained"


# ---------------------------------------------------------------------------
# Baseline rows (climatology-only, clim+SLA GEM)
# ---------------------------------------------------------------------------


def _row_from_pred(key, label, pred, idx, cache, outputs):
    metrics = _profile_metrics_from_pred(pred, idx, cache, outputs)
    metrics["inference"] = {
        "indices": np.asarray(idx, dtype=int),
        "cache": cache,
        "dataset_tag": cache.get("dataset_tag"),
        "n_samples": int(len(idx)),
    }
    return {
        "key": key,
        "label": label,
        "group": "argo",
        "tag": cache.get("dataset_tag"),
        "arch": "baseline",
        "n_test": int(len(idx)),
        "metrics": metrics,
        "T_rmse_common": metrics["raw_profile_rmse_common"]["temperature"],
        "S_rmse_common": metrics["raw_profile_rmse_common"]["salinity"],
        "T_rmse_native": metrics["raw_profile_rmse_native"]["temperature"],
        "S_rmse_native": metrics["raw_profile_rmse_native"]["salinity"],
        "avg_common_rmse": avg_common_rmse(metrics),
    }


def baseline_rows(cache, dl_args, *, split="test"):
    """Climatology-only and GEM baseline rows on the same split as the models."""
    if not cache.get("anomaly_targets"):
        raise ValueError("baselines need an anomaly cache (io.anomaly_targets=true)")
    from scripts.eval_baselines import _gem_predict

    outputs = OrderedDict(cache["outputs"])
    test_idx = split_indices(cache, split, dl_args=dl_args)
    train_idx = split_indices(cache, "train", dl_args=dl_args)

    clim_pred = {name: np.asarray(cache["clim_profiles"][name])[:, test_idx] for name in outputs}
    rows = [_row_from_pred("clim", "Climatology", clim_pred, test_idx, cache, outputs)]

    gem_full = _gem_predict(
        train_idx, test_idx, np.asarray(cache["ssh_obs_sla"]), cache["clim_profiles"],
        cache["profiles"], outputs,
    )
    gem_pred = {name: gem_full[name][:, test_idx] for name in outputs}
    rows.append(_row_from_pred("gem", "Clim + SLA GEM", gem_pred, test_idx, cache, outputs))
    return rows


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def summary_table(rows, *, clim_key="clim"):
    """Print RMSE table + skill vs climatology (1 − RMSE/RMSE_clim, common grid)."""
    clim = next((r for r in rows if r["key"] == clim_key), None)
    hdr = (f"{'label':18s} {'tag':10s} {'avg':>8s} {'T_com':>8s} {'S_com':>8s} "
           f"{'T_skill':>8s} {'S_skill':>8s}")
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda x: x["avg_common_rmse"]):
        if clim is not None and r["key"] != clim_key:
            t_sk = 1.0 - r["T_rmse_common"] / clim["T_rmse_common"]
            s_sk = 1.0 - r["S_rmse_common"] / clim["S_rmse_common"]
            skill = f"{t_sk:8.3f} {s_sk:8.3f}"
        else:
            skill = f"{'—':>8s} {'—':>8s}"
        print(f"{r['label']:18s} {str(r['tag']):10s} {r['avg_common_rmse']:8.4f} "
              f"{r['T_rmse_common']:8.4f} {r['S_rmse_common']:8.4f} {skill}")


def export_results(rows, out_path, *, eval_split="test"):
    import json

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = [
        {k: r[k] for k in ("key", "label", "tag", "arch", "n_test",
                           "T_rmse_common", "S_rmse_common",
                           "T_rmse_native", "S_rmse_native", "avg_common_rmse")}
        | {"checkpoint": str(r.get("checkpoint", ""))}
        for r in rows
    ]
    out_path.write_text(json.dumps({"eval_split": eval_split, "models": serializable}, indent=2) + "\n")
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Cache diagnostics
# ---------------------------------------------------------------------------


def plot_pca_spectrum(caches: Mapping[str, Mapping[str, Any]], *, show=True):
    """Cumulative explained variance of the anomaly PCAs, one panel per variable."""
    import matplotlib.pyplot as plt

    first = next(iter(caches.values()))
    var_names = list(first["outputs"].keys())
    fig, axes = plt.subplots(1, len(var_names), figsize=(6 * len(var_names), 4.5))
    for ax, name in zip(np.atleast_1d(axes), var_names):
        for key, cache in caches.items():
            evr = np.cumsum(cache["pca_models"][name].explained_variance_ratio_)
            ax.plot(np.arange(1, len(evr) + 1), evr, marker="o", ms=3, label=key)
        ax.set_title(f"{name} anomaly PCA")
        ax.set_xlabel("n components")
        ax.set_ylabel("cumulative explained variance")
        ax.axhline(0.99, color="gray", ls="--", lw=0.8)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    plt.tight_layout()
    _finish(fig, None, show)


def plot_climatology_cycle(cache, *, lat0=25.5, lon0=-90.0, max_depth=400.0, show=True):
    """Hovmöller of the harmonic climatology (depth × day-of-year) at a fixed point."""
    import matplotlib.pyplot as plt
    from preproc.climatology import design_matrix

    clim = cache["climatology"]
    basin = clim.norm["basin"]
    doy = np.arange(1.0, 366.0, 5.0)
    n = len(doy)
    X = design_matrix(
        np.full(n, lat0), np.full(n, lon0), None, basin,
        dataset_tag=clim.meta.get("dataset_tag", "argo_v2"), doy=doy,
    )
    z = np.asarray(clim.pres, dtype=float)
    zsel = z <= max_depth
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, name in zip(axes, ("temperature", "salinity")):
        field = (X @ clim.coef[name])[:, zsel].T  # (n_z_sel, n_doy)
        pcm = ax.pcolormesh(doy, z[zsel], field, cmap="RdYlBu_r", shading="auto")
        ax.invert_yaxis()
        ax.set_title(f"Climatology {name} at ({lat0}N, {lon0}E)")
        ax.set_xlabel("day of year")
        ax.set_ylabel("Depth [m]")
        fig.colorbar(pcm, ax=ax, label=VAR_UNITS[name])
    plt.tight_layout()
    _finish(fig, None, show)


# ---------------------------------------------------------------------------
# Skill / error structure
# ---------------------------------------------------------------------------


def plot_skill_by_depth(rows, *, clim_key="clim", show=True):
    """Skill(z) = 1 − RMSE_model(z)/RMSE_clim(z); >0 beats climatology at that depth."""
    import matplotlib.pyplot as plt

    clim = next(r for r in rows if r["key"] == clim_key)
    others = [r for r in rows if r["key"] != clim_key]
    z = clim["metrics"]["depth_m_common"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax, name in zip(axes, ("temperature", "salinity")):
        ref = np.maximum(clim["metrics"]["depth_stats"][name]["rmse"], 1e-9)
        for r in others:
            skill = 1.0 - r["metrics"]["depth_stats"][name]["rmse"] / ref
            ax.plot(skill, z, lw=2, label=r["label"], color=MODEL_COLORS.get(r["key"]))
        ax.axvline(0.0, color="k", lw=1)
        ax.invert_yaxis()
        ax.set_title(f"{name} skill vs climatology")
        ax.set_xlabel("1 − RMSE/RMSE_clim")
        ax.set_ylabel("Depth [m]")
        ax.set_xlim(-0.5, 1.0)
        ax.grid(True, alpha=0.3)
    axes[1].legend(loc="best", fontsize=9)
    plt.tight_layout()
    _finish(fig, None, show)


def plot_bias_by_depth(rows, *, show=True):
    """Mean error (bias) profiles on the common grid — complements RMSE overlay."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax, name in zip(axes, ("temperature", "salinity")):
        for r in rows:
            m = r["metrics"]
            ax.plot(m["depth_stats"][name]["bias"], m["depth_m_common"], lw=2,
                    label=r["label"], color=MODEL_COLORS.get(r["key"]))
        ax.axvline(0.0, color="k", lw=1)
        ax.invert_yaxis()
        ax.set_title(f"{name} bias (pred − obs)")
        ax.set_xlabel(f"Bias [{VAR_UNITS[name]}]")
        ax.set_ylabel("Depth [m]")
        ax.grid(True, alpha=0.3)
    axes[1].legend(loc="best", fontsize=9)
    plt.tight_layout()
    _finish(fig, None, show)


def _common_pred_true(row, name):
    m = row["metrics"]
    mask = common_depth_mask()
    pred_c = align_profiles_to_depth(m["pred_profiles"][name], m["z_native"])[mask]
    true_c = align_profiles_to_depth(m["true_profiles"][name], m["z_native"])[mask]
    return pred_c, true_c, COMMON_DEPTH_M[mask]


def plot_scatter_depths(row, *, depths=(0, 100, 300, 800), show=True):
    """Predicted vs observed scatter with 1:1 line at selected depths (classical)."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, len(depths), figsize=(4 * len(depths), 8))
    fig.suptitle(f"{row['label']}: predicted vs observed", fontweight="bold")
    for r_i, name in enumerate(("temperature", "salinity")):
        pred_c, true_c, z_c = _common_pred_true(row, name)
        for c_i, d in enumerate(depths):
            ax = axes[r_i, c_i]
            iz = int(np.argmin(np.abs(z_c - d)))
            x, y = true_c[iz], pred_c[iz]
            ok = np.isfinite(x) & np.isfinite(y)
            x, y = x[ok], y[ok]
            ax.scatter(x, y, s=8, alpha=0.5, color=MODEL_COLORS.get(row["key"]))
            lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
            pad = 0.05 * (hi - lo + 1e-9)
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1)
            rmse = float(np.sqrt(np.mean((y - x) ** 2)))
            r = float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 else np.nan
            ax.set_title(f"{name[0].upper()} @ {z_c[iz]:.0f} m\nRMSE={rmse:.3f}, r={r:.3f}", fontsize=9)
            ax.set_xlabel(f"observed [{VAR_UNITS[name]}]")
            if c_i == 0:
                ax.set_ylabel(f"predicted [{VAR_UNITS[name]}]")
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _finish(fig, None, show)


def plot_residual_hist(rows, *, layers=((0, 200), (200, 1000), (1000, 1800)), show=True):
    """Residual (pred − obs) histograms per depth layer, models overlaid."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, len(layers), figsize=(4.5 * len(layers), 8))
    for r_i, name in enumerate(("temperature", "salinity")):
        for c_i, (z0, z1) in enumerate(layers):
            ax = axes[r_i, c_i]
            for row in rows:
                pred_c, true_c, z_c = _common_pred_true(row, name)
                sel = (z_c >= z0) & (z_c < z1)
                resid = (pred_c[sel] - true_c[sel]).ravel()
                resid = resid[np.isfinite(resid)]
                ax.hist(resid, bins=60, histtype="step", density=True, lw=1.6,
                        label=row["label"], color=MODEL_COLORS.get(row["key"]))
            ax.axvline(0.0, color="k", lw=0.8)
            ax.set_title(f"{name[0].upper()} residuals {z0}–{z1} m", fontsize=10)
            ax.set_xlabel(f"pred − obs [{VAR_UNITS[name]}]")
            ax.set_yscale("log")
            ax.grid(True, alpha=0.3)
    axes[0, -1].legend(fontsize=8)
    plt.tight_layout()
    _finish(fig, None, show)


# ---------------------------------------------------------------------------
# Seasonal breakdown
# ---------------------------------------------------------------------------


def _sample_months(cache, idx, *, v2_src=None):
    from base.split_utils import sample_dates

    juld = np.asarray(cache["JULD"])[np.asarray(idx, dtype=int)]
    dates = sample_dates(juld, dataset_tag=cache.get("dataset_tag", "argo_v2"), v2_src=v2_src)
    return dates.astype("datetime64[M]").astype(int) % 12 + 1


def plot_monthly_rmse(rows, *, v2_src=None, show=True):
    """RMSE by calendar month (common grid) — seasonal-cycle removal check."""
    import matplotlib.pyplot as plt

    months = np.arange(1, 13)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    width = 0.8 / max(1, len(rows))
    for ax, name in zip(axes, ("temperature", "salinity")):
        for k, row in enumerate(rows):
            pred_c, true_c, _ = _common_pred_true(row, name)
            inf = row["metrics"]["inference"]
            mm = _sample_months(inf["cache"], inf["indices"], v2_src=v2_src)
            vals = []
            for m in months:
                sel = mm == m
                if not sel.any():
                    vals.append(np.nan)
                    continue
                diff = pred_c[:, sel] - true_c[:, sel]
                vals.append(float(np.sqrt(np.nanmean(diff ** 2))))
            ax.bar(months + (k - len(rows) / 2 + 0.5) * width, vals, width=width,
                   label=row["label"], color=MODEL_COLORS.get(row["key"]))
        ax.set_title(f"{name} RMSE by month (test)")
        ax.set_xlabel("month")
        ax.set_ylabel(f"RMSE [{VAR_UNITS[name]}]")
        ax.set_xticks(months)
        ax.grid(True, axis="y", alpha=0.3)
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    _finish(fig, None, show)


# ---------------------------------------------------------------------------
# T–S diagram
# ---------------------------------------------------------------------------


def plot_ts_diagram(row, *, n_profiles=40, seed=0, show=True):
    """Observed vs predicted T–S curves for random test profiles, σ₀ isopycnals."""
    import matplotlib.pyplot as plt

    m = row["metrics"]
    t_true, s_true = m["true_profiles"]["temperature"], m["true_profiles"]["salinity"]
    t_pred, s_pred = m["pred_profiles"]["temperature"], m["pred_profiles"]["salinity"]
    rng = np.random.default_rng(seed)
    sel = rng.choice(t_true.shape[1], size=min(n_profiles, t_true.shape[1]), replace=False)

    fig, ax = plt.subplots(figsize=(8, 7))
    _sigma0_isolines(ax, s_true, t_true)
    for j in sel:
        ax.plot(s_true[:, j], t_true[:, j], color="k", lw=0.6, alpha=0.5)
        ax.plot(np.asarray(s_pred)[:, j], np.asarray(t_pred)[:, j],
                color=MODEL_COLORS.get(row["key"], "#d62728"), lw=0.6, alpha=0.5)
    ax.plot([], [], color="k", lw=1.5, label="observed")
    ax.plot([], [], color=MODEL_COLORS.get(row["key"], "#d62728"), lw=1.5, label=row["label"])
    ax.set_xlabel("Salinity [PSU]")
    ax.set_ylabel("Temperature [°C]")
    ax.set_title(f"T–S diagram: {row['label']} ({len(sel)} test profiles)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _finish(fig, None, show)


def _sigma0_isolines(ax, s_arr, t_arr):
    """σ₀ contours via gsw_torch (best effort — skipped if it fails)."""
    try:
        import torch
        import gsw_torch as gsw

        s_arr, t_arr = np.asarray(s_arr), np.asarray(t_arr)
        s_min, s_max = np.nanpercentile(s_arr, [0.5, 99.5])
        t_min, t_max = np.nanpercentile(t_arr, [0.5, 99.5])
        sg = torch.linspace(float(s_min) - 0.2, float(s_max) + 0.2, 80, dtype=torch.float64)
        tg = torch.linspace(float(t_min) - 1, float(t_max) + 1, 80, dtype=torch.float64)
        S, T = torch.meshgrid(sg, tg, indexing="xy")
        p = torch.zeros_like(S)
        with torch.no_grad():
            sa = gsw.SA_from_SP(S, p, torch.full_like(S, -90.0), torch.full_like(S, 25.0))
            ct = gsw.CT_from_t(sa, T, p)
            sigma0 = gsw.rho(sa, ct, p) - 1000.0
        cs = ax.contour(S.numpy(), T.numpy(), sigma0.numpy(), levels=10,
                        colors="gray", linewidths=0.5, alpha=0.7)
        ax.clabel(cs, fontsize=7, fmt="%.1f")
    except Exception as e:  # isolines are decoration — never fail the plot
        print(f"σ₀ isolines skipped: {type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# Example profiles
# ---------------------------------------------------------------------------


def plot_example_profiles(rows, *, clim_key="clim", picks=("best", "median", "worst"), show=True):
    """Observed / climatology / model profiles for best–median–worst test samples.

    Ranking uses the first non-baseline row's per-profile T RMSE.
    """
    import matplotlib.pyplot as plt

    model_rows = [r for r in rows if r["key"] != clim_key and r["arch"] != "baseline"]
    clim_row = next((r for r in rows if r["key"] == clim_key), None)
    ref = model_rows[0]["metrics"]
    pred_c, true_c, z_c = _common_pred_true(model_rows[0], "temperature")
    per_prof = np.sqrt(np.nanmean((pred_c - true_c) ** 2, axis=0))
    order = np.argsort(per_prof)
    pick_idx = {"best": order[0], "median": order[len(order) // 2], "worst": order[-1]}

    fig, axes = plt.subplots(2, len(picks), figsize=(4.5 * len(picks), 10), sharey=True)
    for c_i, pick in enumerate(picks):
        j = int(pick_idx[pick])
        for r_i, name in enumerate(("temperature", "salinity")):
            ax = axes[r_i, c_i]
            _, true_c_v, z_v = _common_pred_true(model_rows[0], name)
            ax.plot(true_c_v[:, j], z_v, "k-", lw=2, label="observed")
            if clim_row is not None:
                clim_c, _, _ = _common_pred_true(clim_row, name)
                ax.plot(clim_c[:, j], z_v, color=MODEL_COLORS["clim"], ls="--", lw=1.5,
                        label="climatology")
            for r in model_rows:
                p_c, _, _ = _common_pred_true(r, name)
                ax.plot(p_c[:, j], z_v, lw=1.5, label=r["label"], color=MODEL_COLORS.get(r["key"]))
            ax.invert_yaxis()
            if r_i == 0:
                inf = ref["inference"]
                gi = int(np.asarray(inf["indices"])[j])
                lat = float(inf["cache"]["LAT"][gi])
                lon = float(inf["cache"]["LON"][gi])
                ax.set_title(f"{pick} (T RMSE {per_prof[j]:.2f})\n({lat:.1f}N, {lon:.1f}E)", fontsize=10)
            ax.set_xlabel(VAR_UNITS[name])
            if c_i == 0:
                ax.set_ylabel("Depth [m]")
            ax.grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    plt.tight_layout()
    _finish(fig, None, show)


# ---------------------------------------------------------------------------
# Spatial maps
# ---------------------------------------------------------------------------


def plot_rmse_and_delta_maps(row_a, row_b, *, variable="temperature", vmax=2.0, show=True):
    """RMSE map for two models + their difference (B − A) on 1° bins."""
    import matplotlib.pyplot as plt

    try:
        import cartopy.crs as ccrs
        proj = {"projection": ccrs.PlateCarree()}
    except Exception:
        ccrs, proj = None, {}

    grids = []
    for row in (row_a, row_b):
        inf = row["metrics"]["inference"]
        idx = np.asarray(inf["indices"], dtype=int)
        lon = np.asarray(inf["cache"]["LON"])[idx]
        lat = np.asarray(inf["cache"]["LAT"])[idx]
        pred_c, true_c, _ = _common_pred_true(row, variable)
        lon_b, lat_b, grid, nprof = bin_map_scalar_rmse(lon, lat, pred_c, true_c)
        grids.append((lon_b, lat_b, grid))

    lon_b, lat_b = grids[0][0], grids[0][1]
    lon_c = (lon_b[:-1] + lon_b[1:]) / 2
    lat_c = (lat_b[:-1] + lat_b[1:]) / 2
    delta = grids[1][2] - grids[0][2]
    unit = VAR_UNITS[variable]

    fig, axes = plt.subplots(1, 3, figsize=(19, 5), subplot_kw=proj)
    panels = [
        (grids[0][2], f"{row_a['label']}: {variable} RMSE [{unit}]", "YlOrRd", 0.0, vmax),
        (grids[1][2], f"{row_b['label']}: {variable} RMSE [{unit}]", "YlOrRd", 0.0, vmax),
        (delta, f"Δ RMSE ({row_b['label']} − {row_a['label']})", "RdBu_r", -0.5 * vmax, 0.5 * vmax),
    ]
    for ax, (grid, title, cmap, vmin, vmx) in zip(axes, panels):
        if ccrs is not None:
            ax.set_extent([-99, -81, 18, 30])
            ax.coastlines()
            kw = {"transform": ccrs.PlateCarree()}
        else:
            kw = {}
        pcm = ax.pcolormesh(lon_c, lat_c, grid, cmap=cmap, vmin=vmin, vmax=vmx, **kw)
        ax.set_title(title, fontsize=10)
        fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.03)
    plt.tight_layout()
    _finish(fig, None, show)


# ---------------------------------------------------------------------------
# Steric / SSH consistency (Phase B diagnostic)
# ---------------------------------------------------------------------------


def steric_consistency(row):
    """Calibrated steric SLA from predicted profiles vs observed DUACS SLA on the split."""
    from model.steric import compute_clim_steric

    inf = row["metrics"]["inference"]
    cache = inf["cache"]
    for key in ("clim_steric", "steric_calibration", "ssh_obs_sla"):
        if cache.get(key) is None:
            raise KeyError(f"cache missing {key!r} — rebuild with steric_at_build")
    idx = np.asarray(inf["indices"], dtype=int)
    m = row["metrics"]
    pred = {name: np.asarray(m["pred_profiles"][name], dtype=np.float32)
            for name in ("temperature", "salinity")}
    h_pred = compute_clim_steric(pred, cache["PRES"],
                                 np.asarray(cache["LAT"])[idx], np.asarray(cache["LON"])[idx])
    cal = cache["steric_calibration"]
    pred_sla = cal["alpha"] * (h_pred - np.asarray(cache["clim_steric"])[idx]) + cal["beta"]
    obs_sla = np.asarray(cache["ssh_obs_sla"])[idx]
    ok = np.isfinite(pred_sla) & np.isfinite(obs_sla)
    r = float(np.corrcoef(pred_sla[ok], obs_sla[ok])[0, 1]) if ok.sum() > 2 else np.nan
    rmse = float(np.sqrt(np.mean((pred_sla[ok] - obs_sla[ok]) ** 2)))
    return {"pred_sla": pred_sla, "obs_sla": obs_sla, "r": r, "rmse": rmse, "cal": dict(cal)}


def plot_steric_consistency(rows, *, show=True):
    """Scatter of calibrated steric SLA (from predicted T/S) vs observed SLA per model."""
    import matplotlib.pyplot as plt

    results = {}
    for row in rows:
        try:
            results[row["key"]] = (row, steric_consistency(row))
        except KeyError as e:
            print(f"{row['key']}: {e}")
    if not results:
        return None
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5.5), squeeze=False)
    for ax, (row, res) in zip(axes[0], results.values()):
        ok = np.isfinite(res["pred_sla"]) & np.isfinite(res["obs_sla"])
        ax.scatter(res["obs_sla"][ok], res["pred_sla"][ok], s=10, alpha=0.5,
                   color=MODEL_COLORS.get(row["key"]))
        lims = ax.get_xlim()
        ax.plot(lims, lims, "k--", lw=1)
        ax.set_title(f"{row['label']}\nr={res['r']:.3f}, RMSE={res['rmse']:.3f} m "
                     f"(train cal r={res['cal'].get('r_train', float('nan')):.3f})", fontsize=10)
        ax.set_xlabel("observed SLA [m]")
        ax.set_ylabel("calibrated steric SLA [m]")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _finish(fig, None, show)
    return {k: v[1] for k, v in results.items()}


def _finish(fig, out_path, show):
    import matplotlib.pyplot as plt

    if out_path is not None:
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
