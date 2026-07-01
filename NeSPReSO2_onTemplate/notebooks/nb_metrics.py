"""Notebook metrics — one contract for every model, dataset, and plot.

All scalar RMSE values in comparison tables use the **common depth grid** unless
the column is explicitly labelled ``native``. Depth curves and maps use the same
grid and depth range so ISAS (187 levels) and ARGO (1801 levels) are comparable.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

import data_loader.data_loaders as module_data
import model.model as module_arch
from model.loss import (
    decode_latent_profiles,
    load_decoders_from_dir,
    sklearn_inverse_transform_pcs,
)
from parse_config import ConfigParser
from base.util import prepare_device
from preproc.overlap import depth_grid_m, interp_profiles
from train import ensure_cache, set_seed, surface_residual_layout_from_cache

# ---------------------------------------------------------------------------
# Statistics contract (documented in notebook Section 1)
# ---------------------------------------------------------------------------

DEPTH_RANGE_M = (0.0, 1800.0)
DEPTH_STEP_M = 10.0
COMMON_DEPTH_M = depth_grid_m(DEPTH_RANGE_M[0], DEPTH_RANGE_M[1], DEPTH_STEP_M)
SPLIT_SEED_DEFAULT = 42
SPLIT_FRACS = (0.70, 0.15, 0.15)
# Fixed 1° GoM grid (matches legacy v2 maps extent: -99..-81°W, 18..30°N)
GOM_LON_BINS = np.arange(-99.5, -80.5, 1.0)
GOM_LAT_BINS = np.arange(17.5, 30.5, 1.0)


@dataclass(frozen=True)
class StatisticDef:
    name: str
    units: str
    definition: str


STATISTICS: tuple[StatisticDef, ...] = (
    StatisticDef(
        "raw_profile_rmse_common",
        "°C or PSU",
        "sqrt(nanmean((pred−true)²)) over all samples in the split and all depth "
        f"levels on the common grid {DEPTH_RANGE_M[0]}–{DEPTH_RANGE_M[1]} m "
        f"(step {DEPTH_STEP_M} m). Truth = cache['profiles'] (raw, not PCA-reconstructed). "
        "Pred = inverse_PCA(model PCs) or frozen-AE decode (decoder mode) interpolated to the common grid.",
    ),
    StatisticDef(
        "raw_profile_rmse_native",
        "°C or PSU",
        "Same as raw_profile_rmse_common but on each cache's native PRES grid "
        "(no interpolation). Matches eval_run.py raw_profile_rmse when the native "
        "grid spans 0–1800 m.",
    ),
    StatisticDef(
        "depth_rmse",
        "°C or PSU",
        f"Per-depth sqrt(nanmean over samples of (pred−true)²) on the common grid "
        f"within {DEPTH_RANGE_M[0]}–{DEPTH_RANGE_M[1]} m.",
    ),
    StatisticDef(
        "depth_bias",
        "°C or PSU",
        f"Per-depth nanmean(pred−true) on the common grid within {DEPTH_RANGE_M[0]}–"
        f"{DEPTH_RANGE_M[1]} m.",
    ),
    StatisticDef(
        "bin_map_rmse",
        "°C or PSU",
        "Per 1° GoM bin: sqrt(nanmean((pred−true)²)) over profiles and common-grid "
        "depths in that bin (fixed -99..-81°W, 18..30°N extent).",
    ),
    StatisticDef(
        "profile_recon_rmse",
        "°C or PSU",
        "Profile autoencoder / PCA reconstruction RMSE on test-split profiles: "
        "sqrt(nanmean((recon−true)²)) on native depths, NaN-masked.",
    ),
    StatisticDef(
        "pca_target_rmse",
        "PC units",
        "RMSE in PCA component space vs target PCs (diagnostic only; not physical).",
    ),
    StatisticDef(
        "training_loss",
        "unitless",
        "Mean CombinedPCALoss on the evaluated split (not comparable across tags).",
    ),
)


def statistics_markdown() -> str:
    lines = ["| Statistic | Units | Definition |", "|---|---|---|"]
    for s in STATISTICS:
        lines.append(f"| `{s.name}` | {s.units} | {s.definition} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def depth_meters(cache: Mapping[str, Any]) -> np.ndarray:
    pres = cache.get("PRES")
    if pres is None:
        n_z = next(iter(cache["profiles"].values())).shape[0]
        return np.arange(n_z, dtype=np.float64)
    return np.asarray(pres, dtype=np.float64).squeeze()


def profiles_depth_major(cache: Mapping[str, Any], name: str) -> np.ndarray:
    """Return (n_depth, n_samples) float array."""
    prof = np.asarray(cache["profiles"][name], dtype=np.float64)
    n = cache["inputs"].shape[0]
    if prof.shape[0] == n:
        prof = prof.T
    if prof.shape[1] != n:
        raise ValueError(f"profiles[{name!r}] shape {prof.shape} inconsistent with N={n}")
    return prof


def align_profiles_to_depth(
    prof: np.ndarray,
    z_src: np.ndarray,
    z_dst: np.ndarray | None = None,
) -> np.ndarray:
    """Interpolate (n_z_src, n_samples) onto z_dst (default: common grid)."""
    z_dst = COMMON_DEPTH_M if z_dst is None else np.asarray(z_dst, dtype=np.float64)
    return interp_profiles(prof, z_src, z_dst)


def common_depth_mask(z: np.ndarray | None = None) -> np.ndarray:
    z = COMMON_DEPTH_M if z is None else np.asarray(z, dtype=np.float64)
    return (z >= DEPTH_RANGE_M[0]) & (z <= DEPTH_RANGE_M[1])


def scalar_rmse(pred: np.ndarray, true: np.ndarray, depth_mask: np.ndarray | None = None) -> float:
    diff = pred - true
    if depth_mask is not None:
        if depth_mask.shape != diff.shape:
            raise ValueError(f"depth_mask shape {depth_mask.shape} != diff {diff.shape}")
        sel = diff[depth_mask]
        if sel.size == 0:
            return float("nan")
        return float(np.sqrt(np.nanmean(sel**2)))
    return float(np.sqrt(np.nanmean(diff**2)))


def bathy_profile_mask(
    z: np.ndarray,
    bottom_depth: np.ndarray,
    sample_indices: np.ndarray | None = None,
) -> np.ndarray:
    """``(n_depth, n_samples)`` mask: levels with ``z <= bottom_depth``."""
    z = np.asarray(z, dtype=np.float64)
    bd = np.asarray(bottom_depth, dtype=np.float64)
    if sample_indices is not None:
        bd = bd[np.asarray(sample_indices, dtype=int)]
    return z[:, np.newaxis] <= bd[np.newaxis, :]


def depth_rmse_bias(pred: np.ndarray, true: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """pred, true: (n_depth, n_samples) on the same depth axis."""
    residual = pred - true
    rmse_d = np.sqrt(np.nanmean(residual**2, axis=1))
    bias_d = np.nanmean(residual, axis=1)
    return rmse_d, bias_d


def raw_profile_rmse_native(
    pred_profiles: Mapping[str, np.ndarray],
    true_profiles: Mapping[str, np.ndarray],
    outputs: OrderedDict,
    *,
    z: np.ndarray | None = None,
    bottom_depth: np.ndarray | None = None,
    sample_indices: np.ndarray | None = None,
) -> dict[str, float]:
    depth_mask = None
    if bottom_depth is not None and z is not None:
        depth_mask = bathy_profile_mask(z, bottom_depth, sample_indices)
    out = {}
    for name in outputs:
        out[name] = scalar_rmse(pred_profiles[name], true_profiles[name], depth_mask)
    return out


def raw_profile_rmse_common(
    pred_profiles: Mapping[str, np.ndarray],
    true_profiles: Mapping[str, np.ndarray],
    z_src: np.ndarray,
    outputs: OrderedDict,
    *,
    bottom_depth: np.ndarray | None = None,
    sample_indices: np.ndarray | None = None,
) -> dict[str, float]:
    mask = common_depth_mask()
    z_common = COMMON_DEPTH_M[mask]
    bd = None
    if bottom_depth is not None and sample_indices is not None:
        bd = np.asarray(bottom_depth, dtype=np.float64)[np.asarray(sample_indices, dtype=int)]
    out = {}
    for name in outputs:
        pred_c = align_profiles_to_depth(pred_profiles[name], z_src, z_common)
        true_c = align_profiles_to_depth(true_profiles[name], z_src, z_common)
        depth_mask = None
        if bd is not None:
            depth_mask = z_common[:, np.newaxis] <= bd[np.newaxis, :]
        out[name] = scalar_rmse(pred_c, true_c, depth_mask)
    return out


def pcs_to_profiles_depth_major(
    pcs: np.ndarray,
    pca_models: Mapping,
    outputs: OrderedDict,
) -> dict[str, np.ndarray]:
    return sklearn_inverse_transform_pcs(pcs, pca_models, outputs)


# ---------------------------------------------------------------------------
# DataLoader split indices (same as NeSPReSODataLoader)
# ---------------------------------------------------------------------------


def split_indices(
    cache: Mapping[str, Any],
    split: str,
    seed: int = SPLIT_SEED_DEFAULT,
    *,
    dl_args: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Same split indices as ``NeSPReSODataLoader`` (random or chronological)."""
    from base.split_utils import build_split_indices

    n = cache["inputs"].shape[0]
    args = dict(dl_args or {})
    args.setdefault("train_frac", SPLIT_FRACS[0])
    args.setdefault("val_frac", SPLIT_FRACS[1])
    args.setdefault("test_frac", SPLIT_FRACS[2])
    args.setdefault("split_mode", "random")
    args.setdefault("split_seed", int(seed))
    indices = build_split_indices(
        n,
        cache.get("JULD"),
        args,
        dataset_tag=cache.get("dataset_tag", "unknown"),
        v2_src=args.get("v2_src"),
    )
    if split not in indices:
        raise ValueError(f"split must be train|val|test, got {split!r}")
    return np.asarray(indices[split], dtype=int)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def run_inference(
    config: ConfigParser,
    checkpoint_path: str,
    *,
    split: str = "test",
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Forward pass on a split; returns PCs, indices, cache, pca_models, outputs."""
    set_seed(config.config.get("seed", SPLIT_SEED_DEFAULT))
    ensure_cache(config)

    dl_args = dict(config["data_loader"]["args"])
    dl_args["split"] = split
    dl_args["shuffle"] = False
    dl_args["split_seed"] = config.config.get("seed", SPLIT_SEED_DEFAULT)
    from eval_run import _resolve_eval_batch_size

    _resolve_eval_batch_size(dl_args, split)
    data_loader = getattr(module_data, config["data_loader"]["type"])(**dl_args)

    if not data_loader.profiles:
        raise ValueError("cache missing 'profiles'; rebuild cache with --force")

    if device is None:
        device, _ = prepare_device(config["n_gpu"])
    model = config.init_obj("arch", module_arch).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state_dict)
    model.eval()

    pca_models = ckpt.get("pca_models", data_loader.pca_models)
    outputs = OrderedDict(ckpt.get("outputs", dict(data_loader.outputs)))

    pcs_list, idx_list = [], []
    with torch.no_grad():
        for data, _target, indices in data_loader:
            out = model(data.to(device))
            pcs_list.append(out.cpu().numpy())
            idx_list.append(indices.cpu().numpy())

    return {
        "pcs": np.vstack(pcs_list),
        "indices": np.concatenate(idx_list),
        "cache": data_loader.cache,
        "cache_path": data_loader.cache_path,
        "pca_models": pca_models,
        "outputs": outputs,
        "dataset_tag": data_loader.dataset_tag,
        "n_samples": len(data_loader.dataset),
    }


def _profile_metrics_from_pred(
    pred: dict[str, np.ndarray],
    indices: np.ndarray,
    cache: Mapping[str, Any],
    outputs: OrderedDict,
) -> dict[str, Any]:
    """Shared metric path once pred profiles are (n_depth, n_samples) per variable."""
    idx = np.asarray(indices, dtype=int)
    z_native = depth_meters(cache)
    true = {name: profiles_depth_major(cache, name)[:, idx] for name in outputs}
    bottom = cache.get("bottom_depth")

    native = raw_profile_rmse_native(
        pred, true, outputs, z=z_native, bottom_depth=bottom, sample_indices=idx
    )
    common = raw_profile_rmse_common(
        pred, true, z_native, outputs, bottom_depth=bottom, sample_indices=idx
    )

    depth_stats = {}
    for name in outputs:
        pred_c = align_profiles_to_depth(pred[name], z_native)
        true_c = align_profiles_to_depth(true[name], z_native)
        pred_c = pred_c[common_depth_mask()]
        true_c = true_c[common_depth_mask()]
        rmse_d, bias_d = depth_rmse_bias(pred_c, true_c)
        depth_stats[name] = {"rmse": rmse_d, "bias": bias_d}

    return {
        "raw_profile_rmse_native": native,
        "raw_profile_rmse_common": common,
        "depth_stats": depth_stats,
        "depth_m_common": COMMON_DEPTH_M[common_depth_mask()],
        "pred_profiles": pred,
        "true_profiles": true,
        "z_native": z_native,
    }


def profile_metrics_from_pcs(
    pcs: np.ndarray,
    indices: np.ndarray,
    cache: Mapping[str, Any],
    pca_models: Mapping,
    outputs: OrderedDict,
) -> dict[str, Any]:
    pred_all = pcs_to_profiles_depth_major(pcs, pca_models, outputs)
    pred = {name: pred_all[name] for name in outputs}
    return _profile_metrics_from_pred(pred, indices, cache, outputs)


def profile_metrics_from_latents(
    latents: np.ndarray,
    indices: np.ndarray,
    cache: Mapping[str, Any],
    outputs: OrderedDict,
    decoders: Mapping[str, torch.nn.Module],
    *,
    device: torch.device,
) -> dict[str, Any]:
    """Decode frozen profile AEs (decoder training mode) then score in physical space."""
    idx = np.asarray(indices, dtype=int)
    latent_t = torch.tensor(latents, dtype=torch.float32, device=device)
    inputs_t = torch.tensor(cache["inputs"][idx], dtype=torch.float32, device=device)
    layout = surface_residual_layout_from_cache(cache)
    decoded = decode_latent_profiles(
        latent_t,
        decoders,
        outputs,
        inputs=inputs_t,
        surface_residual_layout=layout,
    )
    pred = {
        name: decoded[name].detach().cpu().numpy().T  # (n_depth, n_samples)
        for name in outputs
    }
    return _profile_metrics_from_pred(pred, indices, cache, outputs)


def profile_metrics_from_inference(
    config: ConfigParser,
    checkpoint_path: str,
    *,
    split: str = "test",
    device: torch.device | None = None,
) -> dict[str, Any]:
    """PCA-invert or AE-decode automatically from ``loss_config.mode``."""
    inf = run_inference(config, checkpoint_path, split=split, device=device)
    loss_cfg = config.config.get("loss_config") or {}
    if device is None:
        device, _ = prepare_device(config["n_gpu"])
    if loss_cfg.get("mode") == "decoder":
        decoders = load_decoders_from_dir(loss_cfg["decoder_dir"], inf["outputs"], device)
        metrics = profile_metrics_from_latents(
            inf["pcs"],
            inf["indices"],
            inf["cache"],
            inf["outputs"],
            decoders,
            device=device,
        )
    else:
        metrics = profile_metrics_from_pcs(
            inf["pcs"],
            inf["indices"],
            inf["cache"],
            inf["pca_models"],
            inf["outputs"],
        )
    metrics["inference"] = inf
    return metrics


def avg_common_rmse(metrics: Mapping[str, Any]) -> float:
    """Mean of T/S ``raw_profile_rmse_common``."""
    common = metrics["raw_profile_rmse_common"]
    return float(np.mean([common["temperature"], common["salinity"]]))


def select_best(rows: list[dict[str, Any]], group: str) -> dict[str, Any] | None:
    """Argmin by avg common RMSE within ``isas`` or ``argo`` group."""
    pool = [r for r in rows if r.get("group") == group and "metrics" in r]
    if not pool:
        return None
    return min(pool, key=lambda r: r["avg_common_rmse"])


def plot_depth_rmse_overlay(
    rows: list[dict[str, Any]],
    *,
    labels: Mapping[str, str] | None = None,
    colors: Mapping[str, str] | None = None,
    out_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """One figure: T and S depth-RMSE curves for all compare models."""
    import matplotlib.pyplot as plt

    labels = labels or {}
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.suptitle(
        f"Depth RMSE on common grid {DEPTH_RANGE_M[0]}–{DEPTH_RANGE_M[1]} m",
        fontweight="bold",
    )
    for row in rows:
        key = row["key"]
        m = row["metrics"]
        z = m["depth_m_common"]
        color = (colors or {}).get(key)
        label = labels.get(key, row.get("label", key))
        for ax, var, title in zip(
            axes,
            ("temperature", "salinity"),
            ("Temperature RMSE", "Salinity RMSE"),
        ):
            ax.plot(m["depth_stats"][var]["rmse"], z, lw=2, label=label, color=color)
            ax.invert_yaxis()
            ax.set_title(title)
            ax.set_ylabel("Depth [m]")
            ax.grid(True, alpha=0.3)
    axes[0].set_xlabel("RMSE [°C]")
    axes[1].set_xlabel("RMSE [PSU]")
    axes[1].legend(loc="best", fontsize=8)
    plt.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_bin_maps_best(
    best_isas: dict[str, Any] | None,
    best_argo: dict[str, Any] | None,
    *,
    out_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """Four spatial RMSE maps: T/S for best ISAS and best ARGO by avg common RMSE."""
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt

    entries = []
    if best_isas is not None:
        entries.append((best_isas, "ISAS"))
    if best_argo is not None:
        entries.append((best_argo, "ARGO"))
    if not entries:
        print("No best models to map")
        return

    fig, axes = plt.subplots(
        len(entries),
        2,
        figsize=(14, 6 * len(entries)),
        subplot_kw={"projection": ccrs.PlateCarree()},
        squeeze=False,
    )
    for row_i, (row, group_tag) in enumerate(entries):
        inf = row["metrics"]["inference"]
        m = row["metrics"]
        idx = inf["indices"]
        lon = inf["cache"]["LON"][idx]
        lat = inf["cache"]["LAT"][idx]
        for col, var in enumerate(("temperature", "salinity")):
            ax = axes[row_i, col]
            pred_c = align_profiles_to_depth(m["pred_profiles"][var], m["z_native"])[
                common_depth_mask()
            ]
            true_c = align_profiles_to_depth(m["true_profiles"][var], m["z_native"])[
                common_depth_mask()
            ]
            lon_bins, lat_bins, grid_rmse, _nprof = bin_map_scalar_rmse(
                lon, lat, pred_c, true_c
            )
            lon_centers = (lon_bins[:-1] + lon_bins[1:]) / 2
            lat_centers = (lat_bins[:-1] + lat_bins[1:]) / 2
            ax.set_extent([-99, -81, 18, 30])
            ax.coastlines()
            pcm = ax.pcolormesh(
                lon_centers,
                lat_centers,
                grid_rmse,
                cmap="YlOrRd",
                vmin=0.3,
                vmax=2.0,
                transform=ccrs.PlateCarree(),
            )
            unit = "°C" if var == "temperature" else "PSU"
            ax.set_title(f"{group_tag} best ({row['label']}): {var} RMSE [{unit}]")
            fig.colorbar(pcm, ax=ax, orientation="vertical", pad=0.02, fraction=0.046)
    plt.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def assert_matches_eval_run(
    config: ConfigParser,
    checkpoint_path: str,
    notebook_native: Mapping[str, float],
    *,
    split: str = "test",
    rtol: float = 1e-3,
    atol: float = 1e-3,
) -> None:
    from eval_run import main as eval_main

    report = eval_main(config, checkpoint_path, split=split)
    eval_native = report["raw_profile_rmse"]
    for name, eval_val in eval_native.items():
        nb_val = notebook_native[name]
        if not np.isclose(nb_val, eval_val, rtol=rtol, atol=atol):
            raise AssertionError(
                f"{name}: notebook native={nb_val:.6f} vs eval_run={eval_val:.6f}"
            )


def bin_map_scalar_rmse(
    lon: np.ndarray,
    lat: np.ndarray,
    pred_c: np.ndarray,
    true_c: np.ndarray,
    *,
    lon_bins: np.ndarray | None = None,
    lat_bins: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """1° pooled RMSE on common-grid residuals (depth-major pred_c/true_c)."""
    lon_bins = np.asarray(GOM_LON_BINS if lon_bins is None else lon_bins, dtype=np.float64)
    lat_bins = np.asarray(GOM_LAT_BINS if lat_bins is None else lat_bins, dtype=np.float64)
    lon = np.floor(np.asarray(lon, dtype=np.float64)) + 0.5
    lat = np.floor(np.asarray(lat, dtype=np.float64)) + 0.5
    sq = (pred_c - true_c) ** 2
    grid = np.full((len(lat_bins) - 1, len(lon_bins) - 1), np.nan)
    nprof = np.zeros_like(grid)
    for i in range(len(lon_bins) - 1):
        for j in range(len(lat_bins) - 1):
            in_bin = (
                (lon >= lon_bins[i])
                & (lon < lon_bins[i + 1])
                & (lat >= lat_bins[j])
                & (lat < lat_bins[j + 1])
            )
            if not np.any(in_bin):
                continue
            nprof[j, i] = float(np.sum(in_bin))
            grid[j, i] = float(np.sqrt(np.nanmean(sq[:, in_bin])))
    return lon_bins, lat_bins, grid, nprof


def plot_gom_rmse_map(
    lon_bins: np.ndarray,
    lat_bins: np.ndarray,
    grid_rmse: np.ndarray,
    nprof: np.ndarray,
    *,
    model_key: str,
    dataset_tag: str = "",
    variable: str = "Temperature",
) -> None:
    """Cartopy bin map on the fixed GoM grid (one model per figure)."""
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt

    label = f"{model_key} ({dataset_tag})" if dataset_tag else model_key
    title = f"{label}: {variable} RMSE, 1° bins, common depth grid"

    lon_centers = (lon_bins[:-1] + lon_bins[1:]) / 2
    lat_centers = (lat_bins[:-1] + lat_bins[1:]) / 2
    fig, ax = plt.subplots(1, 1, figsize=(12, 12), subplot_kw={"projection": ccrs.PlateCarree()})
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)
    ax.set_extent([-99, -81, 18, 30])
    ax.coastlines()
    pcm = ax.pcolormesh(
        lon_centers,
        lat_centers,
        grid_rmse,
        cmap="YlOrRd",
        vmin=0.3,
        vmax=2.0,
        transform=ccrs.PlateCarree(),
    )
    fig.colorbar(pcm, ax=ax, orientation="vertical", pad=0.04, fraction=0.046, label="RMSE [°C]")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xticks(np.arange(-99, -81, 2))
    ax.set_yticks(np.arange(18, 31, 2))
    ax.grid(color="gray", linestyle="--", linewidth=0.5)
    for i, lon in enumerate(lon_centers):
        for j, lat in enumerate(lat_centers):
            value = grid_rmse[j, i]
            count = nprof[j, i]
            if np.isnan(value) or count <= 0:
                continue
            ax.text(
                lon,
                lat + 0.2,
                f"{count:.0f}",
                color="gray",
                ha="center",
                va="center",
                fontsize=9,
                transform=ccrs.PlateCarree(),
            )
            ax.text(
                lon,
                lat - 0.2,
                f"{value:.2f}",
                color="black",
                ha="center",
                va="center",
                fontsize=9,
                transform=ccrs.PlateCarree(),
            )
    plt.tight_layout()
    plt.show()


def v2_checkpoint_dims(ckpt: dict) -> tuple[int, list[int], int]:
    """Return (input_dim, layers_config, output_dim) from a v2 ``model_state_dict``."""
    sd = ckpt.get("model_state_dict", ckpt)
    weight_keys = sorted(k for k in sd if k.endswith(".weight") and k.startswith("model."))
    if len(weight_keys) < 2:
        raise ValueError("not a v2 PredictionModel checkpoint")
    input_dim = int(sd[weight_keys[0]].shape[1])
    output_dim = int(sd[weight_keys[-1]].shape[0])
    layers = [int(sd[k].shape[0]) for k in weight_keys[:-1]]
    return input_dim, layers, output_dim


# ---------------------------------------------------------------------------
# Profile representation (PCA vs AE)
# ---------------------------------------------------------------------------


def pca_recon_rmse(profiles: np.ndarray, mask: np.ndarray, n_comp: int) -> float:
    from sklearn.decomposition import PCA

    fit_x = np.nan_to_num(profiles, nan=0.0)
    pca = PCA(n_components=n_comp).fit(fit_x)
    recon = pca.inverse_transform(pca.transform(fit_x))
    valid = ~mask
    if not np.any(valid):
        return float("nan")
    err = (recon - profiles) ** 2
    return float(np.sqrt(np.mean(err[valid])))


def ae_recon_rmse(
    profiles: np.ndarray,
    mask: np.ndarray,
    *,
    variable: str,
    arch: str = "Autoencoder",
    encoding_dim: int = 16,
    device: torch.device,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[float, Any]:
    import sys
    from pathlib import Path

    scripts = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from train_profile_ae import train_variable

    _, stats = train_variable(
        profiles=np.nan_to_num(profiles, nan=0.0),
        mask=mask,
        features=None,
        surface_layout=None,
        variable=variable,
        arch=arch,
        encoding_dim=encoding_dim,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        val_frac=val_frac,
        seed=seed,
    )
    return stats["val_rmse"], stats


def representation_metrics_on_split(
    cache: Mapping[str, Any],
    split: str = "test",
    *,
    encoding_dim: int = 16,
    ae_epochs: int = 50,
    device: torch.device,
    seed: int = SPLIT_SEED_DEFAULT,
) -> list[dict[str, Any]]:
    """PCA-X vs AE-X profile reconstruction on the same test profiles."""
    idx = split_indices(cache, split, seed=seed)
    rows = []
    for name in cache["outputs"]:
        prof = profiles_depth_major(cache, name)[:, idx].T  # sample-major for AE
        mask = np.isnan(prof)
        pca_rmse = pca_recon_rmse(prof, mask, encoding_dim)
        ae_rmse, ae_stats = ae_recon_rmse(
            prof,
            mask,
            variable=name,
            encoding_dim=encoding_dim,
            device=device,
            epochs=ae_epochs,
            seed=seed + encoding_dim,
        )
        rows.append(
            {
                "variable": name,
                "encoding_dim": encoding_dim,
                "split": split,
                "n_profiles": len(idx),
                "pca_recon_rmse": pca_rmse,
                "ae_recon_rmse": ae_rmse,
                "ae_over_pca": ae_rmse / pca_rmse if pca_rmse > 0 else float("nan"),
                "ae_stats": ae_stats,
            }
        )
    return rows
