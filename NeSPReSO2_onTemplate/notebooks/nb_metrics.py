"""Notebook metrics — one contract for every model, dataset, and plot.

All scalar RMSE values in comparison tables use the **common depth grid** unless
the column is explicitly labelled ``native``. Depth curves and maps use the same
grid and depth range so ISAS (187 levels) and ARGO (1801 levels) are comparable.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch

import data_loader.data_loaders as module_data
import model.model as module_arch
from model.loss import sklearn_inverse_transform_pcs
from parse_config import ConfigParser
from playground import prepare_device
from preproc.overlap import depth_grid_m, interp_profiles
from train import ensure_cache, set_seed

# ---------------------------------------------------------------------------
# Statistics contract (documented in notebook Section 1)
# ---------------------------------------------------------------------------

DEPTH_RANGE_M = (0.0, 1800.0)
DEPTH_STEP_M = 10.0
COMMON_DEPTH_M = depth_grid_m(DEPTH_RANGE_M[0], DEPTH_RANGE_M[1], DEPTH_STEP_M)
SPLIT_SEED_DEFAULT = 42
SPLIT_FRACS = (0.70, 0.15, 0.15)


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
        "Pred = inverse_PCA(model PCs) interpolated to the common grid.",
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
        "Spatial bin mean of per-profile scalar RMSE on the common grid (1° bins, "
        "GoM validation/test subset).",
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


def scalar_rmse(pred: np.ndarray, true: np.ndarray) -> float:
    diff = pred - true
    return float(np.sqrt(np.nanmean(diff**2)))


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
) -> dict[str, float]:
    out = {}
    for name in outputs:
        out[name] = scalar_rmse(pred_profiles[name], true_profiles[name])
    return out


def raw_profile_rmse_common(
    pred_profiles: Mapping[str, np.ndarray],
    true_profiles: Mapping[str, np.ndarray],
    z_src: np.ndarray,
    outputs: OrderedDict,
) -> dict[str, float]:
    mask = common_depth_mask()
    z_common = COMMON_DEPTH_M[mask]
    out = {}
    for name in outputs:
        pred_c = align_profiles_to_depth(pred_profiles[name], z_src, z_common)
        true_c = align_profiles_to_depth(true_profiles[name], z_src, z_common)
        out[name] = scalar_rmse(pred_c, true_c)
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


def split_indices(cache: Mapping[str, Any], split: str, seed: int = SPLIT_SEED_DEFAULT) -> np.ndarray:
    import torch
    from torch.utils.data import random_split

    from data_loader.data_loaders import NeSPReSODataset, _split_lengths

    inputs = torch.tensor(cache["inputs"], dtype=torch.float32)
    targets = torch.tensor(cache["targets"], dtype=torch.float32)
    ds = NeSPReSODataset(inputs, targets)
    n = len(ds)
    train_frac, val_frac, test_frac = SPLIT_FRACS
    train_len, val_len, test_len = _split_lengths(n, train_frac, val_frac, test_frac)
    g = torch.Generator().manual_seed(int(seed))
    train_sub, val_sub, test_sub = random_split(ds, [train_len, val_len, test_len], generator=g)
    subsets = {"train": train_sub, "val": val_sub, "test": test_sub}
    if split not in subsets:
        raise ValueError(f"split must be train|val|test, got {split!r}")
    return np.asarray(subsets[split].indices, dtype=int)


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


def profile_metrics_from_pcs(
    pcs: np.ndarray,
    indices: np.ndarray,
    cache: Mapping[str, Any],
    pca_models: Mapping,
    outputs: OrderedDict,
) -> dict[str, Any]:
    idx = np.asarray(indices, dtype=int)
    pred_all = pcs_to_profiles_depth_major(pcs, pca_models, outputs)
    z_native = depth_meters(cache)

    # pred_all columns follow split batch order; idx are global cache indices for truth
    pred = {name: pred_all[name] for name in outputs}
    true = {name: profiles_depth_major(cache, name)[:, idx] for name in outputs}

    native = raw_profile_rmse_native(pred, true, outputs)
    common = raw_profile_rmse_common(pred, true, z_native, outputs)

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
        profiles=profiles,
        mask=mask,
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
