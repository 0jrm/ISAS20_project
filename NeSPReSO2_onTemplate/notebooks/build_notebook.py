#!/usr/bin/env python3
"""Regenerate compare_v2_vs_template.ipynb from notebook modules."""

from __future__ import annotations

import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent / "compare_v2_vs_template.ipynb"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "source": source.splitlines(keepends=True),
        "outputs": [],
        "execution_count": None,
    }


cells = [
    md(
        """# NeSPReSO comparison notebook

End-to-end comparison of **ISAS20** vs **ARGO**, **point** vs **patch** surface models, and **PCA** vs **AE** profile representations.

All comparison tables use the **same statistics contract** (see Section 1). Scalar RMSE in summary tables uses the **common depth grid** (0–1800 m, 10 m steps) so ISAS (187 native levels) and ARGO (1801 levels) are directly comparable. Native-grid RMSE (matches `eval_run.py`) is reported separately.

**Workflow:** setup → statistics contract → profile PCA/AE → cache → smoke train → eval → depth curves & maps → cross-regime → v2 appendix."""
    ),
    md("## Section 0 — Setup"),
    code(
        """from __future__ import annotations

import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

import importlib
import matplotlib.pyplot as plt
import numpy as np
import torch

%matplotlib inline
%load_ext autoreload
%autoreload 2

NOTEBOOK_DIR = Path.cwd().resolve()
if NOTEBOOK_DIR.name == "notebooks":
    TEMPLATE_ROOT = NOTEBOOK_DIR.parent
else:
    TEMPLATE_ROOT = NOTEBOOK_DIR / "NeSPReSO2_onTemplate"
PROJECT_ROOT = TEMPLATE_ROOT.parent

sys.path.insert(0, str(TEMPLATE_ROOT))
sys.path.insert(0, str(TEMPLATE_ROOT / "notebooks"))

V2_REPO = Path("/unity/g2/jmiranda/v2-nespreso")
V2_CHECKPOINT = Path(
    "/unity/g2/jmiranda/SubsurfaceFields/GEM_SubsurfaceFields/saved_models/"
    "model_Test Loss: 0.8847_2024-10-09 20:45:20_sat.pth"
)
V2_DATASET_PICKLE = Path(
    "/unity/g2/jmiranda/SubsurfaceFields/GEM_SubsurfaceFields/config_dataset_full.pkl"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EVAL_SPLIT = "test"
TRAIN_EPOCHS_IF_MISSING = 100  # auto-train when no checkpoint discovered
FORCE_RETRAIN = False          # set True to ignore existing checkpoints
AE_EPOCHS = 50
N_REPEAT = 5

from nb_configs import AE_DEFAULTS, SURFACE_CONFIG_KEYS, make_config_parser
from nb_checkpoints import discover_checkpoint, resolve_or_train
from nb_metrics import (
    COMMON_DEPTH_M,
    DEPTH_RANGE_M,
    STATISTICS,
    statistics_markdown,
    profile_metrics_from_pcs,
    representation_metrics_on_split,
    run_inference,
    assert_matches_eval_run,
    depth_rmse_bias,
    common_depth_mask,
    split_indices,
)
from train import ensure_cache, main as train_main
from eval_run import main as eval_main

print(f"Device: {DEVICE}")
print(f"Template: {TEMPLATE_ROOT}")
print(f"Common depth grid: {DEPTH_RANGE_M[0]}–{DEPTH_RANGE_M[1]} m, {len(COMMON_DEPTH_M)} levels")"""
    ),
    md("## Section 1 — Statistics contract\n\nEvery table and plot in this notebook uses **only** these definitions:"),
    code("print(statistics_markdown())"),
    md("## Section 2 — Inline configs"),
    code(
        """CONFIGS = {key: make_config_parser(key, template_root=TEMPLATE_ROOT) for key in SURFACE_CONFIG_KEYS}

for key, cfg in CONFIGS.items():
    io = cfg.config["io"]
    print(
        f"{key:12s} tag={io['dataset_tag']:8s} arch={cfg.config['arch']['type']:16s} "
        f"outputs={cfg.config['outputs']} epochs={cfg.config['trainer']['epochs']}"
    )"""
    ),
    md(
        """## Section 3 — Profile representation: PCA vs AE

**Statistic:** `profile_recon_rmse` on the **test split** only, native depths, NaN-masked.

Compares PCA-X vs AE-X bottleneck reconstruction (not surface-model prediction)."""
    ),
    code(
        """repr_rows = []
for key in ("isas_patch", "argo_point"):
    cfg = CONFIGS[key]
    ensure_cache(cfg)
    cache_path = cfg.config["data_loader"]["args"]["cache_path"]
    import pickle
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    tag = cache.get("dataset_tag", key)
    for row in representation_metrics_on_split(
        cache, EVAL_SPLIT, encoding_dim=AE_DEFAULTS["encoding_dim"],
        ae_epochs=AE_EPOCHS, device=DEVICE, seed=cfg.config.get("seed", 42),
    ):
        row["config_key"] = key
        row["dataset_tag"] = tag
        repr_rows.append(row)

print(f"{'tag':10s} {'var':12s} {'PCA':>10s} {'AE':>10s} {'AE/PCA':>8s}")
print("-" * 44)
for r in repr_rows:
    print(
        f"{r['dataset_tag']:10s} {r['variable']:12s} {r['pca_recon_rmse']:10.4f} "
        f"{r['ae_recon_rmse']:10.4f} {r['ae_over_pca']:8.3f}"
    )"""
    ),
    md("## Section 4 — Build caches"),
    code(
        """cache_info = {}
for key, cfg in CONFIGS.items():
    path = ensure_cache(cfg)
    import pickle
    with open(path, "rb") as f:
        cache = pickle.load(f)
    cache_info[key] = {"path": path, "cache": cache}
    prof = cache["profiles"]["temperature"]
    n = cache["inputs"].shape[0]
    n_z = prof.shape[0] if prof.shape[1] == n else prof.shape[1]
    print(
        f"{key}: N={cache['inputs'].shape[0]}, native_depths={n_z}, "
        f"cache={Path(path).name}"
    )"""
    ),
    md(
        """## Section 5 — Resolve checkpoints (discover or train)

Search order: known GoM paths → `saved/models/<exper>/…/model_best.pth` → notebook save dir.

If nothing is found (or `FORCE_RETRAIN=True`), trains up to **`TRAIN_EPOCHS_IF_MISSING`** (100) with `monitor=min val_loss`."""
    ),
    code(
        """checkpoints = {}
checkpoint_source = {}

for key, cfg in CONFIGS.items():
    preview = discover_checkpoint(key, cfg, template_root=TEMPLATE_ROOT)
    if preview and not FORCE_RETRAIN:
        checkpoints[key] = preview
        checkpoint_source[key] = "found"
        print(f"{key}: using {preview}")
        continue

    ckpt, src = resolve_or_train(
        key,
        cfg,
        train_fn=train_main,
        max_epochs=TRAIN_EPOCHS_IF_MISSING,
        template_root=TEMPLATE_ROOT,
        force_train=FORCE_RETRAIN,
    )
    checkpoints[key] = ckpt
    checkpoint_source[key] = src
    print(f"{key}: {src} -> {ckpt}")"""
    ),
    md(
        """## Section 6 — Inference & scalar metrics

**Primary comparison table** uses `raw_profile_rmse_common` (common 0–1800 m grid, 181 levels, 10 m step).

| Column | Statistic | Use |
|--------|-----------|-----|
| `T_common` / `S_common` | `raw_profile_rmse_common` | **Cross-model / cross-dataset** comparison |
| `T_native` / `S_native` | `raw_profile_rmse_native` | Matches `eval_run.py` on native PRES grid |

All rows use split=`test`, seed=42, fractions 70/15/15."""
    ),
    code(
        """import importlib
import nb_metrics

importlib.reload(nb_metrics)

summary_rows = []

for key, cfg in CONFIGS.items():
    ckpt = checkpoints.get(key)
    if ckpt is None or not Path(ckpt).exists():
        print(f"{key}: no checkpoint — re-run Section 5")
        continue

    inf = nb_metrics.run_inference(cfg, str(ckpt), split=EVAL_SPLIT, device=DEVICE)
    metrics = nb_metrics.profile_metrics_from_pcs(
        inf["pcs"], inf["indices"], inf["cache"], inf["pca_models"], inf["outputs"]
    )
    eval_report = eval_main(cfg, str(ckpt), split=EVAL_SPLIT)

    row = {
        "config": key,
        "tag": inf["dataset_tag"],
        "arch": cfg.config["arch"]["type"],
        "n_test": inf["n_samples"],
        "T_rmse_common": metrics["raw_profile_rmse_common"]["temperature"],
        "S_rmse_common": metrics["raw_profile_rmse_common"]["salinity"],
        "T_rmse_native": metrics["raw_profile_rmse_native"]["temperature"],
        "S_rmse_native": metrics["raw_profile_rmse_native"]["salinity"],
        "loss": eval_report["loss"],
    }
    summary_rows.append(row)

    try:
        nb_metrics.assert_matches_eval_run(cfg, str(ckpt), metrics["raw_profile_rmse_native"], split=EVAL_SPLIT)
        row["eval_run_ok"] = True
    except AssertionError as e:
        row["eval_run_ok"] = False
        print(f"eval_run cross-check {key}: {e}")

print()
hdr = f"{'config':12s} {'tag':8s} {'T_common':>10s} {'S_common':>10s} {'T_native':>10s} {'S_native':>10s}"
print(hdr)
print("-" * len(hdr))
for r in summary_rows:
    print(
        f"{r['config']:12s} {r['tag']:8s} {r['T_rmse_common']:10.4f} {r['S_rmse_common']:10.4f} "
        f"{r['T_rmse_native']:10.4f} {r['S_rmse_native']:10.4f}"
    )"""
    ),
    md(
        """## Section 7 — Depth curves & spatial maps

Depth curves use `depth_rmse` / `depth_bias` on the **common grid** (same range for all models).

Maps bin spatial RMSE on the same depth-integrated residual field."""
    ),
    code(
        """if not summary_rows:
    print("No checkpoints — skip plots")
else:
    import importlib
    import nb_metrics

    importlib.reload(nb_metrics)
    align_profiles_to_depth = nb_metrics.align_profiles_to_depth
    common_depth_mask = nb_metrics.common_depth_mask

    for row in summary_rows:
        key = row["config"]
        cfg = CONFIGS[key]
        ckpt = checkpoints[key]
        inf = nb_metrics.run_inference(cfg, str(ckpt), split=EVAL_SPLIT, device=DEVICE)
        m = nb_metrics.profile_metrics_from_pcs(
            inf["pcs"], inf["indices"], inf["cache"], inf["pca_models"], inf["outputs"]
        )
        z = m["depth_m_common"]

        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        fig.suptitle(
            f"{key} ({row['tag']}) — common grid {DEPTH_RANGE_M[0]}–{DEPTH_RANGE_M[1]} m",
            fontweight="bold",
        )
        for ax, var, title, unit in [
            (axes[0, 0], "temperature", "Temperature RMSE", "RMSE [°C]"),
            (axes[0, 1], "salinity", "Salinity RMSE", "RMSE [PSU]"),
            (axes[1, 0], "temperature", "Temperature bias", "Bias [°C]"),
            (axes[1, 1], "salinity", "Salinity bias", "Bias [PSU]"),
        ]:
            if "RMSE" in title:
                ax.plot(m["depth_stats"][var]["rmse"], z, lw=2)
            else:
                ax.plot(m["depth_stats"][var]["bias"], z, lw=2)
            ax.invert_yaxis()
            ax.set_title(title)
            ax.set_xlabel(unit)
            ax.set_ylabel("Depth [m]")
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()"""
    ),
    code(
        """# Spatial maps (1° bins) on common-grid depth-integrated residuals
if summary_rows and (checkpoints.get(key)):
    sys.path.insert(0, str(V2_REPO / "src"))
    from nespreso.viz.maps import calculate_average_in_bin, plot_bin_map

    idx = inf["indices"]
    lat = inf["cache"]["LAT"][idx]
    lon = inf["cache"]["LON"][idx]
    lat = np.floor(lat) + 0.5
    lon = np.floor(lon) + 0.5
    lon_bins = np.arange(np.floor(lon.min()) - 0.5, np.ceil(lon.max()) + 1.5, 1.0)
    lat_bins = np.arange(np.floor(lat.min()) - 0.5, np.ceil(lat.max()) + 1.5, 1.0)
    lon_c = lon_bins + 0.5
    lat_c = lat_bins + 0.5

    pred = m["pred_profiles"]["temperature"]
    true = m["true_profiles"]["temperature"]
    z_native = m["z_native"]
    pred_c = align_profiles_to_depth(pred, z_native)[common_depth_mask()]
    true_c = align_profiles_to_depth(true, z_native)[common_depth_mask()]
    res_T = pred_c - true_c
    dpt_idx = np.arange(res_T.shape[0])

    grid_rmse, nprof = calculate_average_in_bin(lon_c, lat_c, lon, lat, res_T, dpt_idx, True)
    plot_bin_map(lon_bins, lat_bins, grid_rmse, nprof, f"{key} temperature", "RMSE (common grid)")"""
    ),
    md(
        """## Section 8 — v2 appendix (forward parity)

The bundled v2 checkpoint is **9 inputs → 512 → 512 → 30 outputs** (15+15 PCs). It does **not** match
current `argo_point` (16+16 = 32 outputs). This cell infers dims from the checkpoint and uses the
`isas_point` cache (9-dim inputs) for the val-split forward pass."""
    ),
    code(
        """# v2 vs template forward pass — PCS should match ~1e-6
if V2_REPO.exists() and V2_CHECKPOINT.exists():
    sys.path.insert(0, str(V2_REPO / "src"))
    from nespreso.models.mlp import PredictionModel as V2PredictionModel
    from model.model import PredictionModel as TemplatePredictionModel
    from torch.utils.data import DataLoader, Subset
    from data_loader.data_loaders import NeSPReSODataset, _collate_with_index
    from nb_metrics import v2_checkpoint_dims

    ckpt = torch.load(V2_CHECKPOINT, map_location="cpu", weights_only=False)
    input_dim, layers, out_dim = v2_checkpoint_dims(ckpt)
    dropout = 0.2
    print(f"v2 checkpoint: input_dim={input_dim}, layers={layers}, output_dim={out_dim}")

    # isas_point cache has 9-dim inputs (point mode); matches v2 ckpt architecture
    cfg = CONFIGS["isas_point"]
    ensure_cache(cfg)
    import pickle
    with open(cfg.config["data_loader"]["args"]["cache_path"], "rb") as f:
        cache = pickle.load(f)
    if cache["inputs"].shape[1] != input_dim:
        raise ValueError(
            f"cache input_dim {cache['inputs'].shape[1]} != checkpoint {input_dim}"
        )

    val_idx = split_indices(cache, "val")
    ds = NeSPReSODataset(
        torch.tensor(cache["inputs"], dtype=torch.float32),
        torch.tensor(cache["targets"], dtype=torch.float32),
    )
    val_loader = DataLoader(
        Subset(ds, val_idx.tolist()), batch_size=512, shuffle=False, collate_fn=_collate_with_index
    )

    v2m = V2PredictionModel(input_dim, layers, out_dim, dropout)
    tplm = TemplatePredictionModel(input_dim, layers, out_dim, dropout)
    v2m.load_state_dict(ckpt["model_state_dict"])
    tplm.load_state_dict(ckpt["model_state_dict"])
    v2m.eval().to(DEVICE)
    tplm.eval().to(DEVICE)

    tpl_pcs, v2_pcs = [], []
    with torch.no_grad():
        for x, _, _ in val_loader:
            x = x.to(DEVICE)
            tpl_pcs.append(tplm(x).cpu().numpy())
            v2_pcs.append(v2m(x).cpu().numpy())
    tpl_pcs = np.vstack(tpl_pcs)
    v2_pcs = np.vstack(v2_pcs)
    print(f"Max |Δ PCS| on val ({len(val_idx)} profiles): {np.max(np.abs(tpl_pcs - v2_pcs)):.3e}")
else:
    print("v2 repo/checkpoint not found — skip appendix")"""
    ),
]

# fix section 4 indentation (no-op guard removed)
for i, c in enumerate(cells):
    if c["cell_type"] == "code" and "align_profiles_to_depth" in "".join(c["source"]):
        src = c["source"]
        if "from nb_metrics import" not in "".join(src):
            c["source"] = ["from nb_metrics import align_profiles_to_depth, common_depth_mask\n"] + src

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "cells": cells,
}

NB_PATH.write_text(json.dumps(nb, indent=1))
print(f"Wrote {NB_PATH} ({len(cells)} cells)")
