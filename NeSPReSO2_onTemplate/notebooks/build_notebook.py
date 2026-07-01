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
        """# NeSPReSO encoding comparison

Compare **PCA-16** vs **AE-128** surface models on a pinned **random 70/15/15** split (seed 42).

| Group | Models |
|-------|--------|
| ISAS | point/patch × PCA-16 / AE-128 |
| ARGO | PCA-15 vs PCA-16 (point PatchConvMLP) |

**Section 8** compares trained **production** ARGO models: point scalars vs L4 spatio-temporal patches (chronological split).

Scalar RMSE uses the **common depth grid** (0–1800 m, 10 m steps). PCA and AE are compared in **profile RMSE space**, not latent dimension count.

**Workflow:** setup → statistics → compare configs → AE decoder artifacts → caches → train/eval → overlay depth curves → best-model maps → **ARGO production point vs L4** → v2 appendix."""
    ),
    md("## Section 0 — Setup"),
    code(
        """from __future__ import annotations

import json
import sys
from pathlib import Path

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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EVAL_SPLIT = "test"
MAX_EPOCHS = 2000
PCA_DIM = 16
AE_DIM = 128
FORCE_RETRAIN = False

from nb_configs import (
    COMPARE_CONFIG_KEYS,
    COMPARE_CONFIGS,
    MANIFEST_PATH,
    compare_matrix_row,
    load_manifest,
    make_compare_config_parser,
)
from nb_checkpoints import checkpoint_epoch, resolve_or_train
from nb_metrics import (
    COMMON_DEPTH_M,
    DEPTH_RANGE_M,
    statistics_markdown,
    profile_metrics_from_inference,
    avg_common_rmse,
    select_best,
    plot_depth_rmse_overlay,
    plot_bin_maps_best,
    assert_matches_eval_run,
    split_indices,
)
from train import ensure_cache, main as train_main
from eval_run import main as eval_main

manifest = load_manifest(MANIFEST_PATH)
print(f"Device: {DEVICE}")
print(f"Template: {TEMPLATE_ROOT}")
print(f"Common depth grid: {DEPTH_RANGE_M[0]}–{DEPTH_RANGE_M[1]} m, {len(COMMON_DEPTH_M)} levels")
if manifest:
    print(f"Manifest: {MANIFEST_PATH} ({len(manifest.get('runs', []))} runs)")"""
    ),
    md("## Section 1 — Statistics contract\n\nEvery table and plot uses **only** these definitions:"),
    code("print(statistics_markdown())"),
    md("## Section 2 — Compare config matrix"),
    code(
        """CONFIGS = {
    key: make_compare_config_parser(key, template_root=TEMPLATE_ROOT)
    for key in COMPARE_CONFIG_KEYS
}

print(f"{'label':16s} {'key':16s} {'tag':8s} {'arch':16s} {'enc':4s} {'latent':>6s} {'epochs':>6s}")
print("-" * 72)
for key, cfg in CONFIGS.items():
    row = compare_matrix_row(key, cfg)
    print(
        f"{row['label']:16s} {key:16s} {row['tag']:8s} {row['arch']:16s} "
        f"{row['encoding']:4s} {row['latent_dim']:6d} {row['epochs']:6d}"
    )"""
    ),
    md(
        """## Section 3 — Profile AE decoders (Phase A)

ISAS surface AE models use frozen decoders trained once via
`scripts/run_encoding_compare_train.sh` Phase A → `saved/decoders/isas20/Autoencoder_dim128/`.

No inline throwaway AE training in this notebook."""
    ),
    code(
        """AE_SUMMARY = TEMPLATE_ROOT / "saved/decoders/isas20/Autoencoder_dim128/summary.json"
if AE_SUMMARY.is_file():
    ae_doc = json.loads(AE_SUMMARY.read_text())
    print(f"AE arch={ae_doc.get('arch')} dim={ae_doc.get('encoding_dim')} cache={Path(ae_doc.get('cache', '')).name}")
    for var, stats in ae_doc.get("variables", {}).items():
        print(
            f"  {var:12s} val_rmse={stats.get('val_rmse', float('nan')):.4f}  "
            f"PCA baseline={stats.get('pca_recon_rmse', float('nan')):.4f}"
        )
else:
    print(f"No AE summary at {AE_SUMMARY} — run scripts/run_encoding_compare_train.sh Phase A first")"""
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
        f"{key}: N={n}, native_depths={n_z}, tag={cache.get('dataset_tag')}, "
        f"cache={Path(path).name}"
    )"""
    ),
    md(
        """## Section 5 — Resolve checkpoints (compare runs only)

Search order: `saved/compare_runs/<key>/model_best.pth` → config save dir.

Reuses checkpoints that reached **`MAX_EPOCHS`**; otherwise resumes/trains with `monitor=min val_loss`.
Production `KNOWN_CHECKPOINTS` are **not** used in compare mode."""
    ),
    code(
        """checkpoints = {}
checkpoint_source = {}

for key, cfg in CONFIGS.items():
    ckpt, src = resolve_or_train(
        key,
        cfg,
        train_fn=train_main,
        max_epochs=MAX_EPOCHS,
        template_root=TEMPLATE_ROOT,
        force_train=FORCE_RETRAIN,
        compare=True,
    )
    checkpoints[key] = ckpt
    checkpoint_source[key] = src
    done = checkpoint_epoch(ckpt)
    epoch_note = f" (epoch {done})" if done is not None else ""
    print(f"{key}: {src}{epoch_note} -> {ckpt}")"""
    ),
    md(
        """## Section 6 — Inference & scalar metrics

**Primary table:** `raw_profile_rmse_common` on the common 0–1800 m grid.

`avg_common_rmse` = mean(T, S) common RMSE — used to rank best model within ISAS / ARGO."""
    ),
    code(
        """summary_rows = []

for key, cfg in CONFIGS.items():
    ckpt = checkpoints.get(key)
    if ckpt is None or not Path(ckpt).exists():
        print(f"{key}: no checkpoint — re-run Section 5 or scripts/run_encoding_compare_train.sh")
        continue

    spec = COMPARE_CONFIGS[key]
    metrics = profile_metrics_from_inference(cfg, str(ckpt), split=EVAL_SPLIT, device=DEVICE)
    eval_report = eval_main(cfg, str(ckpt), split=EVAL_SPLIT)

    row = {
        "key": key,
        "label": spec.label,
        "group": spec.group,
        "tag": metrics["inference"]["dataset_tag"],
        "arch": cfg.config["arch"]["type"],
        "encoding": spec.encoding,
        "n_test": metrics["inference"]["n_samples"],
        "metrics": metrics,
        "T_rmse_common": metrics["raw_profile_rmse_common"]["temperature"],
        "S_rmse_common": metrics["raw_profile_rmse_common"]["salinity"],
        "T_rmse_native": metrics["raw_profile_rmse_native"]["temperature"],
        "S_rmse_native": metrics["raw_profile_rmse_native"]["salinity"],
        "avg_common_rmse": avg_common_rmse(metrics),
        "loss": eval_report["loss"],
    }
    summary_rows.append(row)

    try:
        assert_matches_eval_run(cfg, str(ckpt), metrics["raw_profile_rmse_native"], split=EVAL_SPLIT)
        row["eval_run_ok"] = True
    except AssertionError as e:
        row["eval_run_ok"] = False
        print(f"eval_run cross-check {key}: {e}")

print()
hdr = f"{'label':16s} {'enc':4s} {'avg':>8s} {'T_com':>8s} {'S_com':>8s} {'T_nat':>8s} {'S_nat':>8s}"
print(hdr)
print("-" * len(hdr))
for r in sorted(summary_rows, key=lambda x: (x["group"], x["avg_common_rmse"])):
    print(
        f"{r['label']:16s} {r['encoding']:4s} {r['avg_common_rmse']:8.4f} "
        f"{r['T_rmse_common']:8.4f} {r['S_rmse_common']:8.4f} "
        f"{r['T_rmse_native']:8.4f} {r['S_rmse_native']:8.4f}"
    )

best_isas = select_best(summary_rows, "isas")
best_argo = select_best(summary_rows, "argo")
if best_isas:
    print(f"\\nBest ISAS: {best_isas['label']} avg_common={best_isas['avg_common_rmse']:.4f}")
if best_argo:
    print(f"Best ARGO: {best_argo['label']} avg_common={best_argo['avg_common_rmse']:.4f}")"""
    ),
    md("## Section 7 — Depth RMSE overlay (all 6 models)"),
    code(
        """if summary_rows:
    plot_depth_rmse_overlay(summary_rows)
else:
    print("No checkpoints — skip overlay")"""
    ),
    md(
        """## Section 7b — Spatial maps (best ISAS + best ARGO only)"""
    ),
    code(
        """if summary_rows:
    plot_bin_maps_best(best_isas, best_argo)
else:
    print("No checkpoints — skip maps")"""
    ),
    md(
        """## Section 8 — ARGO production: point vs L4 patch

Compare **trained production** checkpoints (not compare-run re-trains):

| Model | Arch | Inputs | Split |
|-------|------|--------|-------|
| ARGO-point | `PatchConvMLP` | scalar SSS/SST/SSH | chronological 70/15/15 |
| ARGO-L4-patch | `PatchMaskConvMLP` | 5×5×7-day L4 patches + basin + bathy | chronological 70/15/15 |

Checkpoints: `saved/models/NeSPReSO2_ARGO_GoM/` vs `saved/models/NeSPReSO2_ARGO_GoM_patch_l4/`."""
    ),
    code(
        """from nb_configs import (
    PRODUCTION_ARGO_KEYS,
    PRODUCTION_ARGO_SPECS,
    make_production_config_parser,
    production_argo_matrix_row,
)
from nb_checkpoints import checkpoint_epoch, discover_checkpoint

ARGO_PROD = {
    key: make_production_config_parser(key, template_root=TEMPLATE_ROOT)
    for key in PRODUCTION_ARGO_KEYS
}

print(f"{'label':16s} {'tag':10s} {'arch':18s} {'split':14s}")
print("-" * 62)
for key, cfg in ARGO_PROD.items():
    row = production_argo_matrix_row(key, cfg)
    print(f"{row['label']:16s} {row['tag']:10s} {row['arch']:18s} {row['split_mode']:14s}")

argo_prod_ckpts = {}
for key, cfg in ARGO_PROD.items():
    ckpt = discover_checkpoint(key, cfg, template_root=TEMPLATE_ROOT)
    if ckpt is None:
        print(f"{key}: no checkpoint — train with config/argo/config_argo*.json first")
        continue
    argo_prod_ckpts[key] = ckpt
    done = checkpoint_epoch(ckpt)
    epoch_note = f" (epoch {done})" if done is not None else ""
    print(f"{key}: {ckpt}{epoch_note}")"""
    ),
    code(
        """argo_prod_rows = []

for key, cfg in ARGO_PROD.items():
    ckpt = argo_prod_ckpts.get(key)
    if ckpt is None:
        continue
    label = PRODUCTION_ARGO_SPECS[key][0]
    cache_path = ensure_cache(cfg)
    metrics = profile_metrics_from_inference(cfg, str(ckpt), split=EVAL_SPLIT, device=DEVICE)
    eval_report = eval_main(cfg, str(ckpt), split=EVAL_SPLIT)
    row = {
        "key": key,
        "label": label,
        "group": "argo",
        "tag": metrics["inference"]["dataset_tag"],
        "arch": cfg.config["arch"]["type"],
        "n_test": metrics["inference"]["n_samples"],
        "metrics": metrics,
        "T_rmse_common": metrics["raw_profile_rmse_common"]["temperature"],
        "S_rmse_common": metrics["raw_profile_rmse_common"]["salinity"],
        "T_rmse_native": metrics["raw_profile_rmse_native"]["temperature"],
        "S_rmse_native": metrics["raw_profile_rmse_native"]["salinity"],
        "avg_common_rmse": avg_common_rmse(metrics),
        "loss": eval_report["loss"],
        "checkpoint": str(ckpt),
        "cache": cache_path,
    }
    argo_prod_rows.append(row)
    try:
        assert_matches_eval_run(cfg, str(ckpt), metrics["raw_profile_rmse_native"], split=EVAL_SPLIT)
        row["eval_run_ok"] = True
    except AssertionError as e:
        row["eval_run_ok"] = False
        print(f"eval_run cross-check {key}: {e}")

print()
hdr = f"{'label':16s} {'tag':10s} {'avg':>8s} {'T_com':>8s} {'S_com':>8s} {'T_nat':>8s} {'S_nat':>8s}"
print(hdr)
print("-" * len(hdr))
for r in sorted(argo_prod_rows, key=lambda x: x["avg_common_rmse"]):
    print(
        f"{r['label']:16s} {r['tag']:10s} {r['avg_common_rmse']:8.4f} "
        f"{r['T_rmse_common']:8.4f} {r['S_rmse_common']:8.4f} "
        f"{r['T_rmse_native']:8.4f} {r['S_rmse_native']:8.4f}"
    )

if len(argo_prod_rows) == 2:
    pt = next(r for r in argo_prod_rows if r["key"] == "argo_point")
    l4 = next(r for r in argo_prod_rows if r["key"] == "argo_patch_l4")
    delta = l4["avg_common_rmse"] - pt["avg_common_rmse"]
    pct = 100.0 * delta / pt["avg_common_rmse"]
    winner = l4["label"] if delta < 0 else pt["label"]
    print(f"\\nΔ avg_common (L4 − point): {delta:+.4f} ({pct:+.1f}%) — lower is better → {winner}")"""
    ),
    code(
        """if argo_prod_rows:
    plot_depth_rmse_overlay(
        argo_prod_rows,
        colors={"argo_point": "#1f77b4", "argo_patch_l4": "#d62728"},
        out_path=TEMPLATE_ROOT / "notebooks" / "compare_outputs" / "argo_production_depth_rmse.png",
    )
    argo_out = TEMPLATE_ROOT / "notebooks" / "compare_outputs" / "argo_production_results.json"
    argo_out.parent.mkdir(exist_ok=True)
    serializable = [
        {
            "key": r["key"],
            "label": r["label"],
            "tag": r["tag"],
            "arch": r["arch"],
            "checkpoint": r["checkpoint"],
            "cache": str(r["cache"]),
            "n_test": r["n_test"],
            "avg_common_rmse": r["avg_common_rmse"],
            "raw_profile_rmse_common": r["metrics"]["raw_profile_rmse_common"],
            "raw_profile_rmse_native": r["metrics"]["raw_profile_rmse_native"],
            "loss": r["loss"],
        }
        for r in argo_prod_rows
    ]
    argo_out.write_text(json.dumps({"eval_split": EVAL_SPLIT, "models": serializable}, indent=2) + "\\n")
    print(f"Saved {argo_out}")
else:
    print("No ARGO production checkpoints — skip depth overlay")"""
    ),
    md(
        """## Section 9 — v2 appendix (forward parity)

Legacy v2 checkpoint (9 inputs → 30 outputs) vs template `PredictionModel` on ISAS point cache."""
    ),
    code(
        """if V2_REPO.exists() and V2_CHECKPOINT.exists():
    from model.model import PredictionModel as TemplatePredictionModel
    from torch.utils.data import DataLoader, Subset
    from data_loader.data_loaders import NeSPReSODataset, _collate_with_index
    from nb_metrics import v2_checkpoint_dims

    sys.path.insert(0, str(V2_REPO / "src"))
    from nespreso.models.mlp import PredictionModel as V2PredictionModel

    ckpt = torch.load(V2_CHECKPOINT, map_location="cpu", weights_only=False)
    input_dim, layers, out_dim = v2_checkpoint_dims(ckpt)
    print(f"v2 checkpoint: input_dim={input_dim}, layers={layers}, output_dim={out_dim}")

    cfg_legacy = make_compare_config_parser("isas_pt_pca16", template_root=TEMPLATE_ROOT)
    ensure_cache(cfg_legacy)
    import pickle
    with open(cfg_legacy.config["data_loader"]["args"]["cache_path"], "rb") as f:
        cache = pickle.load(f)
    if cache["inputs"].shape[1] != input_dim:
        raise ValueError(f"cache input_dim {cache['inputs'].shape[1]} != checkpoint {input_dim}")

    val_idx = split_indices(cache, "val", dl_args=cfg_legacy.config["data_loader"]["args"])
    ds = NeSPReSODataset(
        torch.tensor(cache["inputs"], dtype=torch.float32),
        torch.tensor(cache["targets"], dtype=torch.float32),
    )
    val_loader = DataLoader(
        Subset(ds, val_idx.tolist()), batch_size=512, shuffle=False, collate_fn=_collate_with_index
    )

    v2m = V2PredictionModel(input_dim, layers, out_dim, 0.2)
    tplm = TemplatePredictionModel(input_dim, layers, out_dim, 0.2)
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
