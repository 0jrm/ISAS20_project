#!/usr/bin/env python3
"""Regenerate compare_anom_point_patch.ipynb from notebook modules."""

from __future__ import annotations

import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent / "compare_anom_point_patch.ipynb"


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
        """# NeSPReSO anomaly models: point vs L4 patch

Compare the **anomaly-target** production models (Phase A redesign) against each other
and against the two non-NN baselines, on the chronological 70/15/15 **test** split.

| Row | Prediction | Inputs |
|-----|-----------|--------|
| Climatology | harmonic climatology (train-fit) | (lat, lon, day-of-year) |
| Clim + SLA GEM | climatology + per-depth SLA regression | + DUACS SLA |
| ANOM-point | climatology + NN anomaly PCs | scalar SSS/SST/SSH (`PatchConvMLP`) |
| ANOM-L4-patch | climatology + NN anomaly PCs | 5×5×7-day L4 patches (`PatchMaskConvMLP`) |

Checkpoints are **loaded if trained** (`saved/models/NeSPReSO2_ARGO_GoM_anom*/`), otherwise
**trained as configured** (set `FORCE_RETRAIN=True` to retrain). All physical-space metrics
include the climatology add-back; scalar RMSE uses the common 0–1800 m @ 10 m grid.

**Workflow:** setup → configs → caches & climatology diagnostics → checkpoints →
baselines → summary table + skill → depth structure → scatter/residuals → T–S →
seasonal → maps → example profiles → steric consistency → export."""
    ),
    md("## Section 0 — Setup"),
    code(
        """from __future__ import annotations

import json
import pickle
import sys
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

%matplotlib inline
%load_ext autoreload
%autoreload 2

NOTEBOOK_DIR = Path.cwd().resolve()
TEMPLATE_ROOT = NOTEBOOK_DIR.parent if NOTEBOOK_DIR.name == "notebooks" else NOTEBOOK_DIR / "NeSPReSO2_onTemplate"
sys.path.insert(0, str(TEMPLATE_ROOT))
sys.path.insert(0, str(TEMPLATE_ROOT / "notebooks"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EVAL_SPLIT = "test"
FORCE_RETRAIN = False
OUT_DIR = TEMPLATE_ROOT / "notebooks" / "compare_outputs"
OUT_DIR.mkdir(exist_ok=True)

from nb_configs import (
    PRODUCTION_ANOM_KEYS,
    PRODUCTION_ARGO_SPECS,
    make_production_config_parser,
    production_argo_matrix_row,
)
from nb_checkpoints import checkpoint_epoch
from nb_metrics import (
    avg_common_rmse,
    assert_matches_eval_run,
    plot_depth_rmse_overlay,
    profile_metrics_from_inference,
    statistics_markdown,
)
import nb_anom
from nb_anom import MODEL_COLORS
from train import ensure_cache, main as train_main
from eval_run import main as eval_main

print(f"Device: {DEVICE}")
print(f"Template: {TEMPLATE_ROOT}")"""
    ),
    md(
        """## Section 1 — Statistics contract

Same definitions as the encoding-compare notebook, with one addition: for anomaly caches,
physical profiles are `PCA⁻¹(PCs) + climatology(sample)` (`reconstruct_physical_profiles`),
so every RMSE below is in **physical** space and directly comparable to the raw-target models."""
    ),
    code("print(statistics_markdown())"),
    md("## Section 2 — Config matrix"),
    code(
        """CONFIGS = {
    key: make_production_config_parser(key, template_root=TEMPLATE_ROOT)
    for key in PRODUCTION_ANOM_KEYS
}

print(f"{'label':16s} {'tag':10s} {'arch':18s} {'split':14s} {'latent':>6s}")
print("-" * 70)
for key, cfg in CONFIGS.items():
    row = production_argo_matrix_row(key, cfg)
    print(f"{row['label']:16s} {row['tag']:10s} {row['arch']:18s} {row['split_mode']:14s} {row['latent_dim']:6d}")
    assert cfg.config["io"].get("anomaly_targets"), f"{key}: io.anomaly_targets must be true"
"""
    ),
    md(
        """## Section 3 — Caches + climatology diagnostics

`ensure_cache` builds the anomaly cache if missing (slow first time: DUACS SSH sampling).
Diagnostics: anomaly-PCA explained variance, steric calibration, SLA coverage."""
    ),
    code(
        """caches = {}
for key, cfg in CONFIGS.items():
    path = ensure_cache(cfg)
    with open(path, "rb") as f:
        caches[key] = pickle.load(f)
    c = caches[key]
    cal = c.get("steric_calibration") or {}
    sla = np.asarray(c.get("ssh_obs_sla"))
    print(
        f"{key}: N={c['inputs'].shape[0]}, tag={c.get('dataset_tag')}, cache={Path(path).name}\\n"
        f"    steric cal: alpha={cal.get('alpha', float('nan')):.3f} r_train={cal.get('r_train', float('nan')):.3f}"
        f"  |  SLA finite: {np.isfinite(sla).mean():.1%}"
    )

nb_anom.plot_pca_spectrum(caches)"""
    ),
    md(
        """### 3b — Fitted harmonic climatology (seasonal cycle)

Depth × day-of-year Hovmöller at a central-GoM point — sanity check that the 30-term
basis captured a physically sensible seasonal thermocline."""
    ),
    code(
        """point_key = PRODUCTION_ANOM_KEYS[0]
nb_anom.plot_climatology_cycle(caches[point_key], lat0=25.5, lon0=-90.0)"""
    ),
    md(
        """## Section 4 — Resolve checkpoints (load last trained, else train)

Search order: known experiment dirs (`saved/models/<name>/*/model_best.pth`, newest/most-trained
wins) → config save dir. Early-stopped checkpoints are accepted as final."""
    ),
    code(
        """checkpoints = {}
for key, cfg in CONFIGS.items():
    ckpt, src = nb_anom.resolve_or_load_production(
        key, cfg, train_fn=train_main, force_train=FORCE_RETRAIN, template_root=TEMPLATE_ROOT
    )
    checkpoints[key] = ckpt
    done = checkpoint_epoch(ckpt)
    epoch_note = f" (epoch {done})" if done is not None else ""
    print(f"{key}: {src}{epoch_note} -> {ckpt}")"""
    ),
    md(
        """## Section 5 — Baselines + model inference

Baselines are computed on the **point cache** split (identical profiles/targets as the point
model). Model rows run full inference; native RMSE is cross-checked against `eval_run.py`."""
    ),
    code(
        """point_cfg = CONFIGS[point_key]
rows = nb_anom.baseline_rows(
    caches[point_key], point_cfg.config["data_loader"]["args"], split=EVAL_SPLIT
)

for key, cfg in CONFIGS.items():
    ckpt = checkpoints.get(key)
    if ckpt is None:
        continue
    label = PRODUCTION_ARGO_SPECS[key][0]
    metrics = profile_metrics_from_inference(cfg, str(ckpt), split=EVAL_SPLIT, device=DEVICE)
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
        "checkpoint": str(ckpt),
    }
    try:
        assert_matches_eval_run(cfg, str(ckpt), metrics["raw_profile_rmse_native"], split=EVAL_SPLIT)
        row["eval_run_ok"] = True
    except AssertionError as e:
        row["eval_run_ok"] = False
        print(f"eval_run cross-check {key}: {e}")
    rows.append(row)

model_rows = [r for r in rows if r["arch"] != "baseline"]
print(f"{len(rows)} rows ({len(model_rows)} models + {len(rows) - len(model_rows)} baselines)")"""
    ),
    md(
        """## Section 6 — Summary table + skill vs climatology

Skill = 1 − RMSE/RMSE_clim (common grid). A model that only re-learns the seasonal cycle
scores ~0; the GEM row shows how much of the remaining signal SLA alone explains."""
    ),
    code(
        """nb_anom.summary_table(rows)

if len(model_rows) == 2:
    a, b = model_rows
    delta = b["avg_common_rmse"] - a["avg_common_rmse"]
    pct = 100.0 * delta / a["avg_common_rmse"]
    winner = b["label"] if delta < 0 else a["label"]
    print(f"\\nΔ avg_common ({b['label']} − {a['label']}): {delta:+.4f} ({pct:+.1f}%) → {winner}")

reference = 0.301  # pre-anomaly point model, plan baseline
for r in model_rows:
    print(f"{r['label']}: avg_common={r['avg_common_rmse']:.4f} vs raw-target baseline {reference}")"""
    ),
    md("## Section 7 — Depth structure: RMSE overlay, skill(z), bias(z)"),
    code(
        """plot_depth_rmse_overlay(
    rows,
    colors=MODEL_COLORS,
    out_path=OUT_DIR / "anom_depth_rmse.png",
)
nb_anom.plot_skill_by_depth(rows)
nb_anom.plot_bias_by_depth(rows)"""
    ),
    md("## Section 8 — Predicted vs observed scatter + residual distributions"),
    code(
        """for r in model_rows:
    nb_anom.plot_scatter_depths(r, depths=(0, 100, 300, 800))"""
    ),
    code("nb_anom.plot_residual_hist(rows)"),
    md(
        """## Section 9 — T–S diagrams

Whole-curve fidelity in T–S space (water-mass structure), with σ₀ isopycnals."""
    ),
    code(
        """for r in model_rows:
    nb_anom.plot_ts_diagram(r, n_profiles=40)"""
    ),
    md(
        """## Section 10 — Seasonal breakdown

With anomaly targets the seasonal cycle lives in the climatology; residual month-to-month
RMSE structure indicates season-dependent skill (e.g. summer barrier layers, winter mixing)."""
    ),
    code(
        """V2_SRC = point_cfg.config["io"].get("v2_src")
nb_anom.plot_monthly_rmse(rows, v2_src=V2_SRC)"""
    ),
    md("## Section 11 — Spatial RMSE maps + point-vs-patch difference"),
    code(
        """if len(model_rows) == 2:
    nb_anom.plot_rmse_and_delta_maps(model_rows[0], model_rows[1], variable="temperature", vmax=2.0)
    nb_anom.plot_rmse_and_delta_maps(model_rows[0], model_rows[1], variable="salinity", vmax=0.4)
else:
    print("Need both model rows for delta maps")"""
    ),
    md("## Section 12 — Example profiles (best / median / worst)"),
    code("nb_anom.plot_example_profiles(rows)"),
    md(
        """## Section 13 — Steric / SSH consistency (Phase B diagnostic)

Calibrated steric SLA from **predicted** T/S vs observed DUACS SLA on the test split.
Run before enabling `steric.enabled` — the train calibration `r` is shown for reference,
and higher test-r here means the model's density field is dynamically consistent with SSH."""
    ),
    code("steric_results = nb_anom.plot_steric_consistency(model_rows)"),
    md("## Section 14 — Export results"),
    code(
        """nb_anom.export_results(
    rows, OUT_DIR / "anom_point_patch_results.json", eval_split=EVAL_SPLIT
)"""
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
