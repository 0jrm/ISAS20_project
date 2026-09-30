#!/usr/bin/env python
"""Regenerate reports/evolution/figs/*.{png,svg} from frozen artifacts only.

No model is run, retrained, or re-scored here. Every number below is copied
verbatim from a source file listed in reports/evolution/PROVENANCE.json for
the corresponding artifact_id. Run under conda env `nespreso`:

    conda run -n nespreso python reports/evolution/make_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Rectangle, Circle
import numpy as np

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figs"
FIGS.mkdir(exist_ok=True)

PASS_COLOR = "#2a7f3f"
FAIL_COLOR = "#b3261e"
NEUTRAL = "#4a5568"
GRID_BG = "#f2f2f2"

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


def savefig(fig, name, svg_only=False):
    fig.tight_layout()
    fig.savefig(FIGS / f"{name}.svg")
    if not svg_only:
        fig.savefig(FIGS / f"{name}.png", dpi=160)
    plt.close(fig)
    print(f"wrote {name}")


# ---------------------------------------------------------------------------
# 1. Lineage DAG  (source: reports/evolution/lineage.json)
# ---------------------------------------------------------------------------

def fig_lineage_dag():
    lineage = json.loads((HERE / "lineage.json").read_text())
    nodes = {n["id"]: n for n in lineage["nodes"]}
    edges = lineage["edges"]

    phases = []
    for n in lineage["nodes"]:
        if n["phase"] not in phases:
            phases.append(n["phase"])
    phase_rank = {p: i for i, p in enumerate(phases)}

    # assign x-slot within a phase row in first-seen order
    row_counts = {}
    pos = {}
    for n in lineage["nodes"]:
        row = phase_rank[n["phase"]]
        col = row_counts.get(row, 0)
        row_counts[row] = col + 1
        pos[n["id"]] = (col, row)
    # center each row
    for row, count in row_counts.items():
        offset = (count - 1) / 2.0
        for nid, (col, r) in pos.items():
            if r == row:
                pos[nid] = (col - offset, row)

    verdict_color = {
        "survivor": PASS_COLOR,
        "killed": FAIL_COLOR,
        "superseded": "#b8860b",
        "erratum": "#5a4fcf",
    }
    edge_style = {
        "led_to": dict(color="#888888", ls="-", lw=1.1),
        "superseded_by": dict(color="#b8860b", ls="--", lw=1.3),
        "erratum_of": dict(color="#5a4fcf", ls=":", lw=1.3),
    }

    n_rows = len(phases)
    max_cols = max(row_counts.values())
    fig, ax = plt.subplots(figsize=(max(14, max_cols * 2.1), n_rows * 1.35 + 1.2))

    xy = {nid: (x * 2.6, -y * 1.9) for nid, (x, y) in pos.items()}

    for e in edges:
        x1, y1 = xy[e["from"]]
        x2, y2 = xy[e["to"]]
        style = edge_style[e["kind"]]
        arrow = FancyArrowPatch((x1, y1 - 0.32), (x2, y2 + 0.32),
                                 arrowstyle="-|>", mutation_scale=10,
                                 connectionstyle="arc3,rad=0.08",
                                 **style, zorder=1, alpha=0.85)
        ax.add_patch(arrow)

    for nid, (x, y) in xy.items():
        n = nodes[nid]
        color = verdict_color[n["verdict"]]
        box = Rectangle((x - 1.15, y - 0.30), 2.3, 0.60,
                         facecolor="white", edgecolor=color, linewidth=2.2,
                         zorder=2)
        ax.add_patch(box)
        label = nid.replace("_", " ")
        ax.text(x, y + 0.08, label, ha="center", va="center", fontsize=6.6,
                wrap=True, zorder=3)
        ax.text(x, y - 0.19, n["verdict"].upper(), ha="center", va="center",
                fontsize=6.0, color=color, fontweight="bold", zorder=3)

    for row, phase in enumerate(phases):
        ax.text(-(max_cols / 2.0) * 2.6 - 1.6, -row * 1.9, phase,
                ha="right", va="center", fontsize=10, fontweight="bold", color=NEUTRAL)

    handles = [mpatches.Patch(facecolor="white", edgecolor=c, linewidth=2, label=k)
               for k, c in verdict_color.items()]
    handles += [plt.Line2D([0], [0], color=s["color"], ls=s["ls"], lw=2, label=k)
                for k, s in edge_style.items()]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.02),
              ncol=len(handles), fontsize=8, frameon=False)

    ax.set_xlim(-(max_cols / 2.0) * 2.6 - 3.4, (max_cols / 2.0) * 2.6 + 1.6)
    ax.set_ylim(-n_rows * 1.9 + 0.8, 1.3)
    ax.axis("off")
    ax.set_title("Experiment lineage DAG (reports/evolution/lineage.json) — top-to-bottom by phase", fontsize=11)
    savefig(fig, "lineage_dag")


# ---------------------------------------------------------------------------
# 2. Matrix gate heatmap (sources: reports/phase5_{A,B,C}_{CRPS,NLL,det}.md)
# ---------------------------------------------------------------------------

MATRIX_LATENT = {
    ("A", "det"): dict(text="T=0.541\n±0.004", gate="PASS", comparable=True),
    ("A", "CRPS"): dict(text="CRPS=1.237\nENCE=0.053", gate="PASS", comparable=False),
    ("A", "NLL"): dict(text="CRPS=1.257\nENCE=0.052", gate="PASS", comparable=False),
    ("B", "det"): dict(text="T=0.534\n±0.001", gate="PASS", comparable=True),
    ("B", "CRPS"): dict(text="CRPS=2.761\nENCE=0.069", gate="PASS", comparable=False),
    ("B", "NLL"): dict(text="CRPS=2.754\nENCE=0.082", gate="PASS", comparable=False),
    ("C", "det"): dict(text="T=0.609\n±0.012", gate="FAIL", comparable=True),
    ("C", "CRPS"): dict(text="CRPS=0.742\nENCE=0.248", gate="FAIL", comparable=False),
    ("C", "NLL"): dict(text="CRPS=0.774\nENCE=0.120", gate="PASS", comparable=False),
}


def fig_matrix_heatmap():
    reps = ["A", "B", "C"]
    heads = ["det", "CRPS", "NLL"]
    rep_label = {"A": "A: separate T/S PCA-16", "B": "B: joint T/S EOF-32", "C": "C: monotone-ρ + spice PCA"}

    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    for i, rep in enumerate(reps):
        for j, head in enumerate(heads):
            cell = MATRIX_LATENT[(rep, head)]
            x, y = j, len(reps) - 1 - i
            color = "#dff3e1" if cell["gate"] == "PASS" else "#fbdede"
            rect = Rectangle((x, y), 1, 1, facecolor=color, edgecolor="#333333", linewidth=1.2)
            ax.add_patch(rect)
            if not cell["comparable"]:
                rect_hatch = Rectangle((x, y), 1, 1, facecolor="none", edgecolor="#888888",
                                        hatch="////", linewidth=0)
                ax.add_patch(rect_hatch)
            ax.text(x + 0.5, y + 0.70, cell["text"], ha="center", va="center", fontsize=8.0)
            gate_color = PASS_COLOR if cell["gate"] == "PASS" else FAIL_COLOR
            ax.text(x + 0.5, y + 0.15, cell["gate"], ha="center", va="center", fontsize=9,
                    fontweight="bold", color=gate_color)
            if rep == "A" and head == "CRPS":
                ring = Circle((x + 0.5, y + 0.5), 0.46, fill=False, edgecolor="#1a1aff",
                               linewidth=3.0, zorder=5)
                ax.add_patch(ring)

    ax.set_xlim(0, len(heads))
    ax.set_ylim(0, len(reps))
    ax.set_xticks([j + 0.5 for j in range(len(heads))])
    ax.set_xticklabels(heads, fontsize=10)
    ax.set_yticks([len(reps) - 1 - i + 0.5 for i in range(len(reps))])
    ax.set_yticklabels([rep_label[r] for r in reps], fontsize=9)
    ax.set_title("Phase 5 matrix — latent-space judgment number per cell\n"
                  "(blue ring = mechanical winner A×CRPS; det cells are physical-space T RMSE already)")
    ax.text(0.0, -0.55, "Hatched cells: PC-space CRPS/NLL judgment number is NOT comparable across "
                          "representations (A: T/S PCA-16, B: joint EOF-32, C: density+spice PCA).\n"
                          "Do not rank hatched cells against each other. See reports/ablation_summary.md "
                          "for the physical-space (comparable) rescoring.",
            transform=ax.transAxes, fontsize=8, color="#333333", va="top")
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)
    savefig(fig, "matrix_gate_heatmap")


# ---------------------------------------------------------------------------
# 3. Depth-band x season heatmaps (source: reports/phase5_A_CRPS_physical_strata.md)
# ---------------------------------------------------------------------------

BANDS = ["0-50", "50-200", "200-800", ">800"]
SEASONS = ["DJF", "MAM", "JJA", "SON"]

CRPS_GRID = np.array([
    [0.2603, 0.2844, 0.3618, 0.2931],
    [0.3446, 0.3441, 0.3753, 0.3144],
    [0.1854, 0.1508, 0.1979, 0.1809],
    [0.0353, 0.0282, 0.0336, 0.0343],
])

ENCE_GRID = np.array([
    [0.3692, 0.3732, 0.6622, 0.4093],
    [0.2521, 0.2103, 0.5560, 0.1826],
    [0.0629, 0.1628, 0.1413, 0.0585],
    [0.3358, 0.5456, 0.3164, 0.4742],
])


def _strata_heatmap(grid, title, name, fail_threshold=None, cmap="viridis"):
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    im = ax.imshow(grid, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(SEASONS)))
    ax.set_xticklabels(SEASONS)
    ax.set_yticks(range(len(BANDS)))
    ax.set_yticklabels(BANDS)
    ax.set_xlabel("season")
    ax.set_ylabel("depth band (m)")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid[i, j]
            txt_color = "white" if im.norm(v) > 0.55 else "black"
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=9, color=txt_color)
            if fail_threshold is not None and v >= fail_threshold:
                rect = Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                  edgecolor=FAIL_COLOR, linewidth=3.0)
                ax.add_patch(rect)
    ax.set_title(title, fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.85)
    savefig(fig, name)


def fig_depthband_season():
    _strata_heatmap(CRPS_GRID, "A×CRPS winner — physical CRPS (T+S) by depth band × season\n"
                                 "source: reports/phase5_A_CRPS_physical_strata.md",
                     "depthband_season_crps", fail_threshold=None, cmap="viridis")
    _strata_heatmap(ENCE_GRID, "A×CRPS winner — physical ENCE(T) by depth band × season\n"
                                 "red border = ENCE(T) ≥ 0.20 gate (surface/thermocline/deep fail — expected)",
                     "depthband_season_ence", fail_threshold=0.20, cmap="magma")


# ---------------------------------------------------------------------------
# 4. OSSE panel (source: reports/osse_results.md, cast_column_s42.json)
# ---------------------------------------------------------------------------

E_LABELS = ["E0", "E1", "E2", "E3", "E4", "E5"]
E_OVERALL = [1.5382, 1.5382, 0.5410, 0.5454, 0.6160, 1.4008]
E_BANDS = {
    "0-100": [2.1986, 2.1986, 1.1934, 1.1510, 1.2119, 1.9488],
    "100-300": [3.1613, 3.1613, 1.0184, 1.0231, 1.1481, 2.8348],
    "300-700": [2.1013, 2.1013, 0.6068, 0.6343, 0.7678, 1.9547],
    ">700": [0.3772, 0.3772, 0.1523, 0.1673, 0.1887, 0.3546],
}


def fig_osse_panel():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1, 1.6]})

    ax = axes[0]
    colors = ["#9aa5b1"] * 6
    ax.bar(E_LABELS, E_OVERALL, color=colors, edgecolor="black")
    for i, v in enumerate(E_OVERALL):
        ax.text(i, v + 0.03, f"{v:.4f}", ha="center", fontsize=8)
    ax.set_ylabel("overall T RMSE (°C), cast columns")
    ax.set_title("Overall — cast-column OSSE, 2021 n=1101")
    ax.annotate("E3 > E2: FAIL\n(0.5454 not < 0.5410)", xy=(3, E_OVERALL[3]), xytext=(2.0, 1.15),
                arrowprops=dict(arrowstyle="->", color=FAIL_COLOR), color=FAIL_COLOR, fontsize=8.5,
                fontweight="bold")
    ax.annotate("E4 ≥ E3: FAIL\n(0.6160 > 0.5454)", xy=(4, E_OVERALL[4]), xytext=(3.6, 1.0),
                arrowprops=dict(arrowstyle="->", color=FAIL_COLOR), color=FAIL_COLOR, fontsize=8.5,
                fontweight="bold")

    ax2 = axes[1]
    x = np.arange(len(E_LABELS))
    width = 0.2
    for k, (band, vals) in enumerate(E_BANDS.items()):
        ax2.bar(x + (k - 1.5) * width, vals, width, label=band)
    ax2.set_xticks(x)
    ax2.set_xticklabels(E_LABELS)
    ax2.set_ylabel("T RMSE (°C)")
    ax2.set_title("By depth band")
    ax2.legend(fontsize=8, title="depth band (m)")

    caveat = ("Caveats: cast-column proxy (truth = ARGO at cast columns, no 2021 ISAS20 grid); "
              "E0≡E1 (background = monthly clim, E1 casts = same clim); "
              "E1–E3 use diagonal-R-only (R_fixed = diag(test RMSE²)).")
    fig.text(0.5, -0.02, caveat, ha="center", fontsize=8.3, color="#333333", wrap=True)
    fig.suptitle("Phase 6 OSSE — E0–E5 (reports/osse_results.md)", fontsize=12)
    savefig(fig, "osse_panel")


# ---------------------------------------------------------------------------
# 5. Ruler sparkline (source: reports/gate_floor_provenance.md)
# ---------------------------------------------------------------------------

def fig_ruler_sparkline():
    points = [
        ("published argo16 T\n(random split)", 0.4158, NEUTRAL),
        ("leaked chrono eval\nof same ckpt", 0.514, FAIL_COLOR),
        ("clean chrono\nargo16 raw T", 0.5367, NEUTRAL),
        ("corrected floor\n(clean × 1.10)", 0.5903, PASS_COLOR),
    ]
    xs = list(range(len(points)))
    ys = [p[1] for p in points]

    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(xs, ys, "-", color="#888888", linewidth=1.6, zorder=1)
    for x, (label, y, color) in zip(xs, points):
        ax.scatter([x], [y], s=140, color=color, zorder=3, edgecolor="black")
        ax.text(x, y + 0.028, f"{y:.4f}", ha="center", fontsize=9, fontweight="bold")
        ax.text(x, y - 0.045, label, ha="center", fontsize=8, color="#333333")
    ax.axhline(0.4574, color="#b8860b", ls="--", lw=1.2)
    ax.text(3.05, 0.4574, "published-random floor 0.4574\n(do NOT use for chrono candidates)",
            fontsize=7.4, color="#8a6d00", va="center")
    ax.set_xlim(-0.5, 4.6)
    ax.set_ylim(0.35, 0.68)
    ax.set_yticks([])
    ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Gate-floor ruler repair (reports/gate_floor_provenance.md)", fontsize=11)
    savefig(fig, "ruler_sparkline")


# ---------------------------------------------------------------------------
# 6. R_cal E-table + depth-band panel (canonical run)
# ---------------------------------------------------------------------------

def fig_rcal_etable_depthband():
    labels = ["E2\nISOP", "E3\nR_fixed", "E4\nR_cal full", "E5\nR_cal+QC"]
    vals = [0.5410, 0.5454, 0.6160, 1.4008]
    colors = [NEUTRAL, NEUTRAL, FAIL_COLOR, "#9aa5b1"]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
    ax = axes[0]
    ax.bar(labels, vals, color=colors, edgecolor="black")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.03, f"{v:.4f}", ha="center", fontsize=9)
    ax.set_ylabel("overall T RMSE (°C)")
    ax.set_title("Canonical cast-column run, 2021 n=1101")
    ax.text(0.02, 0.95, "E3>E2: FAIL\nE4≥E3: FAIL", transform=ax.transAxes, fontsize=9,
            color=FAIL_COLOR, fontweight="bold", va="top")

    ax2 = axes[1]
    bands = list(E_BANDS.keys())
    idx = {"E2": 2, "E3": 3, "E4": 4, "E5": 5}
    x = np.arange(len(bands))
    width = 0.2
    for k, e in enumerate(["E2", "E3", "E4", "E5"]):
        vals_band = [E_BANDS[b][idx[e]] for b in bands]
        ax2.bar(x + (k - 1.5) * width, vals_band, width, label=e)
    ax2.set_xticks(x)
    ax2.set_xticklabels(bands)
    ax2.set_ylabel("T RMSE (°C)")
    ax2.set_title("By depth band (m)")
    ax2.legend(fontsize=8)
    fig.suptitle("R_cal v2 canonical result (reports/osse_results.md)", fontsize=12)
    savefig(fig, "rcal_etable_depthband")


# ---------------------------------------------------------------------------
# 7. Diag-control headline (source: HANDOFF.md, commit 490de67 message)
# ---------------------------------------------------------------------------

def fig_diag_control_headline():
    labels = ["E3\nR_fixed (NeSPReSO, no calibration)", "E4 --rcal diag\n(v1 fallback, diag(Σ_T) only)",
              "E4 --rcal full\n(v2, full localized Σ_T)"]
    vals = [0.5454, 0.5463, 0.6160]
    colors = [NEUTRAL, PASS_COLOR, FAIL_COLOR]

    fig, ax = plt.subplots(figsize=(9.5, 6))
    bars = ax.bar(labels, vals, color=colors, edgecolor="black", width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.018, f"{v:.4f}", ha="center", fontsize=12, fontweight="bold")
    ax.set_ylabel("overall T RMSE (°C), cast columns")
    ax.set_ylim(0.50, 0.68)
    ax.set_title("The structured-covariance strand headline", fontsize=13)

    ax.annotate("", xy=(1, 0.600), xytext=(2, 0.600),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.4))
    ax.text(1.5, 0.610, "gap = CRPS-head\noff-diagonals alone", ha="center", fontsize=9.5,
            fontweight="bold", color=FAIL_COLOR)

    fig.text(0.5, -0.06,
              "diag(Σ) is exactly preserved by the Schur localization, so E3→E4-diag isolates calibrated "
              "marginal variance (helps slightly), while diag→full isolates the cross-level structure alone "
              "(hurts). Localization preserves diag(Σ); the entire 0.546→0.616 degradation is the CRPS-head "
              "off-diagonals alone.",
              ha="center", fontsize=9, wrap=True, color="#222222")
    savefig(fig, "diag_control_headline")


# ---------------------------------------------------------------------------
# 8. R-construction schematic
# ---------------------------------------------------------------------------

def fig_rcal_schematic():
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.4)

    ax.text(5, 6.05, r"$\Sigma_T = V\,\mathrm{diag}((\alpha\sigma)^2)\,V^T$"
                       r"   rank 16 over $n_z=60$ levels   →   ~44-dim near-null space",
            ha="center", fontsize=12)

    def matrix_box(x, y, w, h, label, sub, color):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=color, edgecolor="black", linewidth=1.3))
        ax.text(x + w / 2, y + h / 2 + 0.15, label, ha="center", va="center", fontsize=12, fontweight="bold")
        ax.text(x + w / 2, y + h / 2 - 0.28, sub, ha="center", va="center", fontsize=8.5)

    matrix_box(0.4, 3.6, 1.2, 1.6, "V", "60×16\n(PCA-16 basis,\nphase5_A_CRPS.md)", "#dfe7f5")
    ax.text(1.75, 4.4, r"$\times$", fontsize=16, ha="center")
    matrix_box(2.0, 3.9, 1.0, 1.0, "diag((ασ)²)", "16×16\nCRPS-head\nmarginal var", "#f5e6c8")
    ax.text(3.15, 4.4, r"$\times$", fontsize=16, ha="center")
    matrix_box(3.4, 3.6, 1.2, 1.6, r"$V^T$", "16×60", "#dfe7f5")
    ax.text(4.8, 4.4, "=", fontsize=16, ha="center")
    matrix_box(5.1, 3.4, 1.8, 2.0, r"raw $\Sigma_T$", "60×60, rank 16\n~44-dim near-null\ncond(B+R)≈2e8\n(HANDOFF.md: blows up)", "#f7cccc")

    ax.annotate("", xy=(5.1, 2.9), xytext=(3.9, 3.4),
                arrowprops=dict(arrowstyle="->", lw=1.6, color="black"))
    matrix_box(4.6, 1.0, 2.0, 1.6, r"Schur-localized $R$",
               r"$(\Sigma_T\circ\rho_L)+\mathrm{floor}\cdot I$" "\n"
               r"$\rho_{ij}=\exp(-\frac{1}{2}((z_i-z_j)/L_{loc})^2)$" "\n"
               "L_loc = L_v = 150 m\nfull rank restored", "#d7ecd9")
    ax.annotate("Schur-Hadamard\nlocalization", xy=(5.6, 2.6), xytext=(6.9, 3.0),
                fontsize=8.5, ha="center", arrowprops=dict(arrowstyle="-", color="#555555", lw=0.8))

    ax.text(5, 0.35,
            "Localization is numerically sound — full rank / OI-stable (HANDOFF.md). The structure it\n"
            "preserves is the problem: cross-level terms are basis-induced (shared V), not learned\n"
            "obs-error correlation. Synthetic conditioning numbers for the localized case were requested\n"
            "but are not in a committed source file (see PROVENANCE.json unsourced[]).",
            ha="center", fontsize=8.0, color="#333333")

    ax.set_title("R-construction schematic (reports/osse_preregistration.md §2, HANDOFF.md)", fontsize=11)
    savefig(fig, "rcal_schematic")


# ---------------------------------------------------------------------------
# Step 1b — datacube documentation strand
# ---------------------------------------------------------------------------

def fig_cube_schematic():
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)

    # isometric-ish stacked slabs
    def slab(x, y, w, h, dx, dy, label, color):
        top = np.array([(x, y), (x + w, y), (x + w + dx, y + dy), (x + dx, y + dy)])
        ax.add_patch(plt.Polygon(top, closed=True, facecolor=color, edgecolor="black", linewidth=0.9))
        ax.text(x + w / 2 + dx / 2, y + dy / 2 + h * 0 + 0.02, label, ha="center", va="center", fontsize=8.5)

    channels = [
        ("sst  (OSTIA analysed_sst)", "#ffd9b3"),
        ("sss  (CMEMS sos)", "#c8e6c9"),
        ("ssh: adt / sla / ugos / vgos", "#b3d9ff"),
        ("bathymetry (GEBCO elevation)", "#d9d9d9"),
        ("v3 error: analysis_error / sos_error / err_sla", "#f5b3b3"),
    ]
    x0, y0 = 1.6, 1.2
    dx, dy = 1.3, 0.55
    gap = 0.85
    for i, (label, color) in enumerate(channels):
        slab(x0, y0 + i * gap, 3.2, 0.5, dx, dy, label, color)

    ax.annotate("", xy=(x0 + 0.2, y0 - 0.6), xytext=(x0 + 0.2, y0 + len(channels) * gap + 0.3),
                arrowprops=dict(arrowstyle="-", color="black", lw=1.2))
    ax.text(x0 - 0.35, y0 + len(channels) * gap / 2, "channel\nstack", fontsize=8.5, ha="center", rotation=90)

    ax.annotate("lon (-98..-81°W)", xy=(x0, y0 - 0.55), xytext=(x0 + 3.6, y0 - 0.55),
                arrowprops=dict(arrowstyle="->", lw=1.2), fontsize=9, ha="left", va="center")
    ax.annotate("lat (18..31°N)", xy=(x0, y0 - 0.55), xytext=(x0 + dx + 0.3, y0 + dy + 0.3),
                arrowprops=dict(arrowstyle="->", lw=1.2), fontsize=9, ha="left", va="center")
    ax.text(x0 + dx - 0.1, y0 + dy + 0.55, "time\n(2015-01-01 .. 2022-03-01, daily)", fontsize=8.5, ha="center")

    # profile pin + patch highlight on top slab
    top_i = len(channels) - 1
    px, py = x0 + 2.0, y0 + top_i * gap + 0.25 + 0.35 * (dy / dy)
    pin_x = x0 + 2.0 + dx * 0.55
    pin_y = y0 + top_i * gap + 0.25 + dy * 0.55
    ax.plot([pin_x], [pin_y], marker="v", markersize=14, color="#8b0000", zorder=6)
    ax.text(pin_x, pin_y + 0.35, "ARGO profile\n(lat, lon, time)", fontsize=8, ha="center", color="#8b0000")
    patch = plt.Circle((pin_x, pin_y), 0.32, fill=False, edgecolor="#0033cc", linewidth=2.2, zorder=6)
    ax.add_patch(patch)
    ax.text(pin_x + 0.65, pin_y - 0.35, "extracted patch\n(FeatureSampler\nbilinear weights)", fontsize=7.6,
            color="#0033cc")

    ax.set_title("GoM regional cube — channel stack, extraction pin (utils/v2.json, utils/v3.json,\n"
                 "NeSPReSO2_onTemplate/preproc/cube/cube_schema.py)", fontsize=10.5)
    savefig(fig, "cube_schematic", svg_only=True)


def fig_cube_extraction_inset():
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))

    ax = axes[0]
    ax.set_title("Plan view — spatial_pad = 20 cells", fontsize=10.5)
    ax.add_patch(Rectangle((-20, -20), 40, 40, facecolor="#e8f0fe", edgecolor="#888888"))
    ax.plot(0, 0, marker="v", color="#8b0000", markersize=14, zorder=5)
    ax.text(0, 2, "ARGO cast (lat, lon)", ha="center", fontsize=8.5, color="#8b0000")
    ax.add_patch(Rectangle((-20, -20), 40, 40, fill=False, edgecolor="#0033cc", linewidth=2))
    ax.annotate("", xy=(-20, -25), xytext=(20, -25), arrowprops=dict(arrowstyle="<->"))
    ax.text(0, -27, "spatial_pad = 20 grid cells\n(utils/v2.json, utils/v3.json)", ha="center", fontsize=8.5)
    ax.set_xlim(-30, 30)
    ax.set_ylim(-32, 22)
    ax.set_xlabel("Δ grid cells (lon)")
    ax.set_ylabel("Δ grid cells (lat)")

    ax2 = axes[1]
    ax2.set_title("Time strip — temporal_pad = 6 days", fontsize=10.5)
    days = np.arange(-6, 7)
    ax2.scatter(days, np.zeros_like(days), color="#888888", s=40)
    ax2.scatter([0], [0], color="#8b0000", s=140, zorder=5, marker="v")
    ax2.text(0, 0.15, "cast day t0", ha="center", fontsize=8.5, color="#8b0000")
    ax2.axvspan(-6, 6, color="#0033cc", alpha=0.08)
    ax2.annotate("", xy=(-6, -0.25), xytext=(6, -0.25), arrowprops=dict(arrowstyle="<->", color="#0033cc"))
    ax2.text(0, -0.4, "temporal_pad = 6 days each side\n(utils/v2.json, utils/v3.json)", ha="center", fontsize=8.5)
    ax2.set_ylim(-0.6, 0.4)
    ax2.set_yticks([])
    ax2.set_xlabel("day offset from cast")

    fig.suptitle("Legacy per-station HDF5 extraction geometry (pads unchanged v2→v3: v3.json _notes.pads)",
                 fontsize=11)
    savefig(fig, "cube_extraction_inset", svg_only=True)


def fig_cube_dataflow():
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.axis("off")
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5)

    def box(x, y, w, h, text, color):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=color, edgecolor="black", linewidth=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.6, wrap=True)

    box(0.2, 2.9, 2.6, 1.6, "BEFORE\nPer-station HDF5 patches\nfixed spatial_pad=20 cells /\ntemporal_pad=6 days\n(utils/v2.json, v3.json)", "#f7dede")
    ax.annotate("", xy=(3.2, 3.7), xytext=(2.8, 3.7), arrowprops=dict(arrowstyle="->", lw=1.6))
    box(3.3, 2.9, 2.8, 1.6, "AFTER\nRegional Zarr cube\ngom_cube.zarr, daily,\n2015-01-01..2022-03-01\n(preproc/cube/build_cube.py)", "#dbeadb")
    ax.annotate("", xy=(6.5, 3.7), xytext=(6.1, 3.7), arrowprops=dict(arrowstyle="->", lw=1.6))
    box(6.6, 2.9, 3.2, 1.6, "On-demand interpolation\nFeatureSampler: bilinear\nweights per (lat,lon,time)\n(preproc/features/sampler.py)", "#dbe6f7")

    for i, tgt in enumerate(["point model", "patch model", "residual model"]):
        yb = 0.4 + i * 0.0
        ax.annotate("", xy=(1.6 + i * 3.4, 2.85), xytext=(8.2, 2.85),
                    arrowprops=dict(arrowstyle="-", lw=0)) if False else None
    box(0.4, 0.4, 2.6, 1.1, "point model\n(dissertation winner:\nA×CRPS)", "#eef2ff")
    box(3.6, 0.4, 2.6, 1.1, "patch model", "#eef2ff")
    box(6.8, 0.4, 2.6, 1.1, "residual model", "#eef2ff")
    for x0 in (1.7, 4.9, 8.1):
        ax.annotate("", xy=(x0, 1.5), xytext=(8.2, 2.9),
                    arrowprops=dict(arrowstyle="->", lw=1.1, color="#666666",
                                     connectionstyle="arc3,rad=0.15"))

    ax.text(5.5, 4.65,
            "Datacube contribution: data-quality fix + unified extraction geometry + enabling of v3 error "
            "channels — NOT a claim that patches beat points.",
            ha="center", fontsize=9, color="#333333")
    ax.text(5.5, -0.15,
            "The dissertation-winning model (A×CRPS, Phase 5) is a point model. The patch/residual branch "
            "did not win the matrix (see reports/ablation_summary.md).",
            ha="center", fontsize=8.6, color=FAIL_COLOR)

    ax.set_title("Before/after data-flow (Step 1b)", fontsize=11)
    savefig(fig, "cube_dataflow", svg_only=True)


def fig_cube_stale_fingerprint():
    stale = json.loads((HERE.parent / "stale_by_split.json").read_text())
    splits = [s["split"] for s in stale["splits"]]
    channels = ["stale_frac_SST", "stale_frac_SSH_adt", "stale_frac_SSS"]
    chan_labels = ["SST", "SSH/ADT", "SSS"]

    fig, ax = plt.subplots(figsize=(8, 4.6))
    x = np.arange(len(splits))
    width = 0.25
    for k, (ch, lab) in enumerate(zip(channels, chan_labels)):
        vals = [s[ch] for s in stale["splits"]]
        ax.bar(x + (k - 1) * width, vals, width, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_ylim(0, 0.06)
    ax.axhline(stale["stale_gate_threshold"], color=FAIL_COLOR, ls="--", lw=1.3,
               label=f"gate threshold ({stale['stale_gate_threshold']:.0%})")
    ax.set_ylabel("stale fraction")
    ax.set_title(f"Cube-era stale-satellite fingerprint by split (gate: "
                 f"{'OPEN' if not stale['headline_metrics_embargoed'] else 'EMBARGOED'})", fontsize=11)
    ax.legend(fontsize=8)
    fig.text(0.5, -0.05,
              "'Old' (pre-cube) baseline is not present in any committed artifact under "
              "reports/stale_by_split.{md,json} — not shown (see PROVENANCE.json unsourced[]). "
              "All current values are 0.0%.",
              ha="center", fontsize=8.2, color="#333333", wrap=True)
    savefig(fig, "cube_stale_fingerprint", svg_only=True)


if __name__ == "__main__":
    fig_lineage_dag()
    fig_matrix_heatmap()
    fig_depthband_season()
    fig_osse_panel()
    fig_ruler_sparkline()
    fig_rcal_etable_depthband()
    fig_diag_control_headline()
    fig_rcal_schematic()
    fig_cube_schematic()
    fig_cube_extraction_inset()
    fig_cube_dataflow()
    fig_cube_stale_fingerprint()
    print("done")
