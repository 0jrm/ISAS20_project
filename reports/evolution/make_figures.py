#!/usr/bin/env python3
"""Regenerate reports/evolution/figs/*.{png,svg} from frozen artifact numbers only.

No model is run, retrained, or re-scored. Numbers are copied verbatim from the
source files listed in PROVENANCE.json. Run under conda env nespreso:

    srun --ntasks=1 --cpus-per-task=8 conda run -n nespreso \\
        python3 reports/evolution/make_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, Rectangle
import numpy as np

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figs"
FIGS.mkdir(exist_ok=True)

PASS_C = "#2a7f3f"
FAIL_C = "#b3261e"
SUPER_C = "#6b5b95"
ERRATUM_C = "#c47a00"
NEUTRAL = "#4a5568"

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 12,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def savefig(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGS / f"{name}.svg")
    fig.savefig(FIGS / f"{name}.png", dpi=160)
    plt.close(fig)
    print(f"wrote {name}")


# ---------------------------------------------------------------------------
# 1. Lineage DAG
# ---------------------------------------------------------------------------
def fig_lineage_dag() -> None:
    lineage = json.loads((HERE / "lineage.json").read_text())
    nodes = {n["id"]: n for n in lineage["nodes"]}
    edges = lineage["edges"]

    phases: list[str] = []
    for n in lineage["nodes"]:
        if n["phase"] not in phases:
            phases.append(n["phase"])
    phase_rank = {p: i for i, p in enumerate(phases)}

    row_counts: dict[int, int] = {}
    pos: dict[str, tuple[float, float]] = {}
    for n in lineage["nodes"]:
        row = phase_rank[n["phase"]]
        col = row_counts.get(row, 0)
        row_counts[row] = col + 1
        pos[n["id"]] = (float(col), float(row))
    for row, count in row_counts.items():
        offset = (count - 1) / 2.0
        for nid, (col, r) in list(pos.items()):
            if r == row:
                pos[nid] = (col - offset, row)

    verdict_color = {
        "survivor": PASS_C,
        "killed": FAIL_C,
        "superseded": SUPER_C,
        "erratum": ERRATUM_C,
    }
    edge_style = {
        "led_to": {"ls": "-", "color": "#555555", "lw": 1.2},
        "superseded_by": {"ls": "--", "color": SUPER_C, "lw": 1.4},
        "erratum_of": {"ls": ":", "color": ERRATUM_C, "lw": 1.6},
    }

    fig, ax = plt.subplots(figsize=(14, 10))
    for e in edges:
        x0, y0 = pos[e["from"]]
        x1, y1 = pos[e["to"]]
        st = edge_style[e["kind"]]
        ax.annotate(
            "",
            xy=(x1, y1 - 0.22),
            xytext=(x0, y0 + 0.22),
            arrowprops=dict(
                arrowstyle="-|>",
                color=st["color"],
                lw=st["lw"],
                linestyle=st["ls"],
                connectionstyle="arc3,rad=0.05",
            ),
        )

    for nid, (x, y) in pos.items():
        n = nodes[nid]
        c = verdict_color[n["verdict"]]
        label = nid.replace("phase", "p").replace("_", "\n")
        box = FancyBboxPatch(
            (x - 0.55, y - 0.22),
            1.1,
            0.44,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            facecolor=c,
            edgecolor="black",
            linewidth=0.8,
            alpha=0.9,
            mutation_aspect=0.5,
        )
        ax.add_patch(box)
        ax.text(x, y, label, ha="center", va="center", fontsize=6.5, color="white", fontweight="bold")

    for i, p in enumerate(phases):
        ax.text(-3.2, i, p, ha="right", va="center", fontsize=9, color=NEUTRAL, fontweight="bold")

    ax.set_xlim(-3.5, max(row_counts.values()) + 0.5)
    ax.set_ylim(len(phases) - 0.6, -0.6)
    ax.set_axis_off()
    ax.set_title("Experiment lineage (verdict color; edge kind by style)")

    handles = [
        mpatches.Patch(color=PASS_C, label="survivor"),
        mpatches.Patch(color=FAIL_C, label="killed"),
        mpatches.Patch(color=SUPER_C, label="superseded"),
        mpatches.Patch(color=ERRATUM_C, label="erratum"),
        plt.Line2D([0], [0], color="#555", lw=1.2, label="led_to"),
        plt.Line2D([0], [0], color=SUPER_C, lw=1.4, ls="--", label="superseded_by"),
        plt.Line2D([0], [0], color=ERRATUM_C, lw=1.6, ls=":", label="erratum_of"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.95)
    savefig(fig, "lineage_dag")


# ---------------------------------------------------------------------------
# 2. Matrix gate heatmap — 3x3 latent judgment + glyphs
# ---------------------------------------------------------------------------
def fig_matrix_gate_heatmap() -> None:
    # Verbatim from reports/phase5_{A,B,C}_{det,CRPS,NLL}.md (protocol v2)
    # det cells: T RMSE; prob cells: CRPS / ENCE
    cells = {
        ("A", "det"): {"num": "0.541", "pass": True, "kind": "det"},
        ("A", "CRPS"): {"num": "1.237\nENCE 0.053", "pass": True, "kind": "pc_crps", "winner": True},
        ("A", "NLL"): {"num": "1.257\nENCE 0.052", "pass": True, "kind": "pc_crps"},
        ("B", "det"): {"num": "0.534", "pass": True, "kind": "det"},
        ("B", "CRPS"): {"num": "2.761\nENCE 0.069", "pass": True, "kind": "pc_crps"},
        ("B", "NLL"): {"num": "2.754\nENCE 0.082", "pass": True, "kind": "pc_crps"},
        ("C", "det"): {"num": "0.609", "pass": False, "kind": "det"},
        ("C", "CRPS"): {"num": "0.742\nENCE 0.248", "pass": False, "kind": "pc_crps"},
        ("C", "NLL"): {"num": "0.774\nENCE 0.120", "pass": True, "kind": "pc_crps"},
    }
    reps = ["A", "B", "C"]
    heads = ["det", "CRPS", "NLL"]

    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    for i, r in enumerate(reps):
        for j, h in enumerate(heads):
            cell = cells[(r, h)]
            face = "#d9f0d3" if cell["pass"] else "#f5d0cc"
            rect = Rectangle((j, len(reps) - 1 - i), 1, 1, facecolor=face, edgecolor="#333", lw=1.2)
            ax.add_patch(rect)
            if cell["kind"] == "pc_crps":
                # hatch: PC-space CRPS not comparable across representations
                hatch = Rectangle(
                    (j, len(reps) - 1 - i),
                    1,
                    1,
                    facecolor="none",
                    edgecolor="#666666",
                    hatch="///",
                    lw=0,
                )
                ax.add_patch(hatch)
            glyph = "PASS" if cell["pass"] else "FAIL"
            gcolor = PASS_C if cell["pass"] else FAIL_C
            ax.text(
                j + 0.5,
                len(reps) - 1 - i + 0.62,
                cell["num"],
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color="#222",
            )
            ax.text(
                j + 0.5,
                len(reps) - 1 - i + 0.22,
                glyph,
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color=gcolor,
            )
            if cell.get("winner"):
                ring = Circle(
                    (j + 0.5, len(reps) - 1 - i + 0.5),
                    0.46,
                    fill=False,
                    edgecolor="#1a5276",
                    lw=3.0,
                )
                ax.add_patch(ring)

    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    ax.set_xticks([0.5, 1.5, 2.5])
    ax.set_xticklabels(["det (T RMSE)", "CRPS head", "NLL head"])
    ax.set_yticks([0.5, 1.5, 2.5])
    ax.set_yticklabels(["C dens+spice", "B joint EOF-32", "A sep PCA-16"])
    ax.set_title("Phase 5 matrix — latent judgment numbers + gate glyphs\n(ring = Section 3 winner A×CRPS)")
    ax.set_aspect("equal")
    ax.text(
        0.0,
        -0.35,
        "Footnote: hatched cells are PC-space CRPS — not comparable across representations A/B/C.\n"
        "Det gate: T ≤ 0.5903. Prob gate: ENCE < 0.20. Winner ring from physical-space §3 rule "
        "(ablation_summary.md), not latent CRPS rank.",
        transform=ax.transAxes,
        fontsize=8,
        va="top",
        color=NEUTRAL,
    )
    savefig(fig, "matrix_gate_heatmap")


# ---------------------------------------------------------------------------
# 3–4. Depth-band × season heatmaps (CRPS, ENCE)
# ---------------------------------------------------------------------------
def _strata_heatmaps() -> None:
    # Verbatim means from reports/phase5_A_CRPS_physical_strata.md
    bands = ["0-50", "50-200", "200-800", ">800"]
    seasons = ["DJF", "MAM", "JJA", "SON"]
    crps = np.array(
        [
            [0.2603, 0.2844, 0.3618, 0.2931],
            [0.3446, 0.3441, 0.3753, 0.3144],
            [0.1854, 0.1508, 0.1979, 0.1809],
            [0.0353, 0.0282, 0.0336, 0.0343],
        ]
    )
    ence = np.array(
        [
            [0.3692, 0.3732, 0.6622, 0.4093],
            [0.2521, 0.2103, 0.5560, 0.1826],
            [0.0629, 0.1628, 0.1413, 0.0585],
            [0.3358, 0.5456, 0.3164, 0.4742],
        ]
    )

    def _draw(data, title, name, mark_fail=False):
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        im = ax.imshow(data, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(4))
        ax.set_xticklabels(seasons)
        ax.set_yticks(range(4))
        ax.set_yticklabels(bands)
        ax.set_xlabel("season")
        ax.set_ylabel("depth band (m)")
        ax.set_title(title)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center", fontsize=9, color="#111")
                if mark_fail and data[i, j] >= 0.20:
                    fail = Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor=FAIL_C,
                        lw=3.0,
                    )
                    ax.add_patch(fail)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if mark_fail:
            ax.text(
                0.0,
                -0.18,
                "Red border: ENCE(T) ≥ 0.20 (METRICS_MANIFEST ence_max). "
                "Surface, thermocline, and >800 m lighting up is correct.",
                transform=ax.transAxes,
                fontsize=8,
                color=NEUTRAL,
            )
        savefig(fig, name)

    _draw(crps, "A×CRPS physical CRPS — depth band × season", "depthband_season_crps")
    _draw(
        ence,
        "A×CRPS physical ENCE(T) — depth band × season",
        "depthband_season_ence",
        mark_fail=True,
    )


# ---------------------------------------------------------------------------
# 5. OSSE panel
# ---------------------------------------------------------------------------
def fig_osse_panel() -> None:
    # Verbatim from reports/osse_results.md
    labels = ["E0", "E1", "E2", "E3", "E4", "E5"]
    overall = [1.5382, 1.5382, 0.5410, 0.5454, 0.6160, 1.4008]
    bands = {
        "0-100": [2.1986, 2.1986, 1.1934, 1.1510, 1.2119, 1.9488],
        "100-300": [3.1613, 3.1613, 1.0184, 1.0231, 1.1481, 2.8348],
        "300-700": [2.1013, 2.1013, 0.6068, 0.6343, 0.7678, 1.9547],
        ">700": [0.3772, 0.3772, 0.1523, 0.1673, 0.1887, 0.3546],
    }

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), gridspec_kw={"width_ratios": [1.0, 1.4]})

    ax = axes[0]
    colors = ["#888888", "#888888", "#4a90a4", "#2a7f3f", FAIL_C, "#555555"]
    bars = ax.bar(labels, overall, color=colors, edgecolor="black", lw=0.6)
    ax.set_ylabel("overall T RMSE")
    ax.set_title("OSSE cast-column overall (n=1101, 2021)")
    ax.axhline(0.5410, color="#4a90a4", ls=":", lw=1.0, label="E2 ISOP 0.5410")
    for b, v in zip(bars, overall):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, 1.85)
    ax.text(
        0.02,
        0.97,
        "E3>E2 FAIL\nE4≥E3 FAIL",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=11,
        fontweight="bold",
        color=FAIL_C,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fdecea", edgecolor=FAIL_C),
    )
    ax.text(
        0.98,
        0.55,
        "caveats:\n• cast-column proxy\n• E0≡E1\n• diagonal-R-only (v1)\n"
        "  v2 full-loc E4=0.6160\n  diag-control 0.546",
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=8,
        color=NEUTRAL,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f7f7f7", edgecolor="#ccc"),
    )

    ax2 = axes[1]
    x = np.arange(len(labels))
    width = 0.2
    for k, (band, vals) in enumerate(bands.items()):
        ax2.bar(x + (k - 1.5) * width, vals, width, label=band, edgecolor="black", lw=0.4)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("T RMSE")
    ax2.set_title("By depth band")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.set_ylim(0, 3.6)

    fig.suptitle(
        "Phase 6 OSSE — NeSPReSO ties ISOP (0.545 vs 0.541); calibrated R does not beat fixed R",
        fontsize=11,
    )
    savefig(fig, "osse_panel")


# ---------------------------------------------------------------------------
# 6. Ruler sparkline
# ---------------------------------------------------------------------------
def fig_ruler_sparkline() -> None:
    # Verbatim from reports/gate_floor_provenance.md
    xs = [0, 1, 2, 3]
    ys = [0.4158, 0.514, 0.5367, 0.5903]
    labels = [
        "published\nargo16 T\n(random)\n0.4158",
        "leaked chrono\neval\n0.514",
        "clean chrono\nargo16 raw\n0.5367",
        "corrected\nfloor ×1.10\n0.5903",
    ]
    colors = ["#888888", FAIL_C, PASS_C, "#1a5276"]

    fig, ax = plt.subplots(figsize=(9.5, 3.6))
    ax.plot(xs, ys, color="#333", lw=1.8, zorder=1)
    for x, y, c, lab in zip(xs, ys, colors, labels):
        ax.scatter([x], [y], s=90, color=c, zorder=2, edgecolor="black", lw=0.6)
        ax.annotate(
            lab,
            (x, y),
            textcoords="offset points",
            xytext=(0, 18 if x != 1 else -42),
            ha="center",
            fontsize=8,
            color=c,
        )
    ax.axhline(0.4574, color="#999", ls="--", lw=1.0)
    ax.text(3.05, 0.4574, "0.4574 published-random floor\n(do not use for chrono)", fontsize=7, color="#666", va="center")
    ax.set_xlim(-0.3, 3.6)
    ax.set_ylim(0.35, 0.68)
    ax.set_xticks([])
    ax.set_ylabel("T RMSE / floor")
    ax.set_title("Gate floor provenance — ruler repair (gate_floor_provenance.md)")
    savefig(fig, "ruler_sparkline")


def main() -> None:
    fig_lineage_dag()
    fig_matrix_gate_heatmap()
    _strata_heatmaps()
    fig_osse_panel()
    fig_ruler_sparkline()
    print("done")


if __name__ == "__main__":
    main()
