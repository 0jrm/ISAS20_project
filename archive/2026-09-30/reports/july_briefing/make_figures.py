#!/usr/bin/env python3
"""Schematic figures for the advisor/lab July briefing (no model scoring)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figs"
FIGS.mkdir(exist_ok=True)

plt.rcParams.update({"font.size": 10, "figure.facecolor": "white", "savefig.facecolor": "white"})


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGS / f"{name}.svg")
    fig.savefig(FIGS / f"{name}.png", dpi=160)
    plt.close(fig)
    print("wrote", name)


def _box(ax, x, y, w, h, text, fc="#eef2f5", ec="#333", fs=9):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.02",
            facecolor=fc, edgecolor=ec, lw=1.2,
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, wrap=True)


def fig_system_overview() -> None:
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.set_title("What NeSPReSO does in this project (Gulf of Mexico ARGO profiles)")
    _box(ax, 0.3, 1.2, 2.4, 1.6, "Surface inputs\nat the cast\n(SST, SSH/ADT, SSS,\nmonth / location)", fc="#d9e8f5")
    _box(ax, 3.2, 1.2, 2.4, 1.6, "Shared network\nPatchConvMLP\nd_model=128\n(same for A/B/C)", fc="#f5e6c8")
    _box(ax, 6.1, 1.2, 2.4, 1.6, "Latent scores\n(how A/B/C differ)\n+ optional σ", fc="#e8d9f5")
    _box(ax, 9.0, 1.2, 1.7, 1.6, "Decode to\nT(z), S(z)\nprofiles", fc="#d9f0d3")
    for x0, x1 in [(2.7, 3.2), (5.6, 6.1), (8.5, 9.0)]:
        ax.annotate("", xy=(x1, 2.0), xytext=(x0, 2.0),
                    arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.5))
    ax.text(5.5, 0.35,
            "Training target = real ARGO/CORA profiles (chronological split). L4 maps are inputs, not the hidden truth.",
            ha="center", fontsize=8, color="#444")
    save(fig, "system_overview")


def fig_abc_schematic() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 5.2))
    specs = [
        ("A — separate T & S", "#d9f0d3",
         ["Latent: PCA-16 on T\n+ PCA-16 on S\n(32 scores total)",
          "Decode: inverse PCA\n→ T(z), S(z) directly",
          "Stability: not built in\n(~22% σ₀ inversions\nafter truncation)",
          "Role: skill / UQ baseline"]),
        ("B — joint T/S EOF", "#d9e8f5",
         ["Latent: one EOF-32\non concatenated\nz-scored [T;S]",
          "Decode: inverse EOF\n→ T and S together",
          "Stability: still soft\n(same ~22% class)",
          "Role: “coupled T/S”\nalternative"]),
        ("C — density + spice", "#f5e6c8",
         ["Latent: PCA-16 on\nσ₀ residual + PCA-16\non spice τ",
          "Decode: σ₀,τ → (T,S)\nvia TEOS-10 Newton;\nisotonic on σ₀ at\ninference",
          "Stability: enforced\nat inference\n(not in the head)",
          "Role: physics-aware\nrepresentation"]),
    ]
    for ax, (title, color, lines) in zip(axes, specs):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")
        ax.set_title(title, fontsize=12, pad=8)
        y = 8.5
        for i, line in enumerate(lines):
            _box(ax, 0.5, y - 1.8, 9, 1.7, line, fc=color if i == 0 else "#fffcf7", fs=8.5)
            y -= 2.1
    fig.suptitle("How A, B, and C differ — same inputs & backbone; different profile representation", fontsize=12)
    save(fig, "abc_schematic")


def fig_heads_schematic() -> None:
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_title("Three prediction heads (what the network is trained to output)")
    _box(ax, 0.3, 1.0, 3.2, 3.2,
         "det (deterministic)\n\nPredicts mean only (μ)\nLoss: MSE in latent space\nNo uncertainty σ\n\nJudged by T RMSE\nvs floor 0.5903",
         fc="#eef2f5")
    _box(ax, 3.9, 1.0, 3.2, 3.2,
         "CRPS (probabilistic)\n\nPredicts μ and σ\nLoss: CRPS on scores\nTwo-stage: mean first,\nthen calibrate σ\n\nJudged by CRPS + ENCE\n(+ Spearman ranking)",
         fc="#d9f0d3")
    _box(ax, 7.5, 1.0, 3.2, 3.2,
         "NLL (probabilistic)\n\nPredicts μ and σ\nLoss: Gaussian NLL\nSame two-stage idea\n\nOften sharper or\nbetter-calibrated than\nCRPS depending on cell",
         fc="#e8d9f5")
    ax.text(5.5, 0.35,
            "Shared training protocol in the July matrix: chronological split; seeds {42,43,44};\n"
            "prob heads: stage-2 early-stop on validation ENCE; val-only σ rescaling before test.",
            ha="center", fontsize=8, color="#444")
    save(fig, "heads_schematic")


def fig_e_ladder() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 5.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_title("OSSE ladder E0–E5 — same analysis machinery; only the “casts” and their error change")

    rows = [
        ("E0", "No casts", "Background only\n(monthly clim)", "—", "#eeeeee"),
        ("E1", "Climatology casts", "Same clim as\nbackground", "R_fixed", "#eeeeee"),
        ("E2", "ISOP / MODAS-class", "Ridge on joint-EOF\nscores from SST/SSH", "R_fixed", "#d9e8f5"),
        ("E3", "NeSPReSO casts", "A×CRPS mean profile\nat real 2021 positions", "R_fixed", "#d9f0d3"),
        ("E4", "NeSPReSO + R_cal", "Same casts as E3", "R_cal from\npredicted Σ_T", "#f5e6c8"),
        ("E5", "E4 + QC", "Drop uncertain casts\n(σ̄ > val median)", "R_cal", "#f5d0cc"),
    ]
    headers = ["ID", "Cast source", "What that means", "Observation error R"]
    xs = [0.2, 1.3, 4.0, 8.5]
    ws = [1.0, 2.5, 4.2, 3.2]
    for x, w, h in zip(xs, ws, headers):
        _box(ax, x, 6.0, w, 0.7, h, fc="#1c1a16", ec="#1c1a16", fs=8)
        # white text overlay
        ax.patches[-1].set_facecolor("#333333")
        ax.texts[-1].set_color("white")
        ax.texts[-1].set_fontweight("bold")
    y = 5.1
    for r in rows:
        for x, w, txt, fc in zip(xs, ws, r[:4], [r[4]] * 4):
            _box(ax, x, y, w, 0.75, txt, fc=fc, fs=7.5)
        y -= 0.85
    ax.text(6, 0.25,
            "All E share: background = train-era monthly climatology; column OI in depth; L_v = 150 m.\n"
            "This run is a cast-column proxy (truth = ARGO at cast locations), not a full map-level ISAS analysis.",
            ha="center", fontsize=8, color="#444")
    save(fig, "e_ladder")


def fig_rcal_diag_vs_full() -> None:
    """Headline comparison from HANDOFF / osse_results — no new scoring."""
    labels = ["E3\nfixed R", "E4\ndiag(Σ)", "E4\nfull localized Σ"]
    vals = [0.5454, 0.546, 0.6160]
    colors = ["#2a7f3f", "#4a90a4", "#b3261e"]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bars = ax.bar(labels, vals, color=colors, edgecolor="black", lw=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("overall T RMSE (cast-column, 2021)")
    ax.set_ylim(0, 0.75)
    ax.set_title("Calibrated R: diagonal ties fixed R; full (off-diagonal) covariance hurts")
    ax.text(0.5, -0.18,
            "Sources: osse_results.md (E3=0.5454, E4 full=0.6160); HANDOFF.md (diag-control ≈0.546).\n"
            "Full Σ_T = V diag((α σ)²) Vᵀ, Schur-localized at L_loc=150 m.",
            transform=ax.transAxes, ha="center", fontsize=8, color="#444")
    save(fig, "rcal_diag_vs_full")


def _eq_figure(lines: list[str], name: str, title: str, figsize=(9.5, 2.2)) -> None:
    """Render plain-text equations as an SVG/PNG (no LaTeX / MathJax needed)."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(title, fontsize=11, loc="left", pad=8)
    y0 = 0.72 if len(lines) <= 2 else 0.82
    dy = 0.28 if len(lines) <= 2 else 0.22
    for i, line in enumerate(lines):
        ax.text(
            0.03,
            y0 - i * dy,
            line,
            transform=ax.transAxes,
            fontsize=13,
            family="DejaVu Sans",
            verticalalignment="center",
        )
    # light card background via patch
    ax.add_patch(
        Rectangle((0.01, 0.08), 0.98, 0.78, transform=ax.transAxes,
                  facecolor="#fffcf7", edgecolor="#d6d0c4", lw=1.0, zorder=0)
    )
    save(fig, name)


def fig_equations() -> None:
    _eq_figure(
        [
            "x_a  =  x_b  +  K ( y − H x_b )",
            "K    =  B Hᵀ ( H B Hᵀ + R )⁻¹",
        ],
        "eq_oi_update",
        "Optimal interpolation update (each cast column)",
        figsize=(9.5, 2.0),
    )
    _eq_figure(
        [
            "Σ_T  =  V  diag( (α σ)² )  Vᵀ",
            "R    =  ( Σ_T ∘ ρ ) + ε I",
            "ρ_ij =  exp( −½ ((z_i − z_j) / L_loc)² ) ,   L_loc = L_v = 150 m",
        ],
        "eq_rcal",
        "Calibrated observation-error R from the CRPS head (E4)",
        figsize=(10.5, 2.6),
    )


def fig_cube_dataflow() -> None:
    """Before/after surface-input plumbing (briefing §1.5)."""
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.axis("off")
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5)

    def box(x, y, w, h, text, color):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=color, edgecolor="black", linewidth=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.6)

    box(0.2, 2.9, 2.6, 1.6,
        "BEFORE\nPer-station HDF5 patches\nfixed spatial / temporal pads\n(utils/v2.json, v3.json)",
        "#f7dede")
    ax.annotate("", xy=(3.2, 3.7), xytext=(2.8, 3.7), arrowprops=dict(arrowstyle="->", lw=1.6))
    box(3.3, 2.9, 2.8, 1.6,
        "AFTER\nRegional Zarr cube\ngom_cube.zarr, daily\nGoM box, 2015–2022",
        "#dbeadb")
    ax.annotate("", xy=(6.5, 3.7), xytext=(6.1, 3.7), arrowprops=dict(arrowstyle="->", lw=1.6))
    box(6.6, 2.9, 3.2, 1.6,
        "On-demand sampling\nbilinear at (lat, lon, time)\n→ model inputs",
        "#dbe6f7")

    box(0.4, 0.4, 2.6, 1.1, "point model\n(July default: A×CRPS)", "#eef2ff")
    box(3.6, 0.4, 2.6, 1.1, "patch model", "#eef2ff")
    box(6.8, 0.4, 2.6, 1.1, "residual model", "#eef2ff")
    for x0 in (1.7, 4.9, 8.1):
        ax.annotate(
            "",
            xy=(x0, 1.5),
            xytext=(8.2, 2.9),
            arrowprops=dict(arrowstyle="->", lw=1.1, color="#666666", connectionstyle="arc3,rad=0.15"),
        )

    ax.text(
        5.5, 4.65,
        "Datacube = data-quality + unified extraction (and a path for v3 error channels) — "
        "not a claim that patches beat points.",
        ha="center", fontsize=9, color="#333333",
    )
    ax.text(
        5.5, -0.05,
        "July’s default model (A×CRPS) is a point model. Patch/residual did not win the Phase 5 matrix.",
        ha="center", fontsize=8.5, color="#b3261e",
    )
    ax.set_title("Surface inputs: before / after the regional datacube")
    save(fig, "cube_dataflow")


def main() -> None:
    fig_system_overview()
    fig_abc_schematic()
    fig_heads_schematic()
    fig_e_ladder()
    fig_rcal_diag_vs_full()
    fig_equations()
    fig_cube_dataflow()
    print("done")


if __name__ == "__main__":
    main()
