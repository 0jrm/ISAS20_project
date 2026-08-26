#!/usr/bin/env python3
"""Landscape explainer PDF for the three v2 models handed to nespreso_api.

Visual language follows repo-root explainer.png: nested color rails on
pseudocode, matching equations, a diagram. Numbers from code and
reports/heave_da_serve_spec.json. v1 (PCA, MLP, interpolated sats, Argo)
is treated as known.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np

HERE = Path(__file__).resolve().parent
OUT_PDF = HERE / "nespreso_v2_da_architecture.pdf"
FIGS = HERE / "figs"
FIGS.mkdir(exist_ok=True)

W, H = 16.0, 9.0
N_PAGES = 13

C = dict(
    ink="#1c1a16",
    mute="#5c5850",
    paper="#f7f4ee",
    card="#fffdf8",
    line="#d8d2c6",
    enc="#2f6f4e",
    sat="#c45c26",
    fuse="#1d7a74",
    mlp="#a33b66",
    mu="#3d5a80",
    sig="#6b3fa0",
    warp="#b56a1a",
    res="#d46a5a",
    warn="#9b2226",
    ok="#2f6f4e",
    ice="#e8f2ee",
    blush="#f8e8ee",
    sand="#f4ead8",
    mist="#e7eef5",
    lilac="#efe6f5",
)


def _rc():
    plt.rcParams.update(
        {
            "font.size": 10,
            "font.family": "DejaVu Sans",
            "text.color": C["ink"],
            "figure.facecolor": C["paper"],
            "savefig.facecolor": C["paper"],
            "pdf.fonttype": 42,
        }
    )


def _new():
    fig = plt.figure(figsize=(W, H), facecolor=C["paper"])
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), W, H, fc=C["paper"], ec="none"))
    return fig, ax


def _box(ax, x, y, w, h, fc=C["card"], ec=C["line"], lw=1.2, r=0.08, z=2):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0.012,rounding_size={r}",
            facecolor=fc,
            edgecolor=ec,
            lw=lw,
            zorder=z,
        )
    )


def _txt(ax, x, y, s, fs=10, c=C["ink"], w="normal", ha="left", va="center", family=None, rot=0):
    kw = dict(ha=ha, va=va, fontsize=fs, color=c, fontweight=w, zorder=4, rotation=rot)
    if family:
        kw["fontfamily"] = family
    ax.text(x, y, s, **kw)


def _title(ax, s, sub=None):
    _txt(ax, 0.4, H - 0.38, s, fs=17.5, w="bold")
    if sub:
        _txt(ax, 0.4, H - 0.72, sub, fs=9.8, c=C["mute"])


def _footer(ax, n):
    ax.add_patch(Rectangle((0, 0), W, 0.32, fc="#efebe3", ec="none", zorder=1))
    _txt(
        ax,
        0.35,
        0.16,
        "NeSPReSO v2  ·  handed to nespreso_api  ·  serve μ only; ingest R is Dai σ_o after H",
        fs=7.4,
        c=C["mute"],
    )
    _txt(ax, W - 0.4, 0.16, f"{n} / {N_PAGES}", fs=8.5, c=C["mute"], ha="right", w="bold")


def _arrow(ax, x0, y0, x1, y1, c=C["ink"], lw=1.4):
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>",
            mutation_scale=11,
            lw=lw,
            color=c,
            zorder=5,
        )
    )


def _code(ax, x, y, s, c=C["ink"], fs=8.0):
    _txt(ax, x, y, s, fs=fs, c=c, family="DejaVu Sans Mono", va="center")


def _rail(ax, x, y0, y1, color, label):
    ax.add_patch(Rectangle((x, y0), 0.09, y1 - y0, fc=color, ec="none", zorder=3, alpha=0.92))
    _txt(
        ax,
        x - 0.07,
        (y0 + y1) / 2,
        label,
        fs=7.0,
        c=color,
        w="bold",
        rot=90,
        ha="center",
        va="center",
    )


def _callout(ax, x, y, w, h, title, body, fc=C["sand"], ec=C["warp"]):
    _box(ax, x, y, w, h, fc=fc, ec=ec, lw=1.3)
    _txt(ax, x + 0.16, y + h - 0.24, title, fs=9.2, w="bold", c=ec)
    _txt(ax, x + 0.16, y + 0.16, body, fs=8.2, c=C["ink"], va="bottom")


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


def page_cover(ax):
    _title(
        ax,
        "Three networks go to the DA pipeline",
        "v1 (PCA, MLP, interpolated satellites, Argo) is assumed. This is the v2 architecture that the API actually loads.",
    )
    cells = [
        (
            0.4,
            C["ice"],
            C["enc"],
            "A_CRPS",
            "frozen ingest  ·  OSSE x_b",
            "PatchConvMLP, point mode\n9-d in  →  32-d PCA μ\n16 T PCs + 16 S PCs on native z\nCRPS head (serve μ only)\nseeds {42, 43, 44}",
        ),
        (
            5.55,
            C["sand"],
            C["warp"],
            "HeaveFast",
            "heave-family challenger",
            "HeaveResidualFast wraps\nPatchConvMLP (empty clone of HeaveResidual)\n11-d in  →  35-d μ\n3 warp + 16+16 residual PCs\nONI/RONI spliced at load",
        ),
        (
            10.7,
            C["lilac"],
            C["sig"],
            "ops",
            "LC-box second challenger",
            "Same HeaveResidualFast class\n30-d in  →  35-d μ\nn_enc=11 (harmonics+ENSO+local sat)\nn_sat=19 cube operators\nOwn residual PCA (ops cache)",
        ),
    ]
    for x, fc, ec, name, role, body in cells:
        _box(ax, x, 3.55, 4.85, 4.35, fc=fc, ec=ec, lw=2.0, r=0.12)
        _txt(ax, x + 0.28, 7.48, name, fs=20, w="bold", c=ec)
        _txt(ax, x + 0.28, 7.05, role, fs=10, c=C["mute"])
        _txt(ax, x + 0.28, 5.35, body, fs=10.4, va="center")

    _box(ax, 0.4, 0.52, 15.2, 2.78, fc=C["card"], ec=C["line"])
    _txt(ax, 0.6, 2.95, "The fork is output geometry and surface extras, not a new network class.", fs=12, w="bold")
    _txt(
        ax,
        0.6,
        1.7,
        "Every served cell is PatchConvMLP in point mode (patch_shape=None, residual=False).\n"
        "HeaveResidualFast is weight-compatible with HeaveResidual: same trunk, 35-d head, warp decode. Fast only batches searchsorted.\n"
        "Conv3, bathy, bathy_wind, Heave s42d, Latent, Direct were trained. They are not in the API enum.\n"
        "Legacy /v1_profile stays Ozavala TorchScript + 15-PC sklearn. Do not point NESPRESO_MODEL_PATH at these .pth files.",
        fs=10.1,
        va="center",
    )
    _footer(ax, 1)


def page_roster(ax):
    _title(ax, "Who is who, and what DA actually receives", "Roles from reports/heave_da_serve_spec.json. Pedigree beats a 0.01 °C native-T win.")
    headers = ["", "A_CRPS", "HeaveFast", "ops"]
    rows = [
        ("Role", "Frozen TSIS ingest + OSSE x_b", "Basin heave challenger", "LC-box only"),
        ("Class", "PatchConvMLP", "HeaveResidualFast", "HeaveResidualFast"),
        ("input_dim", "9", "11", "30"),
        ("n_enc / n_sat", "6 / 3", "8 / 3", "11 / 19"),
        ("output_dim μ", "32", "35", "35"),
        ("ENSO", "no", "ONI+RONI after 6 harmonics", "same splice, then 19 ops"),
        ("Decode", "cache PCA inverse, native z", "warp + residual PCA + unwarp", "same; PCA from ops cache"),
        ("Clim prior", "inside the PCA mean", "basin nanmean of cache T/S", "same fallback, ops cache"),
        ("Loss", "PCAHeteroLoss CRPS on 32 PCs", "heave_residual_fast CRPS", "same"),
        ("Stop", "stage-2 val ENCE (protocol v2)", "val_loss, patience 500", "val_loss, patience 500"),
        ("Seeds", "42, 43, 44", "42", "42"),
        ("Cache", "train_ready_3adcff404b0b.pkl", "same 9-d cache + splice", "train_ready_heave_ops.pkl"),
    ]
    xs = [0.35, 3.55, 8.05, 12.2]
    ws = [3.1, 4.4, 4.05, 3.4]
    y = 7.85
    fcs = [C["mist"], C["ice"], C["sand"], C["lilac"]]
    for x, w, h, fc in zip(xs, ws, headers, fcs):
        _box(ax, x, y, w, 0.42, fc=fc, ec=C["line"], r=0.04)
        _txt(ax, x + w / 2, y + 0.21, h, fs=9.5, w="bold", ha="center")
    y = 7.38
    for i, row in enumerate(rows):
        bg = "#ffffff" if i % 2 == 0 else "#f3efe6"
        for j, (x, w) in enumerate(zip(xs, ws)):
            _box(ax, x, y, w, 0.48, fc=bg, ec=C["line"], r=0.02, lw=0.6)
            _txt(
                ax,
                x + (0.12 if j == 0 else w / 2),
                y + 0.24,
                row[j],
                fs=7.5 if j else 8.1,
                w="bold" if j == 0 else "normal",
                ha="left" if j == 0 else "center",
            )
        y -= 0.50
    _callout(
        ax,
        0.35,
        0.48,
        15.3,
        0.88,
        "Analogy.",
        "A_CRPS is a 16-band equalizer on the raw profile. HeaveFast is landmark registration first (pin mixed layer and 26 °C),\n"
        "then a 16-band equalizer on the leftover wrinkles. ops is HeaveFast wearing 19 extra surface gauges. Same face, different makeup.",
        fc="#fff8e8",
        ec=C["warp"],
    )
    _footer(ax, 2)


def page_backbone(ax):
    _title(
        ax,
        "Shared backbone: PatchConvMLP in point mode",
        "All three cells. residual=False, patch_shape=None. Nested the same way the RIR slide nested groups inside groups.",
    )
    _box(ax, 0.3, 0.5, 6.55, 7.25, fc="#fbfaf6", ec=C["line"], lw=1.4)
    _txt(ax, 0.5, 7.5, "forward(x)    # B × input_dim", fs=10, w="bold", family="DejaVu Sans Mono")
    _rail(ax, 0.42, 6.78, 7.22, C["enc"], "encoder")
    _code(ax, 0.72, 7.00, "enc = x[:, : n_enc]", C["enc"])
    _rail(ax, 0.42, 6.38, 6.72, C["sat"], "satellite")
    _code(ax, 0.72, 6.55, "sat = x[:, n_enc :]", C["sat"])
    _code(ax, 0.72, 6.12, "h_e = Linear(n_enc, 128)(enc)", C["enc"])
    _code(ax, 0.72, 5.78, "h_s = Linear(n_sat, 128)(sat)", C["sat"])
    _rail(ax, 0.42, 5.28, 5.62, C["fuse"], "fusion skip")
    _code(ax, 0.72, 5.45, "h   = h_e + h_s", C["fuse"])
    _rail(ax, 0.42, 4.38, 5.18, C["mlp"], "MLP trunk")
    _code(ax, 0.72, 4.95, "h = Dropout(ReLU(Linear(128,1024)(h)), 0.2)", C["mlp"], 7.4)
    _code(ax, 0.72, 4.55, "h = Dropout(ReLU(Linear(1024,1024)(h)), 0.2)", C["mlp"], 7.4)
    _rail(ax, 0.42, 3.55, 4.22, C["mu"], "μ head")
    _code(ax, 0.72, 3.90, "μ = Linear(1024, d)(h)", C["mu"])
    _rail(ax, 0.42, 3.05, 3.48, C["sig"], "σ head")
    _code(ax, 0.72, 3.26, "σ = softplus(Linear(1024,d)(h)) + 0.001", C["sig"], 7.5)
    _code(ax, 0.72, 2.80, "return cat(μ, σ)           # B × 2d", C["ink"])
    _txt(ax, 0.55, 2.35, "Init that matters", fs=9.5, w="bold")
    _txt(
        ax,
        0.55,
        1.45,
        "σ Linear: weight = 0, bias = 0.5413\n"
        "softplus(0.5413) ≈ 1  →  σ starts near 1+σ_min\n"
        "Heave warp μ: first 3 rows of μ Linear zeroed\n"
        "  so raw=0 → MLD 50 m, D26 120 m at step 0",
        fs=8.0,
        va="center",
        family="DejaVu Sans Mono",
    )

    _box(ax, 7.05, 4.55, 4.35, 3.2, fc=C["mist"], ec=C["mu"], lw=1.4)
    _txt(ax, 7.22, 7.45, "Fusion (not concat)", fs=11, w="bold", c=C["fuse"])
    _txt(ax, 9.2, 6.85, r"$h_0 = W_e x_{\mathrm{enc}} + b_e + W_s x_{\mathrm{sat}} + b_s$", fs=10.5, ha="center")
    _txt(ax, 9.2, 6.25, r"$h_0 \in \mathbb{R}^{128}$", fs=10, ha="center", c=C["mute"])
    _txt(ax, 7.22, 5.75, "Dual head", fs=11, w="bold", c=C["sig"])
    _txt(ax, 9.2, 5.25, r"$\sigma = \mathrm{softplus}(W_\sigma h) + \sigma_{\min}$", fs=11, ha="center")
    _txt(ax, 9.2, 4.80, r"$\sigma_{\min}=10^{-3}$", fs=9.5, ha="center", c=C["mute"])

    _callout(
        ax,
        7.05,
        2.55,
        4.35,
        1.80,
        "Analogy.",
        "Two translators into the same 128-d dialect,\nthen mixed on one bus (add, not concat).\nThe MLP is a 1024-wide desk. μ and σ are\ntwo faders on that desk, not two nets.",
        fc="#e8f6f3",
        ec=C["fuse"],
    )

    _box(ax, 11.6, 0.5, 4.05, 7.25, fc=C["card"], ec=C["line"])
    _txt(ax, 13.62, 7.5, "data flow", fs=11, w="bold", ha="center")
    blocks = [
        (6.85, 0.70, "x  (B, D_in)", C["mist"], C["ink"]),
        (6.05, 0.70, "split  n_enc | n_sat", "#fff", C["ink"]),
        (5.15, 1.15, "Linear → 128\n+  Linear → 128", C["ice"], C["enc"]),
        (4.15, 0.70, "add   h = h_e + h_s", C["ice"], C["fuse"]),
        (3.15, 0.95, "Linear 128→1024\nReLU, Drop 0.2", C["blush"], C["mlp"]),
        (2.15, 0.95, "Linear 1024→1024\nReLU, Drop 0.2", C["blush"], C["mlp"]),
        (1.15, 0.85, "μ  and  σ  heads", C["lilac"], C["sig"]),
        (0.45, 0.55, "cat [μ | σ]", C["sand"], C["warp"]),
    ]
    bx, bw = 11.85, 3.55
    for y, hh, t, fc, ec in blocks:
        _box(ax, bx, y, bw, hh, fc=fc, ec=ec, lw=1.3, r=0.06)
        _txt(ax, bx + bw / 2, y + hh / 2, t, fs=8.2, ha="center")
    for i in range(len(blocks) - 1):
        y0 = blocks[i][0]
        y1 = blocks[i + 1][0] + blocks[i + 1][1]
        _arrow(ax, bx + bw / 2, y0, bx + bw / 2, y1 + 0.02, c=C["mute"], lw=1.1)
    _txt(
        ax,
        13.62,
        0.22,
        "No ResidualLinearBlock. No Conv2d.\nServed cells are point MLPs.",
        fs=7.5,
        c=C["mute"],
        ha="center",
        va="bottom",
    )
    _footer(ax, 3)


def page_inputs(ax):
    _title(
        ax,
        "Input encoding: harmonics, then a splice, then satellites",
        "v1 already used cyclic time/lat/lon + SSS/SST/SSH. v2 only adds where those columns sit, and two optional extras.",
    )
    _box(ax, 0.35, 5.35, 7.4, 2.55, fc=C["mist"], ec=C["mu"], lw=1.4)
    _txt(ax, 0.55, 7.6, "Six harmonics  (n_enc base = 6)", fs=12, w="bold", c=C["mu"])
    for i, e in enumerate(
        [
            r"$(\cos,\,\sin)\,2\pi\,\mathrm{doy}/365$",
            r"$(\cos,\,\sin)\,2\pi\,\mathrm{lat}/180$",
            r"$(\cos,\,\sin)\,2\pi\,\mathrm{lon}/360$",
        ]
    ):
        _txt(ax, 4.05, 7.05 - i * 0.48, e, fs=13, ha="center")
    _txt(ax, 0.55, 5.58, "doy from the cache builder (juld % 365). MATLAB datenum is fine if doy matches.", fs=8, c=C["mute"])

    _box(ax, 8.0, 5.35, 7.65, 2.55, fc=C["sand"], ec=C["warp"], lw=1.4)
    _txt(ax, 8.2, 7.6, "ENSO splice  (HeaveFast, ops only)", fs=12, w="bold", c=C["warp"])
    _txt(
        ax,
        8.2,
        6.45,
        "inject_enso_columns inserts ONI, RONI after the 6 harmonics,\nbefore the satellite block. Cache on disk stays 9-d (HeaveFast)\nor 28-d (ops). The extra 2 appear at load.\n\nA_CRPS never splices. Its n_enc stays 6.",
        fs=10,
        va="center",
    )

    _txt(ax, 0.4, 4.95, "Column layout the network actually sees", fs=12, w="bold")

    def layout_row(y, name, chunks, note):
        _txt(ax, 0.4, y + 0.42, name, fs=10, w="bold")
        x = 2.55
        for lab, w, fc, ec in chunks:
            _box(ax, x, y, w, 0.55, fc=fc, ec=ec, r=0.04, lw=1.1)
            _txt(ax, x + w / 2, y + 0.28, lab, fs=7.4, ha="center", w="bold")
            x += w + 0.06
        _txt(ax, 15.55, y + 0.28, note, fs=8, c=C["mute"], ha="right")

    layout_row(
        4.15,
        "A_CRPS    9",
        [("6 harmonics", 3.6, C["mist"], C["mu"]), ("SSS SST SSH", 2.8, C["sand"], C["sat"])],
        "n_enc=6   n_sat=3",
    )
    layout_row(
        3.25,
        "HeaveFast 11",
        [
            ("6 harmonics", 3.0, C["mist"], C["mu"]),
            ("ONI RONI", 1.8, C["ice"], C["enc"]),
            ("SSS SST SSH", 2.8, C["sand"], C["sat"]),
        ],
        "n_enc=8   n_sat=3",
    )
    layout_row(
        2.35,
        "ops      30",
        [
            ("6 harm.", 1.7, C["mist"], C["mu"]),
            ("ONI RONI", 1.5, C["ice"], C["enc"]),
            ("SSS SST SSH", 2.1, C["sand"], C["sat"]),
            ("19 cube operators", 5.6, C["lilac"], C["sig"]),
        ],
        "n_enc=11  n_sat=19",
    )
    _box(ax, 0.35, 0.5, 15.3, 1.55, fc=C["card"], ec=C["line"])
    _txt(ax, 0.55, 1.72, "The ops split is the one people mis-wire.", fs=11, w="bold", c=C["warn"])
    _txt(
        ax,
        0.55,
        1.05,
        "ops puts local SSS/SST/SSH in the encoder half (n_enc=11), and the 19 operators in the satellite half.\n"
        "HeaveFast puts local SSS/SST/SSH in the satellite half (n_sat=3). Same three numbers, different bus.\n"
        "Fusion is still add in 128-d. The Linear maps change width; the diagram on the previous page does not.",
        fs=9.5,
        va="center",
    )
    _footer(ax, 4)


def page_heads(ax):
    _title(
        ax,
        "Dual head: a mean you serve, a spread you do not",
        "probabilistic=True, n_quantiles=0. Forward always returns cat(μ, σ). The API keeps [:, :d].",
    )
    _box(ax, 0.35, 4.7, 7.5, 3.2, fc=C["card"], ec=C["mlp"], lw=1.5)
    _txt(ax, 4.1, 7.6, "shared trunk  h ∈ R^{1024}", fs=12, w="bold", ha="center", c=C["mlp"])
    _box(ax, 0.6, 5.55, 3.3, 1.55, fc=C["mist"], ec=C["mu"], lw=1.6)
    _txt(ax, 2.25, 6.7, "μ Linear(1024, d)", fs=11, w="bold", ha="center", c=C["mu"])
    _txt(ax, 2.25, 6.15, "unconstrained", fs=9, ha="center", c=C["mute"])
    _box(ax, 4.3, 5.55, 3.3, 1.55, fc=C["lilac"], ec=C["sig"], lw=1.6)
    _txt(ax, 5.95, 6.7, "σ Linear(1024, d)", fs=11, w="bold", ha="center", c=C["sig"])
    _txt(ax, 5.95, 6.15, "softplus + 0.001", fs=9, ha="center", c=C["mute"])
    _txt(ax, 4.1, 5.1, "output width = 2d.   A_CRPS: 64.   Heave: 70.", fs=9.5, ha="center", c=C["mute"])

    _box(ax, 8.1, 4.7, 7.5, 3.2, fc=C["card"], ec=C["ok"], lw=1.5)
    _txt(ax, 8.3, 7.6, "What the API is allowed to write", fs=12, w="bold", c=C["ok"])
    _txt(
        ax,
        8.3,
        6.15,
        "1. Take μ = out[:, :d]. Never write σ into NetCDF err/R.\n"
        "2. Ingest R is Dai σ_o after TSIS’s H, floors 0.05 °C / 0.02 psu.\n"
        "3. CRPS-head σ is a training object. Physical ENCE(T)=0.236\n"
        "   (A_CRPS) and ENCE(σ_D26)=0.52 (HeaveFast) both miss 0.20.\n"
        "4. A full localized Σ_T already lost the column OSSE\n"
        "   (E4 0.616 vs diag 0.546). Do not rebuild it from this head.",
        fs=10,
        va="center",
    )

    _box(ax, 0.35, 0.5, 7.5, 3.95, fc=C["ice"], ec=C["enc"], lw=1.4)
    _txt(ax, 0.55, 4.15, "A_CRPS  —  two-stage protocol v2", fs=12, w="bold", c=C["enc"])
    _txt(
        ax,
        0.55,
        2.45,
        "Stage 1. Freeze σ. MSE on μ. Learn the 32 PC means.\n"
        "Stage 2. Unfreeze σ. μ learning rate × 0.1. Switch to CRPS.\n"
        "         Early-stop on val ENCE, never on val loss.\n"
        "         (Loss plateaus before calibration matures.)\n\n"
        "Checkpoints: p5_A_CRPS_v2_s{42,43,44}_s2 / model_best.pth",
        fs=10,
        va="center",
    )
    _box(ax, 8.1, 0.5, 7.5, 3.95, fc=C["sand"], ec=C["warp"], lw=1.4)
    _txt(ax, 8.3, 4.15, "HeaveFast / ops  —  joint CRPS from epoch 1", fs=12, w="bold", c=C["warp"])
    _txt(
        ax,
        8.3,
        2.45,
        "No freeze_sigma. μ and σ train together under heave_residual_fast.\n"
        "Monitor min val_loss, early_stop=500, Adam 1e-3, batch 512.\n"
        "HeaveFast ran 2091 epochs (patience actually used).\n"
        "ops ran 714. Same seed 42.\n\n"
        "Do not treat Fast vs s42d as an architecture bake-off.\n"
        "Fast is the same science, batched searchsorted, ~2.5× cheaper.",
        fs=10,
        va="center",
    )
    _footer(ax, 5)


def page_acrps(ax):
    _title(
        ax,
        "A_CRPS: 32 knobs on native z",
        "Flavor A from the July matrix. Separate T and S PCA-16. No warp. Frozen Phase 5/6 cell.",
    )
    _box(ax, 0.3, 0.5, 6.4, 7.25, fc="#fbfaf6", ec=C["enc"], lw=1.5)
    _txt(ax, 0.5, 7.48, "decode  (gold: _decode_pcs_to_ts)", fs=10.5, w="bold")
    _rail(ax, 0.42, 5.85, 7.15, C["mu"], "split")
    _code(ax, 0.7, 6.95, "α_T = μ[:, :16]", C["mu"])
    _code(ax, 0.7, 6.50, "α_S = μ[:, 16:32]", C["mu"])
    _code(ax, 0.7, 6.05, "# ignore μ[:, 32:]  (= σ)", C["mute"])
    _rail(ax, 0.42, 3.85, 5.65, C["ok"], "PCA inverse")
    _code(ax, 0.7, 5.35, "T = α_T @ V_T + T̄", C["enc"])
    _code(ax, 0.7, 4.85, "S = α_S @ V_S + S̄", C["enc"])
    _code(ax, 0.7, 4.30, "V_* = pca.components_    # 16 × 1801", C["mute"], 7.5)
    _code(ax, 0.7, 3.95, "T̄  = pca.mean_          # 1801", C["mute"], 7.5)
    _txt(ax, 0.55, 3.45, "Loss (the config name lies)", fs=10.5, w="bold", c=C["warn"])
    _txt(
        ax,
        0.55,
        2.15,
        "loss_config.mode = \"combined\" but prob_mode is set,\n"
        "so make_loss returns PCAHeteroLoss, not CombinedPCALoss.\n"
        "Training CRPS is on the 32 PC scores. There is no\n"
        "profile-MSE term while the CRPS head is on.\n"
        "Stage 1 (frozen σ) is MSE on those same 32 scores.",
        fs=8.4,
        va="center",
    )
    _code(ax, 0.7, 1.05, "L = mean CRPS(μ_k, σ_k, α_k^true)", C["sig"])
    _txt(ax, 0.55, 0.68, "k = 1…32, independent. Shared V does not couple the loss.", fs=7.6, c=C["mute"])

    _box(ax, 6.9, 4.85, 8.75, 2.9, fc=C["mist"], ec=C["mu"], lw=1.4)
    _txt(ax, 7.1, 7.45, "Reconstruction", fs=12, w="bold", c=C["mu"])
    _txt(ax, 11.25, 6.85, r"$T(z)=\bar T(z)+\sum_{k=1}^{16}\alpha_k\,v_k^{(T)}(z)$", fs=14, ha="center")
    _txt(ax, 11.25, 6.15, r"$S(z)=\bar S(z)+\sum_{k=1}^{16}\alpha_k^{(S)}\,v_k^{(S)}(z)$", fs=14, ha="center")
    _txt(
        ax,
        7.1,
        5.35,
        "Native 1 m grid, z = 0…1800 m, n_z = 1801. Pair the checkpoint with cache 3adcff404b0b.\n"
        "Never mix this PCA with models/pca_stats.pkl (that is v1, 15 PCs).",
        fs=9,
        va="center",
    )
    _callout(
        ax,
        6.9,
        2.85,
        8.75,
        1.80,
        "Analogy.",
        "A graphic equalizer with 16 sliders for temperature shape and 16 for salinity.\n"
        "The mean profile is the flat response. The network does not invent depths; it only turns knobs.\n"
        "Truncation to 16 is why a perfect network still cannot beat T1 (truth-through-PCA ≈ 0.12 °C in 50–200 m).",
        fc="#e7eef5",
        ec=C["mu"],
    )
    _box(ax, 6.9, 0.5, 8.75, 2.15, fc=C["card"], ec=C["line"])
    _txt(ax, 7.1, 2.32, "Why this cell is still the first ingest", fs=11, w="bold")
    _txt(
        ax,
        7.1,
        1.35,
        "Level N² inversion 0.0029 vs HeaveFast 0.008. Frozen E3 pedigree (CRPS 0.545 vs ISOP 0.541).\n"
        "z-PCA-16 is a better T1 of shape than warp-of-climatology (D26 8.3 m vs 61 m on truth).\n"
        "SLA already owns large heave. The leftover DA problem is shape, which is this head’s native output.",
        fs=9.2,
        va="center",
    )
    _footer(ax, 6)


def page_crps(ax):
    _title(
        ax,
        "CRPS: the scoring rule the head was trained with",
        "Closed form for a Gaussian. Same formula in evalphys.calibration.gaussian_crps and the torch mirror.",
    )
    _box(ax, 0.35, 5.15, 9.3, 2.75, fc=C["mist"], ec=C["sig"], lw=1.5)
    _txt(ax, 0.55, 7.6, "Gaussian CRPS", fs=13, w="bold", c=C["sig"])
    _txt(
        ax,
        5.0,
        6.85,
        r"$\mathrm{CRPS}(\mathcal{N}(\mu,\sigma^2), y)=\sigma[z(2\Phi(z)-1)+2\phi(z)-1/\sqrt{\pi}]$",
        fs=13.5,
        ha="center",
    )
    _txt(ax, 5.0, 6.15, r"$z=(y-\mu)/\sigma,\quad \sigma\leftarrow\max(\sigma,\sigma_{\min})$", fs=12, ha="center")
    _txt(
        ax,
        0.55,
        5.50,
        "Mean over batch and output index. No covariance. Each PC (or MLD, D26) is a separate 1-d exam.",
        fs=9.5,
        c=C["mute"],
    )

    _box(ax, 9.9, 5.15, 5.75, 2.75, fc=C["card"], ec=C["line"])
    xx = np.linspace(-3.2, 3.2, 200)
    pdf = np.exp(-0.5 * xx**2) / np.sqrt(2 * np.pi)
    px0, px1, py0, py1 = 10.2, 15.3, 5.4, 7.55
    ax.add_patch(Rectangle((px0, py0), px1 - px0, py1 - py0, fc="#fbfaf6", ec=C["line"], lw=0.8, zorder=2))
    xs = px0 + (xx - xx.min()) / (xx.max() - xx.min()) * (px1 - px0 - 0.3) + 0.1
    ys = py0 + 0.25 + pdf / pdf.max() * (py1 - py0 - 0.7)
    ax.plot(xs, ys, color=C["sig"], lw=2.0, zorder=4)
    zt = 0.9
    xt = px0 + (zt - xx.min()) / (xx.max() - xx.min()) * (px1 - px0 - 0.3) + 0.1
    ax.plot([xt, xt], [py0 + 0.2, py1 - 0.15], color=C["warn"], lw=1.6, ls="--", zorder=4)
    _txt(ax, xt + 0.15, py1 - 0.28, "y", fs=9, c=C["warn"])
    _txt(ax, (px0 + px1) / 2, 7.7, "predict a whole distribution, not a point", fs=8.5, ha="center", c=C["mute"])

    _box(ax, 0.35, 0.5, 5.0, 4.4, fc=C["ice"], ec=C["enc"], lw=1.4)
    _txt(ax, 0.55, 4.6, "What CRPS wants", fs=12, w="bold", c=C["enc"])
    _txt(
        ax,
        0.55,
        2.55,
        "Close μ, and a σ that matches typical |y−μ|.\n"
        "Too-narrow σ is punished (overconfident).\n"
        "Too-wide σ is punished (useless forecast).\n\n"
        "ENCE asks a different question: are the\n"
        "stated σ the right size on average?\n"
        "Gate in this project: ENCE < 0.20.\n\n"
        "A_CRPS: ENCE in PC space ≈ 0.05 (pass),\n"
        "physical ENCE(T) ≈ 0.236 (fail).\n"
        "The loss never saw °C. It saw PC scores.",
        fs=9.8,
        va="center",
    )
    _box(ax, 5.55, 0.5, 5.0, 4.4, fc=C["sand"], ec=C["warp"], lw=1.4)
    _txt(ax, 5.75, 4.6, "NLL vs CRPS vs ENCE", fs=12, w="bold", c=C["warp"])
    _txt(
        ax,
        5.75,
        2.55,
        "NLL (Seitzer β-NLL, β=0.5) is the other\n"
        "probabilistic cell in the July matrix.\n"
        "A×NLL was close; A×CRPS won CRPS+ENCE.\n\n"
        "Quantile heads (n_quantiles=9) exist in\n"
        "the class. Served cells do not use them.\n\n"
        "Spearman(σ, |error|) is a ranking check,\n"
        "not a size check. Do not use it as R.",
        fs=9.8,
        va="center",
    )
    _box(ax, 10.75, 0.5, 4.9, 4.4, fc=C["blush"], ec=C["warn"], lw=1.4)
    _txt(ax, 10.95, 4.6, "Off-diagonals are fake", fs=12, w="bold", c=C["warn"])
    _txt(
        ax,
        10.95,
        2.55,
        "If you form Σ_T = V diag((ασ)²) Vᵀ\n"
        "you get cross-level correlations from\n"
        "the shared basis V, not from a learned\n"
        "joint covariance.\n\n"
        "That object is OI-stable after Schur\n"
        "localization — and still worse than diag.\n"
        "The head was trained marginally.\n"
        "Treat σ as 32 separate widths, then\n"
        "throw them away at ingest.",
        fs=9.8,
        va="center",
    )
    _footer(ax, 7)


def page_heave_warp(ax):
    _title(
        ax,
        "Heave: pin two landmarks, then PCA the wrinkles",
        "This is the RIR analogue. Outer residual = vertical registration. Inner residual = shape on a canonical z-grid.",
    )
    _box(ax, 0.3, 0.5, 6.55, 7.25, fc="#fbfaf6", ec=C["warp"], lw=1.5)
    _txt(ax, 0.5, 7.48, "HeaveResidualFast.forward + decode", fs=10, w="bold")
    _rail(ax, 0.42, 6.55, 7.22, C["mlp"], "trunk")
    _code(ax, 0.7, 6.95, "y = PatchConvMLP(x)           # 70-d", C["mlp"])
    _code(ax, 0.7, 6.62, "μ, σ = y[:, :35], y[:, 35:]", C["mlp"])
    _rail(ax, 0.42, 4.55, 6.40, C["warp"], "outer: warp")
    _code(ax, 0.7, 6.15, "raw = μ[:, :3]", C["warp"])
    _code(ax, 0.7, 5.75, "mld = 50 · exp(raw0)", C["warp"])
    _code(ax, 0.7, 5.35, "d26 = mld + 5 + 65 · exp(raw1)", C["warp"])
    _code(ax, 0.7, 4.95, "stretch = 1 + 0.3 tanh(raw2)   # dead", C["mute"], 7.5)
    _code(ax, 0.7, 4.62, "clamp layers ≥ 5 m, ≤ z_bot−5", C["mute"], 7.5)
    _rail(ax, 0.42, 3.05, 4.40, C["res"], "inner: residual PCs")
    _code(ax, 0.7, 4.10, "T_res = μ[:, 3:19]  @ V_T + m_T", C["res"], 7.6)
    _code(ax, 0.7, 3.70, "S_res = μ[:, 19:35] @ V_S + m_S", C["res"], 7.6)
    _code(ax, 0.7, 3.28, "# V_* from TRAIN warped residuals", C["mute"], 7.3)
    _rail(ax, 0.42, 1.55, 2.95, C["fuse"], "short skip")
    _code(ax, 0.7, 2.65, "T_c = warp(T_clim; mld, d26) + T_res", C["fuse"], 7.3)
    _code(ax, 0.7, 2.20, "S_c = warp(S_clim; mld, d26) + S_res", C["fuse"], 7.3)
    _code(ax, 0.7, 1.75, "# clim = nanmean(cache T/S)", C["mute"], 7.3)
    _rail(ax, 0.42, 0.62, 1.42, C["warp"], "long unwarp")
    _code(ax, 0.7, 1.15, "T, S = unwarp(T_c, S_c; mld, d26)", C["warp"], 7.5)

    _box(ax, 7.05, 3.55, 8.6, 4.2, fc=C["card"], ec=C["warp"], lw=1.4)
    _txt(ax, 11.35, 7.5, "piecewise-linear landmark map", fs=11, w="bold", ha="center", c=C["warp"])

    def knots(x0, y0, labels, title, color):
        _txt(ax, x0 + 1.6, y0 + 3.35, title, fs=9, w="bold", ha="center", c=color)
        zs = [0.0, 0.28, 0.52, 1.0]
        for z, lab in zip(zs, labels):
            yy = y0 + 0.15 + (1 - z) * 3.0
            ax.plot([x0 + 0.35, x0 + 2.85], [yy, yy], color=color, lw=1.0, zorder=3)
            _txt(ax, x0 + 3.05, yy, lab, fs=8, c=color, va="center")
        ax.plot([x0 + 1.6, x0 + 1.6], [y0 + 0.15, y0 + 3.15], color=color, lw=2.2, zorder=3)
        ax.scatter(
            [x0 + 1.6] * 4,
            [y0 + 0.15 + (1 - z) * 3.0 for z in zs],
            s=28,
            color=color,
            zorder=4,
        )

    knots(7.25, 3.7, ["0 m", "MLD (pred)", "D26 (pred)", "1800 m"], "physical z", C["sat"])
    knots(12.05, 3.7, ["0 m", "50 m", "120 m", "1800 m"], "canonical z", C["enc"])
    for yp, yc in [(6.35, 6.35), (5.55, 5.85), (4.85, 5.15), (3.85, 3.85)]:
        _arrow(ax, 10.3, yp, 12.15, yc, c=C["warp"], lw=1.2)

    _callout(
        ax,
        7.05,
        0.5,
        8.6,
        2.85,
        "Analogy. Face alignment, then expression PCA.",
        "Warp is the landmark step: pin the mixed-layer base and the 26 °C isotherm to a template (50 m, 120 m).\n"
        "Everything between those pins is stretched or compressed linearly. Deep water below D26 is stretched to 1800 m.\n"
        "Residual PCA then models what is left on that aligned grid — the wrinkles, not the pose.\n"
        "Unwarp puts the edited template back onto the predicted physical landmarks. Stretch (raw2) is predicted and ignored.",
        fc="#fff3e0",
        ec=C["warp"],
    )
    _footer(ax, 8)


def page_heave_decode(ax):
    _title(
        ax,
        "Heave decode: basin-mean prior, train-only residual PCA",
        "physical_ts in HeaveResidualFastLoss. Gold path: thermocline_scorecard._decode_heave_ts. Match tol T/S 1e-5.",
    )
    _box(ax, 0.35, 4.85, 7.6, 2.95, fc=C["mist"], ec=C["mu"], lw=1.4)
    _txt(ax, 0.55, 7.5, "Warp decode (exp, not tanh)", fs=12, w="bold", c=C["mu"])
    _txt(ax, 4.15, 6.85, r"$\mathrm{MLD}=50\,\exp(r_0),\quad r_0=\mu_0$", fs=13, ha="center")
    _txt(ax, 4.15, 6.25, r"$\mathrm{D26}=\mathrm{MLD}+5+65\,\exp(r_1)$", fs=13, ha="center")
    _txt(
        ax,
        0.55,
        5.45,
        "tanh floor used to vanish at 10 m. Exp keeps gradient at the\n"
        "canonical point (raw=0 → 50 m / 120 m). σ in metres: ∂(a e^x)/∂x = a e^x.",
        fs=9.2,
        va="center",
    )
    _box(ax, 8.15, 4.85, 7.5, 2.95, fc=C["lilac"], ec=C["sig"], lw=1.4)
    _txt(ax, 8.35, 7.5, "σ of landmarks (Jacobian, clamp ignored)", fs=12, w="bold", c=C["sig"])
    _txt(ax, 11.9, 6.85, r"$\sigma_{\mathrm{MLD}}=\sigma_0\cdot\mathrm{MLD}$", fs=13, ha="center")
    _txt(ax, 11.9, 6.25, r"$\sigma_{\mathrm{D26}}=\sqrt{\sigma_{\mathrm{MLD}}^2+(\sigma_1\cdot\mathrm{gap})^2}$", fs=13, ha="center")
    _txt(
        ax,
        8.35,
        5.45,
        "gap = 65 exp(r1). σ_stretch is computed by the Linear and then\n"
        "dropped: CRPS state is [MLD, D26, 32 PCs], not the 3 raw warp logits.",
        fs=9.2,
        va="center",
    )
    steps = [
        (0.35, C["sand"], C["warp"], "1. Prior", "T_clim, S_clim =\nnanmean of cache\ntrue_profiles\nbroadcast to every row.\nNot a mapped WOA.\nNot nearest Argo."),
        (5.55, C["ice"], C["enc"], "2. Warp prior", "Sample clim at the\nphysical z that each\ncanonical node maps to\n(searchsorted lerp).\nAdd T_res, S_res on\ncanonical nodes."),
        (10.75, C["blush"], C["res"], "3. Unwarp", "Sample (prior+res) at\nthe canonical z of each\nphysical node.\nResult lives on native\n1 m z, 0–1800 m."),
    ]
    for x, fc, ec, t, b in steps:
        _box(ax, x, 1.85, 4.9, 2.75, fc=fc, ec=ec, lw=1.5, r=0.1)
        _txt(ax, x + 0.22, 4.28, t, fs=13, w="bold", c=ec)
        _txt(ax, x + 0.22, 3.05, b, fs=9.3, va="center")
    _arrow(ax, 5.25, 3.2, 5.55, 3.2, c=C["ink"], lw=1.6)
    _arrow(ax, 10.45, 3.2, 10.75, 3.2, c=C["ink"], lw=1.6)
    _box(ax, 0.35, 0.48, 15.3, 1.18, fc=C["card"], ec=C["warn"], lw=1.3)
    _txt(
        ax,
        0.55,
        1.07,
        "Residual PCA is not the cache['pca_models'] native-z basis. It is fitted inside the loss on train-split warped residuals\n"
        "(chronological 70/15/15). Extract t_components / t_mean / s_components / s_mean from a loss built on that cache. Ops uses its own.",
        fs=9.5,
        va="center",
    )
    _footer(ax, 9)


def page_heave_loss(ax):
    _title(
        ax,
        "Heave loss: geometry + CRPS + a weak thermocline slope",
        "z-PCA cache targets are ignored (del target). Three terms, three scales.",
    )
    _box(ax, 0.35, 5.55, 15.3, 2.35, fc=C["mist"], ec=C["mu"], lw=1.4)
    _txt(ax, 8.0, 7.55, r"$L = 10\cdot L_{\mathrm{geom}} + 1\cdot L_{\mathrm{CRPS}} + 0.1\cdot L_{\partial_z T}$", fs=16, ha="center")
    _txt(
        ax,
        8.0,
        6.55,
        r"$L_{\mathrm{geom}}=\mathrm{mean}[(\mathrm{MLD}-\mathrm{MLD}^{\mathrm{true}})^2/50^2+(\mathrm{D26}-\mathrm{D26}^{\mathrm{true}})^2/120^2]$"
        "      "
        r"$L_{\partial_z T}=\mathrm{mean}\,(\partial_z\hat T-\partial_z T)^2$  on 50–200 m",
        fs=11,
        ha="center",
    )
    _txt(
        ax,
        8.0,
        5.85,
        "CRPS state = [MLD (m), D26 (m), 16 T residual PCs, 16 S residual PCs]. 34 scalars, not 35. Stretch is absent.",
        fs=9.5,
        ha="center",
        c=C["mute"],
    )
    cols = [
        (
            0.35,
            C["sand"],
            C["warp"],
            "Geometry  ×10",
            "Puts MLD/D26 in metres, scaled by the\n"
            "canonical 50 / 120 so the two landmarks\n"
            "share a dimensionless MSE.\n\n"
            "True MLD/D26 from evalphys on Argo T/S;\n"
            "NaN → 50 m / 120 m. That is also the\n"
            "network’s step-0 output. Honest start.",
        ),
        (
            5.55,
            C["lilac"],
            C["sig"],
            "CRPS  ×1",
            "Same Gaussian CRPS as A_CRPS, now on\n"
            "a mixed state: two depths + 32 scores.\n\n"
            "Landmark σ is Jacobian-transformed so\n"
            "the units are metres. PC σ stays in\n"
            "score units. One mean() over all 34.",
        ),
        (
            10.75,
            C["blush"],
            C["res"],
            "dT/dz  ×0.1",
            "After the full warp decode, compare\n"
            "vertical T gradient to Argo in 50–200 m.\n\n"
            "This is the only physical-space term.\n"
            "It is weak on purpose. It does not fix\n"
            "level N² (still ~3× A_CRPS).",
        ),
    ]
    for x, fc, ec, t, b in cols:
        _box(ax, x, 1.85, 4.9, 3.45, fc=fc, ec=ec, lw=1.5)
        _txt(ax, x + 0.2, 4.98, t, fs=13, w="bold", c=ec)
        _txt(ax, x + 0.2, 3.4, b, fs=9.4, va="center")
    _box(ax, 0.35, 0.48, 15.3, 1.18, fc="#fff4f0", ec=C["warn"], lw=1.3)
    _txt(
        ax,
        0.55,
        1.07,
        "Dead channel. stretch is in the 35-d μ, has a σ, and is initialized at 1.0. It is not in L_geom, not in the CRPS state,\n"
        "and not in unwarp. Fast vs s42d share this. Do not invent a story for it at ingest. Serve T(z), S(z) from physical_ts.",
        fs=9.5,
        va="center",
    )
    _footer(ax, 10)


def page_ops(ax):
    _title(
        ax,
        "ops: 19 cube operators on the satellite bus",
        "Same HeaveResidualFast. Different n_enc/n_sat split. Operators from gom_cube.zarr at request time, not from an Argo-row cache.",
    )
    names = [
        ("sst.grad_x@local", "∂SST/∂x, native grid"),
        ("sst.grad_y@local", "∂SST/∂y, native grid"),
        ("sst.grad_x@1.0deg", "same after 1° Gaussian"),
        ("sst.grad_y@1.0deg", ""),
        ("sss.grad_x@local", "∂SSS/∂x"),
        ("sss.grad_y@local", "∂SSS/∂y"),
        ("sss.grad_x@1.0deg", "1° SSS gradient"),
        ("sss.grad_y@1.0deg", ""),
        ("ssh.grad_x@local", "∂η/∂x"),
        ("ssh.grad_y@local", "∂η/∂y"),
        ("ssh.grad_x@1.0deg", "1° SSH gradient"),
        ("ssh.grad_y@1.0deg", ""),
        ("ssh.laplacian@1.0deg", "∇²η at 1°"),
        ("sst.tendency@7d", "LSQ slope, 7-day stack"),
        ("ssh.tendency@7d", "LSQ slope, 7-day stack"),
        ("ssh.geo_u@local", "−(g/f) ∂η/∂y"),
        ("ssh.geo_v@local", "+(g/f) ∂η/∂x"),
        ("ssh.geo_u@1.0deg", "geostrophy at 1°"),
        ("ssh.geo_v@1.0deg", ""),
    ]
    _box(ax, 0.3, 0.5, 7.3, 7.25, fc=C["card"], ec=C["sig"], lw=1.4)
    _txt(ax, 0.5, 7.48, "OP_NAMES  (order is the sat block)", fs=11, w="bold", c=C["sig"])
    y = 7.05
    for i, (n, note) in enumerate(names):
        _txt(ax, 0.55, y, f"{i+1:2d}  {n}", fs=7.7, family="DejaVu Sans Mono", va="center")
        if note:
            _txt(ax, 5.15, y, note, fs=7.3, c=C["mute"], va="center")
        y -= 0.32

    _box(ax, 7.8, 5.15, 7.85, 2.6, fc=C["mist"], ec=C["mu"], lw=1.4)
    _txt(ax, 8.0, 7.45, "Geostrophy and tendency", fs=12, w="bold", c=C["mu"])
    _txt(ax, 11.7, 6.85, r"$u = -(g/f)\,\partial_y\eta,\quad v=(g/f)\,\partial_x\eta$", fs=13, ha="center")
    _txt(ax, 11.7, 6.25, r"$f=2\Omega\sin\phi,\quad |f|<10^{-8}\ \to\ 10^{-8}$", fs=11, ha="center")
    _txt(
        ax,
        8.0,
        5.55,
        "Tendency: LSQ slope per pixel on a 7-day stack. Need ≥ 3 finite days or NaN.",
        fs=8.8,
        c=C["mute"],
    )
    _box(ax, 7.8, 2.55, 7.85, 2.4, fc=C["sand"], ec=C["warp"], lw=1.4)
    _txt(ax, 8.0, 4.65, "How they are sampled", fs=12, w="bold", c=C["warp"])
    _txt(
        ax,
        8.0,
        3.55,
        "preproc.features.operators on gom_cube.zarr, then bilinear\n"
        "sample at (lat, lon, date). ops_zscored=false — feed raw.\n"
        "Do not use train_ready_2ab55b15b14f.pkl (Argo-row aligned).\n"
        "Missing cube → HTTP 503. Do not zero-fill.",
        fs=9.5,
        va="center",
    )
    _box(ax, 7.8, 0.5, 7.85, 1.85, fc=C["blush"], ec=C["warn"], lw=1.4)
    _txt(ax, 8.0, 2.05, "Why it exists", fs=12, w="bold", c=C["warn"])
    _txt(
        ax,
        8.0,
        1.2,
        "Basin thermocline σ_T after H: HeaveFast 1.20, ops 1.21.\n"
        "LC box (24–28°N, 88–84°W): ops 1.024 vs HeaveFast 1.051.\n"
        "If TSIS takes one err profile, ship HeaveFast. Two: split LC.",
        fs=9.2,
        va="center",
    )
    _footer(ax, 11)


def page_serve(ax):
    _title(
        ax,
        "Serve contract: the API is a decoder, not a trainer",
        "Pins from reports/heave_da_serve_spec.json. Load that file. Do not retrain.",
    )
    rules = [
        ("μ only", "If forward width is 2d, take [:, :d]. CRPS σ is not R."),
        ("Cache pairing", "Never mix a checkpoint with a different cache. PCA lives in the pickle."),
        ("R = Dai after H", "sigma_o_hycom.csv, floors 0.05 °C / 0.02 psu. Not 1 m RMSE, not dense Σ."),
        ("H is TSIS’s", "Do not apply HYCOM layer_sample in the API. TSIS applies H on cycle thknss."),
        ("torch.load", "map_location='cpu', weights_only=False. Arch from ckpt['config']['arch']."),
        ("Enum", "A_CRPS | HeaveFast | ops. Optional ?seed=42|43|44 on A_CRPS, default 42."),
        ("Legacy SAT", "/v1_profile and /v1_profile/grid stay Ozavala. Different PCA, depth, architecture."),
        ("Do not serve", "conv3, bathy, bathy_wind, heave_s42d, Latent, Direct."),
    ]
    y = 7.35
    for i, (t, b) in enumerate(rules):
        x = 0.35 if i % 2 == 0 else 8.15
        if i % 2 == 0 and i:
            y -= 1.32
        _box(ax, x, y - 1.12, 7.5, 1.18, fc=C["card"], ec=C["line"], lw=1.1)
        _txt(ax, x + 0.18, y - 0.22, t, fs=11, w="bold", c=C["enc"] if i < 6 else C["warn"])
        _txt(ax, x + 0.18, y - 0.68, b, fs=9.0)
    _box(ax, 0.35, 0.48, 15.3, 1.22, fc=C["ice"], ec=C["enc"], lw=1.3)
    _txt(
        ax,
        0.55,
        1.09,
        "Suggested slices. (1) A_CRPS 9-d sat path, cache PCA inverse. (2) HeaveFast ONI/RONI + warp decode + basin-mean clim.\n"
        "(3) ops cube operators at request time. Client model= kwarg; default URL stays Ozavala SAT. Depth grid is cache['PRES'], 1801 × 1 m.",
        fs=9.6,
        va="center",
    )
    _footer(ax, 12)


def page_compare(ax):
    _title(ax, "One picture of the two decode geometries", "Same trunk. Different thing being added to a prior.")
    z = np.linspace(0, 1, 80)

    def _prof(x0, y0, w, h, curve, color, label):
        xs = x0 + 0.2 + np.asarray(curve) * (w - 0.3)
        ys = y0 + h - 0.15 - z * (h - 0.3)
        ax.plot(xs, ys, color=color, lw=2.0, zorder=4)
        _txt(ax, x0 + w / 2, y0 - 0.22, label, fs=8, ha="center", c=color)

    mean = 0.35 + 0.45 * np.exp(-((z - 0.22) / 0.12) ** 2)
    mode = 0.12 * np.sin(2 * np.pi * z * 2) * np.exp(-z * 1.5)
    rec = mean + 0.7 * mode

    _box(ax, 0.35, 0.5, 7.5, 7.25, fc=C["ice"], ec=C["enc"], lw=2.0)
    _txt(ax, 4.1, 7.45, "A_CRPS  —  additive in z", fs=16, w="bold", ha="center", c=C["enc"])
    _prof(0.7, 4.35, 1.7, 2.55, mean, C["mute"], "PCA mean")
    _txt(ax, 2.7, 5.6, "+", fs=22, w="bold", ha="center")
    _prof(3.05, 4.35, 1.7, 2.55, 0.5 + mode, C["mlp"], "16 T modes")
    _txt(ax, 5.05, 5.6, "=", fs=22, w="bold", ha="center")
    _prof(5.4, 4.35, 1.7, 2.55, rec, C["enc"], "T(z)")
    _txt(
        ax,
        0.6,
        2.35,
        "Prior is the PCA mean profile (one vector per variable).\n"
        "Coefficients α live in a Euclidean 32-d box the MLP knows.\n"
        "No vertical rearranging. A 3 °C jump stays at its z.\n\n"
        "That is why T1 of shape is good, and why SLA still owns heave:\n"
        "this head never moves the thermocline as a degree of freedom.",
        fs=10.2,
        va="center",
    )

    mean2 = 0.4 + 0.4 * np.exp(-((z - 0.35) / 0.1) ** 2)
    wrk = 0.4 + 0.4 * np.exp(-((z - 0.22) / 0.1) ** 2)
    rec2 = wrk + 0.08 * np.sin(6 * np.pi * z) * np.exp(-z)
    _box(ax, 8.15, 0.5, 7.5, 7.25, fc=C["sand"], ec=C["warp"], lw=2.0)
    _txt(ax, 11.9, 7.45, "Heave  —  additive on a warped z", fs=16, w="bold", ha="center", c=C["warp"])
    _prof(8.5, 4.35, 1.55, 2.55, mean2, C["mute"], "basin clim")
    _txt(ax, 10.25, 5.6, "↝", fs=20, ha="center", c=C["warp"])
    _prof(10.55, 4.35, 1.55, 2.55, wrk, C["warp"], "warped")
    _txt(ax, 12.3, 5.6, "+", fs=22, w="bold", ha="center")
    _prof(12.55, 4.35, 1.7, 2.55, rec2, C["res"], "unwarped")
    _txt(
        ax,
        8.4,
        2.35,
        "Prior is a single Gulf-wide nanmean, then moved so that\n"
        "its MLD/D26 sit on the predicted landmarks.\n"
        "Residual PCs add shape on that aligned grid.\n\n"
        "T1 of warp-clim is poor (D26 61 m on truth). The leftover\n"
        "is still shape. HeaveFast wins native T by 0.012 °C and\n"
        "loses level N² by 3×. First ingest stays A_CRPS.",
        fs=10.2,
        va="center",
    )
    _footer(ax, 13)


PAGES = [
    page_cover,
    page_roster,
    page_backbone,
    page_inputs,
    page_heads,
    page_acrps,
    page_crps,
    page_heave_warp,
    page_heave_decode,
    page_heave_loss,
    page_ops,
    page_serve,
    page_compare,
]


def main() -> None:
    assert len(PAGES) == N_PAGES
    _rc()
    with PdfPages(OUT_PDF) as pdf:
        d = pdf.infodict()
        d["Title"] = "NeSPReSO v2 architecture for the DA pipeline"
        d["Author"] = "ISAS20_project / reports/da_arch_explainer"
        for i, fn in enumerate(PAGES, 1):
            fig, ax = _new()
            fn(ax)
            pdf.savefig(fig)
            fig.savefig(FIGS / f"p{i:02d}_{fn.__name__.removeprefix('page_')}.png", dpi=130)
            plt.close(fig)
            print("page", i, fn.__name__)
    print("wrote", OUT_PDF)


if __name__ == "__main__":
    main()
