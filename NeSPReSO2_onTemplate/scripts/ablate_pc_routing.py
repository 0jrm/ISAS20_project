#!/usr/bin/env python3
"""PC-routing lever: train-only ridge (Cell0), spec JSON, TSV scoreboard."""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_REPO = _ROOT.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from base.split_utils import build_split_indices
from model.loss import sklearn_inverse_transform_pcs
from model.pc_routing import (
    EASY_INPUT_NAMES,
    EASY_PC_IDX,
    N_PC,
    apply_ridge,
    aux_names,
    cols_for,
    column_names,
    fit_ridge,
    max_abs_r,
    sigma_floor_from_r,
    weights_from_r,
)

CACHE = _REPO / "data/cache/train_ready_heave_ops_pca32.pkl"
SPEC_PATH = _REPO / "reports/pc_routing_spec.json"
TSV_PATH = _REPO / "reports/eval_pc_routing_ablation.tsv"
MD_PATH = _REPO / "reports/eval_pc_routing_ablation.md"
CORR_JSON = _REPO / "reports/pc_input_correlation.json"

# Short id → table title. Keep ids stable for append/tmux; never show the id as the name.
CELL_LABELS = {
    "ops": "A_CRPS_z32 + 19 cube operators (frozen sat-skill leader)",
    "roni": "A_CRPS_z32 + ONI/RONI, no operators (frozen)",
    "stoch_eof": "Stochastic EOF, 9-d SSS/SST/SSH (frozen ENCE bar)",
    "pair_argo": "Pair MLP: satellites + earlier Argo PCs (frozen; needs a neighbor)",
    "persist": "Persistence: copy earlier Argo profile, no network (frozen)",
    "clim": "Climatology: decode all 64 PCs as 0",
    "cell0_gem": "Linear GEM: ridge SSH/SST/SSS/season/lon → T PC1–4 and S PC1–2; other PCs climatology",
    "cell0_ridge64": "Linear ridge: same 6 inputs → all 64 T/S PCs",
    "cell4_trunc16": "Linear ridge on 64 PCs, then zero T/S PC 17–32",
    "pc_skip_smoke": "Skip-easy MLP, 2+2 epoch smoke (not a bake-off)",
    "pc_skip": "Skip-easy MLP: frozen linear GEM on T PC1–4 / S PC1–2, network residual on the rest",
    "pc_aux": "Aux stem: 19 operators added only onto hard PCs (not T1–2 / S1–2)",
    "pc_wk": "PC CRPS reweighted by unexplained variance (1−r²)",
    "pc_sigma": "σ floor √(1−r²) on PCs the surface does not linearly explain",
    "pc_steric": "PC1 steric pin: T_PC1 and S_PC1 tied to SSH",
    "pc_heave_warp": "Heave warp predicted from SST/SSH only (physical T/S)",
}

FROZEN = (
    ("ops", 0.53545, 0.09077, 0.67058, float("nan")),
    ("roni", 0.53819, 0.09057, 0.46134, float("nan")),
    ("stoch_eof", 0.54503, 0.08929, 0.151, float("nan")),
    ("pair_argo", 0.5309, 0.0846, 0.279, float("nan")),
    ("persist", 0.68539, 0.10455, float("nan"), float("nan")),
)

GROUPS = (
    ("Frozen comparators (not retrained)", ("ops", "roni", "stoch_eof", "pair_argo", "persist")),
    ("Linear probes (no neural net)", ("clim", "cell0_gem", "cell0_ridge64", "cell4_trunc16")),
    ("Trained cells", ("pc_skip", "pc_aux", "pc_wk", "pc_sigma", "pc_steric", "pc_heave_warp")),
    ("Diagnostics, not bake-off", ("pc_skip_smoke",)),
)

TSV_FIELDS = ("cell", "label", "T", "S", "ence_T", "T_50_200", "n", "note", "source")

DL = {
    "split_mode": "chronological",
    "train_frac": 0.7,
    "val_frac": 0.15,
    "test_frac": 0.15,
}


def _load_cache(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def _rmse(diff: np.ndarray) -> float:
    return float(np.sqrt(np.nanmean(diff ** 2)))


def _band_rmse(pred_nd: np.ndarray, true_nd: np.ndarray, z: np.ndarray, lo: float, hi: float) -> float:
    m = (z >= lo) & (z < hi)
    if not np.any(m):
        return float("nan")
    return _rmse(pred_nd[m, :] - true_nd[m, :])


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 20:
        return float("nan")
    aa, bb = a[m], b[m]
    sa, sb = aa.std(), bb.std()
    if sa < 1e-12 or sb < 1e-12:
        return float("nan")
    return float(np.dot(aa - aa.mean(), bb - bb.mean()) / (m.sum() * sa * sb))


def _write_tsv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TSV_FIELDS, delimiter="\t")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in TSV_FIELDS})


def _fmt(x) -> str:
    if x is None or x == "":
        return "—"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(v):
        return "—"
    return f"{v:.3f}"


def _label(cell: str) -> str:
    return CELL_LABELS.get(cell, cell)


def _annotate(row: dict) -> dict:
    out = dict(row)
    cell = str(out.get("cell", ""))
    out["label"] = _label(cell)
    if not out.get("note") or out.get("note") == cell:
        out["note"] = _label(cell)
    if cell == "ops":
        try:
            v = float(out.get("T_50_200") or "nan")
        except (TypeError, ValueError):
            v = float("nan")
        if np.isfinite(v) and 0.4 < v < 0.6:
            out["T_50_200"] = ""
    return out


def _write_md(rows: list[dict], spec: dict | None, path: Path) -> None:
    by_id = {str(r.get("cell")): _annotate(r) for r in rows}
    lines = [
        "# PC-routing ablation",
        "",
        "Chronological Argo test, n=623. Short ids (`pc_skip`, `ops`) are for scripts.",
        "The names below are what the row actually is.",
        "",
        "Frozen sat-only top 3: A_CRPS_z32+ops, A_CRPS_z32+RONI, stochastic EOF.",
        "Pair MLP and persistence are reference rows. They are not mixed into sat-only training.",
        "",
        "Win vs A_CRPS_z32+ops: T RMSE ≤ 0.525, or raw ENCE(T) ≤ 0.20 with T no worse than 0.545,",
        "or a clear 50–200 m T drop.",
        "",
        "T and S are physical RMSE (°C, psu). ENCE(T) is raw test calibration.",
        "50–200 m is T RMSE for linear probes and ENCE(T) in that band for trained cells.",
    ]
    seen = set()
    for title, ids in GROUPS:
        chunk = [by_id[i] for i in ids if i in by_id]
        if not chunk:
            continue
        lines += [
            "",
            f"## {title}",
            "",
            "| model | T RMSE | S RMSE | ENCE(T) | 50–200 m | n |",
            "|-------|-------:|-------:|--------:|---------:|--:|",
        ]
        for row in chunk:
            seen.add(row["cell"])
            lines.append(
                f"| {row['label']} | {_fmt(row['T'])} | {_fmt(row['S'])} | {_fmt(row['ence_T'])} | "
                f"{_fmt(row['T_50_200'])} | {row.get('n', '')} |"
            )
    leftover = [by_id[k] for k in by_id if k not in seen]
    if leftover:
        lines += [
            "",
            "## Other",
            "",
            "| model | T RMSE | S RMSE | ENCE(T) | 50–200 m | n |",
            "|-------|-------:|-------:|--------:|---------:|--:|",
        ]
        for row in leftover:
            lines.append(
                f"| {row['label']} | {_fmt(row['T'])} | {_fmt(row['S'])} | {_fmt(row['ence_T'])} | "
                f"{_fmt(row['T_50_200'])} | {row.get('n', '')} |"
            )
    if spec:
        r_k = np.asarray(spec["r_k"], dtype=np.float64)
        lines += [
            "",
            "## Train-only linear structure",
            "",
            f"SSH vs T_PC1 r = {spec.get('train_ssh_tpc1_r', float('nan')):.3f} (train).",
            f"Easy PCs {spec.get('easy_pc_idx')} from {spec.get('easy_input_names')}.",
            f"max |r| T_PC1={r_k[0]:.3f} T_PC5={r_k[4]:.3f} T_PC16={r_k[15]:.3f} T_PC32={r_k[31]:.3f}.",
            "",
            "## Kill gates",
            "",
            spec.get("kill_notes", "See TSV."),
        ]
    path.write_text("\n".join(lines) + "\n")


def cmd_fit(args: argparse.Namespace) -> int:
    cache = _load_cache(Path(args.cache))
    X = np.asarray(cache["inputs"], dtype=np.float64)
    Y = np.asarray(cache["targets"], dtype=np.float64)
    n, d = X.shape
    if Y.shape[1] != N_PC:
        raise ValueError(f"expected 64 PCs, got {Y.shape}")
    splits = build_split_indices(
        n, np.asarray(cache["JULD"]), DL, dataset_tag="argo_v2"
    )
    tr, te = np.asarray(splits["train"]), np.asarray(splits["test"])
    easy_cols = cols_for(d, EASY_INPUT_NAMES)
    easy_pc = list(EASY_PC_IDX)
    W, b = fit_ridge(X[tr][:, easy_cols], Y[tr][:, easy_pc], alpha=float(args.alpha))
    r_k = max_abs_r(X[tr], Y[tr])
    w_k = weights_from_r(r_k)
    ssh = X[tr][:, cols_for(d, ("ssh",))[0]]
    tpc1 = Y[tr][:, 0]
    spc1 = Y[tr][:, 32]
    a_t, b_t = np.polyfit(np.nan_to_num(ssh), np.nan_to_num(tpc1), 1)
    a_s, b_s = np.polyfit(np.nan_to_num(ssh), np.nan_to_num(spc1), 1)

    def decode_pcs(pcs: np.ndarray) -> dict:
        return sklearn_inverse_transform_pcs(
            pcs.astype(np.float32),
            cache["pca_models"],
            cache["outputs"],
        )

    def score(pcs: np.ndarray, idx: np.ndarray) -> dict[str, float]:
        pred = decode_pcs(pcs)
        z = np.asarray(cache["PRES"], dtype=np.float64).reshape(-1)
        out = {}
        for name in ("temperature", "salinity"):
            truth = np.asarray(cache["profiles"][name], dtype=np.float64)
            diff = pred[name] - truth[:, idx]
            out[name] = _rmse(diff)
            if name == "temperature":
                out["T_50_200"] = _band_rmse(pred[name], truth[:, idx], z, 50.0, 200.0)
        pc_err = pcs - Y[idx]
        out["per_pc_rmse"] = np.sqrt(np.nanmean(pc_err ** 2, axis=0)).tolist()
        return out

    gem = np.zeros((len(te), N_PC), dtype=np.float64)
    gem[:, easy_pc] = apply_ridge(X[te][:, easy_cols], W, b)
    s0 = score(gem, te)

    ridge_all_W, ridge_all_b = fit_ridge(X[tr][:, easy_cols], Y[tr], alpha=float(args.alpha))
    full = apply_ridge(X[te][:, easy_cols], ridge_all_W, ridge_all_b)
    s_full = score(full, te)
    trunc = full.copy()
    trunc[:, 16:32] = 0.0
    trunc[:, 48:] = 0.0
    s_trunc = score(trunc, te)

    clim = np.zeros((len(te), N_PC), dtype=np.float64)
    s_clim = score(clim, te)

    kill = []
    if abs(s0["temperature"] - 0.53545) <= 0.02:
        kill.append("Cell0 T is within 0.02 of ops. MLP is not earning the easy PCs.")
    else:
        kill.append(
            f"Cell0 T={s0['temperature']:.3f} vs ops 0.535. Skip still worth training. "
            "Do not put aux on PC1."
        )
    d_trunc = abs(s_trunc["temperature"] - s_full["temperature"])
    if d_trunc < 0.01:
        kill.append(
            f"Trunc16 ΔT={d_trunc:.4f} vs full-ridge. High PCs add almost nothing linearly. "
            "Do not train a new 16-PC sat net. Pair keeps 32."
        )
    else:
        kill.append(f"Trunc16 ΔT={d_trunc:.4f}. High PCs move linear T. Pair cell stays relevant.")

    spec = {
        "easy_input_names": list(EASY_INPUT_NAMES),
        "easy_pc_idx": easy_pc,
        "aux_input_names": aux_names(d),
        "input_dim_cache": d,
        "column_names_cache": column_names(d),
        "alpha": float(args.alpha),
        "W": W.tolist(),
        "b": b.tolist(),
        "r_k": r_k.tolist(),
        "w_k": w_k.tolist(),
        "sigma_floor_k": sigma_floor_from_r(r_k).tolist(),
        "pc1_ssh": {"t": [float(a_t), float(b_t)], "s": [float(a_s), float(b_s)]},
        "train_ssh_tpc1_r": _pearson(ssh, tpc1),
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "kill_notes": " ".join(kill),
        "cell0": {
            "T": s0["temperature"],
            "S": s0["salinity"],
            "T_50_200": s0["T_50_200"],
            "per_pc_rmse": s0["per_pc_rmse"],
        },
        "cell0_ridge64": {"T": s_full["temperature"], "S": s_full["salinity"], "T_50_200": s_full["T_50_200"]},
        "cell4_trunc16": {"T": s_trunc["temperature"], "S": s_trunc["salinity"], "T_50_200": s_trunc["T_50_200"]},
        "clim": {"T": s_clim["temperature"], "S": s_clim["salinity"], "T_50_200": s_clim["T_50_200"]},
    }
    out = Path(args.spec)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(spec, indent=2) + "\n")
    print(f"saved {out} Cell0 T={s0['temperature']:.4f} S={s0['salinity']:.4f} ssh-T1 r={spec['train_ssh_tpc1_r']:.3f}")

    rows = []
    for cell, T, S, ence, band in FROZEN:
        rows.append(
            _annotate(
                {
                    "cell": cell,
                    "T": T,
                    "S": S,
                    "ence_T": ence,
                    "T_50_200": band,
                    "n": 623,
                    "note": _label(cell),
                    "source": "frozen",
                }
            )
        )
    rows.append(
        _annotate(
            {
                "cell": "cell0_gem",
                "T": s0["temperature"],
                "S": s0["salinity"],
                "ence_T": "",
                "T_50_200": s0["T_50_200"],
                "n": len(te),
                "source": str(out),
            }
        )
    )
    rows.append(
        _annotate(
            {
                "cell": "cell0_ridge64",
                "T": s_full["temperature"],
                "S": s_full["salinity"],
                "ence_T": "",
                "T_50_200": s_full["T_50_200"],
                "n": len(te),
                "source": str(out),
            }
        )
    )
    rows.append(
        _annotate(
            {
                "cell": "cell4_trunc16",
                "T": s_trunc["temperature"],
                "S": s_trunc["salinity"],
                "ence_T": "",
                "T_50_200": s_trunc["T_50_200"],
                "n": len(te),
                "source": str(out),
            }
        )
    )
    rows.append(
        _annotate(
            {
                "cell": "clim",
                "T": s_clim["temperature"],
                "S": s_clim["salinity"],
                "ence_T": "",
                "T_50_200": s_clim["T_50_200"],
                "n": len(te),
                "source": str(out),
            }
        )
    )
    tsv = Path(args.tsv)
    _write_tsv(rows, tsv)
    _write_md(rows, spec, Path(args.md))
    print(f"saved {tsv} {args.md}")
    return 0


def cmd_append(args: argparse.Namespace) -> int:
    tsv = Path(args.tsv)
    rows = []
    if tsv.is_file():
        with tsv.open() as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
    payload = json.loads(Path(args.eval_json).read_text())
    raw = payload.get("raw_profile_rmse") or {}
    ence = ""
    band = ""
    cal_path = Path(args.cal_json) if args.cal_json else None
    if cal_path and cal_path.is_file():
        cal = json.loads(cal_path.read_text())
        tr = cal.get("test_raw") or {}
        ence = (tr.get("temperature") or {}).get("ence", "")
        band = ((tr.get("temperature_by_band") or {}).get("50-200") or {}).get("ence_T", "")
        if args.band_rmse:
            band = ((tr.get("temperature_by_band") or {}).get("50-200") or {}).get("crps_T", band)
    row = _annotate(
        {
            "cell": args.cell,
            "T": raw.get("temperature", payload.get("T", "")),
            "S": raw.get("salinity", payload.get("S", "")),
            "ence_T": ence,
            "T_50_200": band,
            "n": payload.get("n_samples", 623),
            "note": args.note or _label(args.cell),
            "source": str(Path(args.eval_json)),
        }
    )
    rows = [_annotate(r) for r in rows if r.get("cell") != args.cell]
    rows.append(row)
    spec = json.loads(Path(args.spec).read_text()) if Path(args.spec).is_file() else None
    _write_tsv(rows, tsv)
    _write_md(rows, spec, Path(args.md))
    print(f"appended {args.cell} T={row['T']}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    tsv = Path(args.tsv)
    with tsv.open() as f:
        rows = [_annotate(r) for r in csv.DictReader(f, delimiter="\t")]
    spec = json.loads(Path(args.spec).read_text()) if Path(args.spec).is_file() else None
    _write_tsv(rows, tsv)
    _write_md(rows, spec, Path(args.md))
    print(f"wrote {args.md}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fit")
    p.add_argument("--cache", default=str(CACHE))
    p.add_argument("--spec", default=str(SPEC_PATH))
    p.add_argument("--tsv", default=str(TSV_PATH))
    p.add_argument("--md", default=str(MD_PATH))
    p.add_argument("--alpha", type=float, default=1.0)
    p = sub.add_parser("append")
    p.add_argument("--cell", required=True)
    p.add_argument("--eval-json", required=True)
    p.add_argument("--cal-json", default="")
    p.add_argument("--note", default="")
    p.add_argument("--spec", default=str(SPEC_PATH))
    p.add_argument("--tsv", default=str(TSV_PATH))
    p.add_argument("--md", default=str(MD_PATH))
    p.add_argument("--band-rmse", action="store_true")
    p = sub.add_parser("report")
    p.add_argument("--spec", default=str(SPEC_PATH))
    p.add_argument("--tsv", default=str(TSV_PATH))
    p.add_argument("--md", default=str(MD_PATH))
    args = ap.parse_args()
    if args.cmd == "fit":
        return cmd_fit(args)
    if args.cmd == "append":
        return cmd_append(args)
    return cmd_report(args)


if __name__ == "__main__":
    raise SystemExit(main())
