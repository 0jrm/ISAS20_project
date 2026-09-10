#!/usr/bin/env python3
"""Correlate 32 T/S PCs with inputs.

Default: dataset PCs vs base+ops+bathy+wind (all rows).
With -c/-r: predicted μ PCs (and residuals) vs the columns the net actually saw
on chronological test. Pair the checkpoint with the cache it was trained on.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy import stats

_ROOT = Path(__file__).resolve().parents[1]
ROOT = _ROOT.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

OPS_PCA32 = ROOT / "data/cache/train_ready_heave_ops_pca32.pkl"
BATHY_WIND = ROOT / "data/cache/train_ready_heave_bathy_wind.pkl"
OUT_JSON = ROOT / "reports/pc_input_correlation.json"
OUT_MD = ROOT / "reports/pc_input_correlation.md"
PC_LABELS = [f"T_PC{i+1}" for i in range(32)] + [f"S_PC{i+1}" for i in range(32)]

# ─── input names ─────────────────────────────────────────────────────────────
BASE_NAMES = [
    "timecos", "timesin", "latcos", "latsin", "loncos", "lonsin",
    "sss", "sst", "ssh",
]
OP_NAMES = [
    "sst.grad_x@local", "sst.grad_y@local",
    "sst.grad_x@1deg",  "sst.grad_y@1deg",
    "sss.grad_x@local", "sss.grad_y@local",
    "sss.grad_x@1deg",  "sss.grad_y@1deg",
    "ssh.grad_x@local", "ssh.grad_y@local",
    "ssh.grad_x@1deg",  "ssh.grad_y@1deg",
    "ssh.laplacian@1deg",
    "sst.tendency@7d",  "ssh.tendency@7d",
    "ssh.geo_u@local",  "ssh.geo_v@local",
    "ssh.geo_u@1deg",   "ssh.geo_v@1deg",
]
BATHY_WIND_NAMES = ["bathy", "wind_u", "wind_v", "wind_speed"]


def load_cache(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def pearson_mat(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(n_in, n_pc) Pearson r and two-sided p. NaN if <20 finite pairs or zero var."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    n_in, n_pc = X.shape[1], Y.shape[1]
    r_mat = np.full((n_in, n_pc), np.nan)
    p_mat = np.full((n_in, n_pc), np.nan)
    for j in range(n_in):
        xj = X[:, j]
        mask = np.isfinite(xj)
        for k in range(n_pc):
            yk = Y[:, k]
            m2 = mask & np.isfinite(yk)
            if m2.sum() < 20:
                continue
            if xj[m2].std() < 1e-12 or yk[m2].std() < 1e-12:
                continue
            r, p = stats.pearsonr(xj[m2], yk[m2])
            r_mat[j, k] = r
            p_mat[j, k] = p
    return r_mat, p_mat


def _pc_block(title: str, r_mat: np.ndarray, names: list[str], offset: int) -> list[str]:
    block = [
        "",
        f"## {title}",
        "",
        "| PC | max|r| | best input | 2nd | 3rd |",
        "|----|--------|------------|-----|-----|",
    ]
    n_in = r_mat.shape[0]
    for k in range(32):
        col = r_mat[:, offset + k]
        ranked = sorted(
            ((abs(col[j]), names[j], col[j]) for j in range(n_in) if np.isfinite(col[j])),
            reverse=True,
        )
        top = ranked[:3]
        cells = [f"{nm}({r:+.3f})" for _, nm, r in top]
        while len(cells) < 3:
            cells.append("—")
        mx = top[0][0] if top else float("nan")
        block.append(
            f"| {PC_LABELS[offset + k]} | {mx:.3f} | {cells[0]} | {cells[1]} | {cells[2]} |"
        )
    return block


def _skill_r(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 20 or a[m].std() < 1e-12 or b[m].std() < 1e-12:
        return float("nan")
    return float(stats.pearsonr(a[m], b[m])[0])


def _top3(col: np.ndarray, names: list[str]) -> str:
    ranked = sorted(
        ((abs(col[j]), names[j], col[j]) for j in range(len(names)) if np.isfinite(col[j])),
        reverse=True,
    )
    if not ranked:
        return "—"
    _, nm, r = ranked[0]
    return f"{nm}({r:+.3f})"


def write_dataset_md(
    path: Path,
    *,
    n: int,
    names: list[str],
    r_mat: np.ndarray,
    p_mat: np.ndarray,
) -> None:
    n_in, n_pc = r_mat.shape
    sig = 0.01
    n_tests = n_in * n_pc
    alpha_bf = sig / n_tests
    lines = [
        "# PC–Input Correlation Report",
        "",
        f"N = {n:,} samples · {n_in} inputs · {n_pc} PCs (32 T + 32 S)",
        f"Bonferroni threshold: α={sig}/({n_in}×{n_pc}) = {alpha_bf:.2e}",
        "",
        "## Top |r| ≥ 0.3 (Bonferroni-significant)",
        "",
        "| Input | PC | r | p |",
        "|-------|-----|-----|-----|",
    ]
    hits = []
    for j, nm in enumerate(names):
        for k, pc in enumerate(PC_LABELS):
            r = r_mat[j, k]
            p = p_mat[j, k]
            if np.isfinite(r) and abs(r) >= 0.3 and p < alpha_bf:
                hits.append((abs(r), nm, pc, r, p))
    hits.sort(reverse=True)
    for _, nm, pc, r, p in hits[:60]:
        lines.append(f"| {nm} | {pc} | {r:+.3f} | {p:.1e} |")
    lines += ["", "## Strong correlates per input (|r| ≥ 0.5, p < Bonferroni)", ""]
    for j, nm in enumerate(names):
        strong = [
            (abs(r_mat[j, k]), PC_LABELS[k], r_mat[j, k], p_mat[j, k])
            for k in range(n_pc)
            if np.isfinite(r_mat[j, k]) and abs(r_mat[j, k]) >= 0.5 and p_mat[j, k] < alpha_bf
        ]
        if strong:
            strong.sort(reverse=True)
            top = ", ".join(f"{pc}(r={r:+.2f})" for _, pc, r, _ in strong[:5])
            lines.append(f"- **{nm}**: {top}")
    lines += _pc_block("All 32 T PCs — top-3 inputs (no |r| cutoff)", r_mat, names, 0)
    lines += _pc_block("All 32 S PCs — top-3 inputs (no |r| cutoff)", r_mat, names, 32)
    lines += [
        "",
        "## Summary: max |r| per input across all PCs",
        "",
        "| Input | max|r| | best PC |",
        "|-------|--------|---------|",
    ]
    for j, nm in enumerate(names):
        row = r_mat[j]
        valid = [(abs(row[k]), PC_LABELS[k], row[k]) for k in range(n_pc) if np.isfinite(row[k])]
        if valid:
            valid.sort(reverse=True)
            mr, bpc, rv = valid[0]
            lines.append(f"| {nm} | {mr:.3f} | {bpc}(r={rv:+.3f}) |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def run_dataset() -> int:
    print("Loading ops+pca32 cache …", flush=True)
    ops = load_cache(OPS_PCA32)
    print("Loading bathy+wind cache …", flush=True)
    bw = load_cache(BATHY_WIND)

    inputs_ops = np.asarray(ops["inputs"], dtype=np.float32)
    targets = np.asarray(ops["targets"], dtype=np.float32)

    N, D_ops = inputs_ops.shape
    print(f"  ops inputs: {inputs_ops.shape}, targets: {targets.shape}")

    n_base = len(BASE_NAMES)
    n_ops = len(OP_NAMES)
    if D_ops < n_base:
        base_cols = list(range(D_ops))
        base_names_used = BASE_NAMES[:D_ops]
        ops_cols = []
        ops_names_used = []
    else:
        base_cols = list(range(min(n_base, D_ops)))
        base_names_used = BASE_NAMES[: len(base_cols)]
        ops_cols = list(range(n_base, min(n_base + n_ops, D_ops)))
        ops_names_used = OP_NAMES[: len(ops_cols)]

    juld_ops = np.asarray(ops["JULD"])
    juld_bw = np.asarray(bw["JULD"])
    if np.array_equal(juld_ops, juld_bw):
        bw_inputs = np.asarray(bw["inputs"], dtype=np.float32)
        bw_extra = bw_inputs[:, n_base:]
        n_bw_extra = bw_extra.shape[1]
        bw_names_used = BATHY_WIND_NAMES[:n_bw_extra]
        print(f"  bathy+wind extra cols: {n_bw_extra} → {bw_names_used}")
    else:
        print("  WARN: JULD mismatch bathy_wind — skipping", file=sys.stderr)
        bw_extra = np.empty((N, 0), dtype=np.float32)
        bw_names_used = []

    all_cols = []
    all_names = []
    for c, nm in zip(base_cols, base_names_used):
        all_cols.append(inputs_ops[:, c])
        all_names.append(nm)
    for c, nm in zip(ops_cols, ops_names_used):
        all_cols.append(inputs_ops[:, c])
        all_names.append(nm)
    for ci in range(bw_extra.shape[1]):
        all_cols.append(bw_extra[:, ci])
        all_names.append(bw_names_used[ci])

    X = np.column_stack(all_cols).astype(np.float64)
    Y = targets.astype(np.float64)
    print(f"  inputs: {X.shape[1]} ({all_names[:5]}…), PCs: {Y.shape[1]}")
    r_mat, p_mat = pearson_mat(X, Y)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(
            {"input_names": all_names, "pc_labels": PC_LABELS, "r": r_mat.tolist(), "p": p_mat.tolist()},
            indent=2,
        )
    )
    print(f"  saved {OUT_JSON}")
    write_dataset_md(OUT_MD, n=N, names=all_names, r_mat=r_mat, p_mat=p_mat)
    print(f"  saved {OUT_MD}")
    return 0


def _load_model_test(config_path: str, checkpoint: str, split: str):
    import torch
    import model.model as module_arch
    import data_loader.data_loaders as module_data
    from parse_config import ConfigParser, validate_config
    from train import ensure_cache, set_seed
    from base.util import prepare_device, read_json

    cfg_dict = read_json(config_path)
    validate_config(cfg_dict)
    config = ConfigParser(cfg_dict, run_id="")
    set_seed(config.config.get("seed", 42))
    ensure_cache(config)
    device, _ = prepare_device(config["n_gpu"])
    model = config.init_obj("arch", module_arch).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)))
    model.eval()

    dl0 = dict(config["data_loader"]["args"])
    dl0["split"] = split
    dl0["shuffle"] = False
    ip = config.config.get("input_params")
    if ip:
        dl0["input_params"] = ip
    loader = getattr(module_data, config["data_loader"]["type"])(**dl0)
    return config, model, loader, device


def run_model(args: argparse.Namespace) -> int:
    import torch
    from scripts.ablate_inputs import column_names

    config, model, loader, device = _load_model_test(args.config, args.checkpoint, args.split)
    n_enc = int(config["arch"]["args"]["n_enc"])
    n_sat = int(config["arch"]["args"]["n_sat"])
    names = column_names(n_enc, n_sat)
    d = int(config["arch"]["args"]["output_dim"])
    xs, ys, yh, idx = [], [], [], []
    with torch.no_grad():
        for data, target, indices in loader:
            data = data.to(device)
            out = model(data)
            xs.append(data.cpu().numpy())
            ys.append(target.cpu().numpy())
            yh.append(out[:, :d].cpu().numpy())
            idx.append(indices.cpu().numpy())
    X = np.concatenate(xs, axis=0)
    Y = np.concatenate(ys, axis=0)
    Yhat = np.concatenate(yh, axis=0)
    te = np.concatenate(idx)
    if X.shape[1] != len(names):
        raise ValueError(f"inputs {X.shape[1]} != names {len(names)}")
    if Y.shape[1] != d or Yhat.shape[1] != d:
        raise ValueError(f"PC width {Y.shape[1]}/{Yhat.shape[1]} != output_dim {d}")
    resid = Y - Yhat
    r_truth, _p_truth = pearson_mat(X, Y)
    r_pred, p_pred = pearson_mat(X, Yhat)
    r_resid, _p_resid = pearson_mat(X, resid)
    skill = np.array([_skill_r(Yhat[:, k], Y[:, k]) for k in range(d)])

    unused_names, r_unused = [], None
    cache = loader.cache
    if BATHY_WIND.is_file() and "JULD" in cache:
        bw = load_cache(BATHY_WIND)
        if np.array_equal(np.asarray(cache["JULD"]), np.asarray(bw["JULD"])):
            extra = np.asarray(bw["inputs"], dtype=np.float64)[te, len(BASE_NAMES) :]
            unused_names = BATHY_WIND_NAMES[: extra.shape[1]]
            r_unused, _ = pearson_mat(extra, resid)

    n_test = int(Y.shape[0])
    out_json = Path(args.out_json or "../reports/pc_pred_correlation.json")
    if not out_json.is_absolute():
        out_json = _ROOT / out_json
    out_md = Path(args.out_md) if args.out_md else out_json.with_suffix(".md")
    if not out_md.is_absolute():
        out_md = _ROOT / out_md
    payload = {
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "split": args.split,
        "n_test": n_test,
        "input_names": names,
        "pc_labels": PC_LABELS[:d] if d != 64 else PC_LABELS,
        "r_truth": r_truth.tolist(),
        "r_pred": r_pred.tolist(),
        "r_resid": r_resid.tolist(),
        "p_pred": p_pred.tolist(),
        "pc_skill": skill.tolist(),
        "unused_names": unused_names,
        "r_resid_unused": None if r_unused is None else r_unused.tolist(),
        "note": (
            "Chrono test. r_pred = Pearson(input, model μ PC). "
            "r_resid = Pearson(input, truth−μ). pc_skill = Pearson(μ, truth) per PC. "
            "Not a retrain; pair checkpoint with its cache."
        ),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n")

    labels = payload["pc_labels"]
    lines = [
        "# Predicted PC–input correlation",
        "",
        f"Checkpoint `{args.checkpoint}`. Chronological `{args.split}`, n={n_test}.",
        "r_pred: inputs vs model μ PCs. Residual: truth − μ. Skill: r(μ, truth) per PC.",
        "",
        "## Per-PC skill r(μ, truth)",
        "",
        "| PC | r | truth best input | pred best input | residual max\\|r\\| |",
        "|----|--:|------------------|-----------------|------------------:|",
    ]
    n_t = d // 2
    for k in range(d):
        lab = labels[k] if k < len(labels) else f"PC{k+1}"
        lines.append(
            f"| {lab} | {skill[k]:+.3f} | {_top3(r_truth[:, k], names)} | "
            f"{_top3(r_pred[:, k], names)} | {np.nanmax(np.abs(r_resid[:, k])):.3f} |"
        )
        if k == n_t - 1:
            lines += ["", "### Salinity", ""]
            lines += [
                "| PC | r | truth best input | pred best input | residual max\\|r\\| |",
                "|----|--:|------------------|-----------------|------------------:|",
            ]
    lines += _pc_block("Predicted μ — all 32 T PCs, top-3 inputs", r_pred, names, 0)
    lines += _pc_block("Predicted μ — all 32 S PCs, top-3 inputs", r_pred, names, 32)
    lines += _pc_block("Residual (truth−μ) — all 32 T PCs", r_resid, names, 0)
    lines += _pc_block("Residual (truth−μ) — all 32 S PCs", r_resid, names, 32)
    if unused_names and r_unused is not None:
        lines += [
            "",
            "## Unused channels vs residual PCs (bathy/wind, not in this net)",
            "",
            "| channel | max\\|r\\| | best residual PC |",
            "|---------|--------:|------------------|",
        ]
        for j, nm in enumerate(unused_names):
            row = r_unused[j]
            ranked = sorted(
                ((abs(row[k]), labels[k], row[k]) for k in range(d) if np.isfinite(row[k])),
                reverse=True,
            )
            if ranked:
                mr, bpc, rv = ranked[0]
                lines.append(f"| {nm} | {mr:.3f} | {bpc}(r={rv:+.3f}) |")
    easy = list(range(4)) + [n_t, n_t + 1]
    hard_skill = float(np.nanmean([skill[k] for k in range(d) if k not in easy]))
    easy_skill = float(np.nanmean([skill[k] for k in easy if k < d]))
    lines += [
        "",
        "## Summary",
        "",
        f"Mean r(μ, truth) on easy PCs (T1–4, S1–2) = {easy_skill:.3f}; "
        f"on the other {d - len(easy)} PCs = {hard_skill:.3f}.",
        f"Wrote `{out_json}`.",
        "",
    ]
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n")
    print(f"n={n_test} easy_skill={easy_skill:.3f} hard_skill={hard_skill:.3f}")
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-c", "--config", default="")
    ap.add_argument("-r", "--checkpoint", default="")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out-json", default="")
    ap.add_argument("--out-md", default="")
    args = ap.parse_args()
    if bool(args.config) != bool(args.checkpoint):
        raise SystemExit("need both --config and --checkpoint, or neither")
    if args.config:
        return run_model(args)
    return run_dataset()


if __name__ == "__main__":
    raise SystemExit(main())
