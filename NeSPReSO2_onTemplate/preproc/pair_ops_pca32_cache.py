#!/usr/bin/env python3
"""Clone heave-ops inputs onto the 32-PC A_CRPS_z32 PCA (same profiles, new pickle)."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_OPS = _ROOT / "data/cache/train_ready_heave_ops.pkl"
_DEFAULT_PCA32 = _ROOT / "data/cache/train_ready_4ee013852d33.pkl"
_DEFAULT_OUT = _ROOT / "data/cache/train_ready_heave_ops_pca32.pkl"
_PCA_KEYS = ("pca_models", "targets", "outputs", "weights", "pcs_by_name")


def pair(ops_path: Path, pca32_path: Path, out_path: Path) -> Path:
    with open(ops_path, "rb") as f:
        ops = pickle.load(f)
    with open(pca32_path, "rb") as f:
        pca32 = pickle.load(f)
    if not np.array_equal(np.asarray(ops["JULD"]), np.asarray(pca32["JULD"])):
        raise ValueError("JULD mismatch; refuse to pair")
    t_ops = np.asarray(ops["profiles"]["temperature"])
    t_32 = np.asarray(pca32["profiles"]["temperature"])
    if t_ops.shape != t_32.shape or float(np.nanmax(np.abs(t_ops - t_32))) != 0.0:
        raise ValueError("temperature profiles mismatch; refuse to pair")
    n_t = int(pca32["outputs"]["temperature"])
    n_s = int(pca32["outputs"]["salinity"])
    if n_t != 32 or n_s != 32:
        raise ValueError(f"expected 32+32 PCA, got {n_t}+{n_s}")
    if int(ops["outputs"]["temperature"]) == 32:
        raise ValueError(f"{ops_path} already has 32-PC outputs")
    out = dict(ops)
    for k in _PCA_KEYS:
        if k not in pca32:
            raise KeyError(k)
        out[k] = pca32[k]
    out["paired_from"] = {"ops": str(ops_path), "pca32": str(pca32_path)}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(out, f, protocol=4)
    tmp.replace(out_path)
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ops", type=Path, default=_DEFAULT_OPS)
    ap.add_argument("--pca32", type=Path, default=_DEFAULT_PCA32)
    ap.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    args = ap.parse_args()
    path = pair(args.ops, args.pca32, args.out)
    with open(path, "rb") as f:
        c = pickle.load(f)
    print(
        path,
        "inputs",
        c["inputs"].shape,
        "targets",
        c["targets"].shape,
        "T_pc",
        int(c["outputs"]["temperature"]),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
