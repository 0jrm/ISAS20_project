"""Train-only PC routing spec: easy PCs, aux columns, w_k, sigma floor."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from preproc.export_heave_ablation_cache import OP_NAMES
from preproc.preproc_isas_sat import ENCODING_KEYS, SAT_KEYS

EASY_INPUT_NAMES = ("timecos", "timesin", "loncos", "sss", "sst", "ssh")
EASY_PC_IDX = (0, 1, 2, 3, 32, 33)  # T1-4, S1-2
N_PC = 64
W_CLIP = (0.05, 4.0)


def column_names(n_dim: int) -> list[str]:
    if n_dim == 9:
        return list(ENCODING_KEYS) + list(SAT_KEYS)
    if n_dim == 28:
        return list(ENCODING_KEYS) + list(SAT_KEYS) + list(OP_NAMES)
    if n_dim == 30:
        return list(ENCODING_KEYS) + ["oni", "roni"] + list(SAT_KEYS) + list(OP_NAMES)
    if n_dim == 11:
        return list(ENCODING_KEYS) + ["oni", "roni"] + list(SAT_KEYS)
    raise ValueError(f"no PC-routing column map for input_dim={n_dim}")


def name_index(n_dim: int) -> dict[str, int]:
    names = column_names(n_dim)
    return {n: i for i, n in enumerate(names)}


def cols_for(n_dim: int, names: Sequence[str]) -> list[int]:
    idx = name_index(n_dim)
    return [idx[n] for n in names]


def aux_names(n_dim: int) -> list[str]:
    names = column_names(n_dim)
    core = set(ENCODING_KEYS) | set(SAT_KEYS) | {"oni", "roni"}
    return [n for n in names if n not in core]


def max_abs_r(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Per-column max |Pearson r| of Y vs X. Train rows only."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    n_pc = Y.shape[1]
    r_k = np.zeros(n_pc, dtype=np.float64)
    x_ok = np.isfinite(X).all(axis=0)
    for k in range(n_pc):
        y = Y[:, k]
        m_y = np.isfinite(y)
        best = 0.0
        for j in range(X.shape[1]):
            if not x_ok[j]:
                continue
            m = m_y & np.isfinite(X[:, j])
            if m.sum() < 20:
                continue
            xj = X[m, j]
            yk = y[m]
            sx = xj.std()
            sy = yk.std()
            if sx < 1e-12 or sy < 1e-12:
                continue
            r = float(np.dot(xj - xj.mean(), yk - yk.mean()) / (m.sum() * sx * sy))
            best = max(best, abs(r))
        r_k[k] = best
    return r_k


def weights_from_r(r_k: np.ndarray) -> np.ndarray:
    w = 1.0 - np.clip(np.asarray(r_k, dtype=np.float64), 0.0, 1.0) ** 2
    return np.clip(w, W_CLIP[0], W_CLIP[1])


def sigma_floor_from_r(r_k: np.ndarray) -> np.ndarray:
    return np.sqrt(np.clip(1.0 - np.clip(np.asarray(r_k, dtype=np.float64), 0.0, 1.0) ** 2, 0.0, 1.0))


def fit_ridge(X: np.ndarray, Y: np.ndarray, *, alpha: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.linear_model import Ridge

    X = np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0.0)
    Y = np.nan_to_num(np.asarray(Y, dtype=np.float64), nan=0.0)
    mdl = Ridge(alpha=alpha, fit_intercept=True)
    mdl.fit(X, Y)
    return np.asarray(mdl.coef_.T, dtype=np.float64), np.asarray(mdl.intercept_, dtype=np.float64)


def apply_ridge(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0.0) @ np.asarray(W) + np.asarray(b)


def load_spec(src: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(src, Mapping):
        return dict(src)
    path = Path(src)
    if not path.is_file():
        alt = Path(__file__).resolve().parents[2] / path
        if alt.is_file():
            path = alt
    return json.loads(Path(path).read_text())


def resolve_spec_path(raw: str | Path) -> Path:
    path = Path(raw)
    if path.is_file():
        return path
    alt = Path(__file__).resolve().parents[2] / path
    if alt.is_file():
        return alt
    cwd = Path.cwd() / path
    if cwd.is_file():
        return cwd
    return path
