"""CPC ONI / RONI lookup by profile date. No cache rebuild — splice at load."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

from base.split_utils import sample_dates

# CPC overlapping seasons; index = calendar month of the middle month (Jan=DJF).
_SEASONS = (
    "DJF",
    "JFM",
    "FMA",
    "MAM",
    "AMJ",
    "MJJ",
    "JJA",
    "JAS",
    "ASO",
    "SON",
    "OND",
    "NDJ",
)
ENSO_KEYS = ("oni", "roni")
_DEFAULT_DIR = Path(__file__).resolve().parents[2] / "data" / "indices"


def default_index_paths(index_dir: str | Path | None = None) -> dict[str, Path]:
    root = Path(index_dir) if index_dir else _DEFAULT_DIR
    return {"oni": root / "oni.ascii.txt", "roni": root / "RONI.ascii.txt"}


def parse_cpc_index(path: Path) -> dict[tuple[int, str], float]:
    """Map (year, SEAS) → anomaly. ONI has TOTAL+ANOM; RONI has ANOM only."""
    out: dict[tuple[int, str], float] = {}
    text = Path(path).read_text()
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 3 or parts[0] not in _SEASONS:
            continue
        seas, year_s, *rest = parts
        try:
            year = int(year_s)
            anom = float(rest[-1])
        except ValueError:
            continue
        out[(year, seas)] = anom
    return out


def _load_tables(index_dir: str | Path | None = None) -> dict[str, dict[tuple[int, str], float]]:
    paths = default_index_paths(index_dir)
    tables = {}
    for key, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"ENSO index file missing: {path}")
        tables[key] = parse_cpc_index(path)
    return tables


def lookup_enso(
    juld: np.ndarray,
    *,
    dataset_tag: str = "argo_v2",
    keys: tuple[str, ...] = ENSO_KEYS,
    index_dir: str | Path | None = None,
) -> np.ndarray:
    """(N, len(keys)) CPC anomalies; season = overlapping triplet whose middle month matches."""
    dates = sample_dates(np.asarray(juld, dtype=np.float64), dataset_tag=dataset_tag)
    years = dates.astype("datetime64[Y]").astype(int) + 1970
    months = dates.astype("datetime64[M]").astype(int) % 12  # 0=Jan
    tables = _load_tables(index_dir)
    n = dates.size
    out = np.full((n, len(keys)), np.nan, dtype=np.float32)
    for j, key in enumerate(keys):
        tab = tables[key]
        for i in range(n):
            seas = _SEASONS[int(months[i])]
            out[i, j] = tab.get((int(years[i]), seas), np.nan)
    return out


def enso_keys_wanted(input_params: Mapping[str, bool] | None) -> tuple[str, ...]:
    flags = input_params or {}
    return tuple(k for k in ENSO_KEYS if flags.get(k))


def enso_column_list(
    juld: np.ndarray,
    input_params: Mapping[str, bool],
    *,
    dataset_tag: str = "argo_v2",
    index_dir: str | Path | None = None,
) -> list[np.ndarray]:
    keys = enso_keys_wanted(input_params)
    if not keys:
        return []
    vals = np.nan_to_num(
        lookup_enso(juld, dataset_tag=dataset_tag, keys=keys, index_dir=index_dir), nan=0.0
    )
    return [vals[:, j] for j in range(vals.shape[1])]


def inject_enso_columns(
    inputs: np.ndarray,
    juld: np.ndarray,
    *,
    dataset_tag: str,
    input_params: Mapping[str, bool],
    n_enc_base: int = 6,
    index_dir: str | Path | None = None,
    expected_dim: int | None = None,
) -> np.ndarray:
    """Splice ONI/RONI after harmonic encodings when the cache was built without them."""
    keys = enso_keys_wanted(input_params)
    if not keys:
        return np.asarray(inputs, dtype=np.float32)
    x = np.asarray(inputs, dtype=np.float32)
    if expected_dim is not None and x.shape[1] == int(expected_dim):
        return x
    vals = lookup_enso(juld, dataset_tag=dataset_tag, keys=keys, index_dir=index_dir)
    vals = np.nan_to_num(vals, nan=0.0).astype(np.float32)
    n0 = int(n_enc_base)
    return np.concatenate([x[:, :n0], vals, x[:, n0:]], axis=1)
