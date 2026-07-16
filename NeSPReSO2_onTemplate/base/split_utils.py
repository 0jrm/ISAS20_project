"""Train/val/test assignment: chronological (dissertation default) or random (legacy)."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import numpy as np
import torch
from torch.utils.data import Subset, random_split


EPOCH_1950 = datetime(1950, 1, 1)
_EPOCH_1950_64 = np.datetime64("1950-01-01")
_ORDINAL_1_64 = np.datetime64("0001-01-01")  # == datetime.fromordinal(1)


def _parse_iso_date(s: str) -> date:
    return datetime.strptime(s.strip()[:10], "%Y-%m-%d").date()


def juld_to_datetime(juld: float, *, dataset_tag: str, v2_src: str | None = None) -> datetime:
    """Convert cache JULD to datetime (ISAS days-since-1950; ARGO MATLAB datenum)."""
    if dataset_tag == "isas20":
        return EPOCH_1950 + __import__("datetime").timedelta(days=float(juld))
    try:
        from nespreso.utils.time import datenum_to_datetime
    except ImportError:
        if not v2_src:
            raise
        import sys

        sys.path.insert(0, str(v2_src))
        from nespreso.utils.time import datenum_to_datetime
    return datenum_to_datetime(float(juld))


def dates_to_juld(dates, *, dataset_tag: str) -> np.ndarray:
    """Inverse of :func:`sample_dates`: calendar dates -> cache JULD (date-resolution).

    ``dates`` is anything ``numpy`` reads as ``datetime64[D]`` (ISO strings, date objects,
    datetime64). Exact inverse of ``sample_dates`` for whole days — see
    ``selfcheck.py::test_dates_to_juld_round_trip``.

    ISAS: days since 1950-01-01. ARGO: MATLAB datenum, where
    ``juld = date.toordinal() + 366`` (``sample_dates`` decodes ``d - 367`` days from
    ordinal 1, and ``(0001-01-01).toordinal() == 1``).
    """
    d64 = np.asarray(dates, dtype="datetime64[D]")
    if dataset_tag == "isas20":
        return (d64 - _EPOCH_1950_64).astype("timedelta64[D]").astype(np.float64)
    return (d64 - _ORDINAL_1_64).astype("timedelta64[D]").astype(np.float64) + 367.0


def sample_dates(
    juld: np.ndarray,
    *,
    dataset_tag: str,
    v2_src: str | None = None,
) -> np.ndarray:
    """Per-sample calendar dates as datetime64[D].

    Vectorized affine calendar arithmetic — equivalent to calling
    ``juld_to_datetime(v).date()`` per element (verified against that loop on real
    caches + exhaustive synthetic edge cases), but avoids one Python datetime
    object per sample. ``v2_src`` is accepted for signature compatibility but
    unused: date-only truncation never needs the external ``nespreso`` import.
    """
    d = np.asarray(juld, dtype=np.float64).ravel()
    if dataset_tag == "isas20":
        days_from_epoch = d
        epoch64 = _EPOCH_1950_64
    else:
        # MATLAB datenum: datetime.fromordinal(int(d) - 366) + timedelta(days=d % 1)
        # == datetime.fromordinal(1) + (d - 367) days, for d > 0 (int(d) + d % 1 == d).
        days_from_epoch = d - 367.0
        epoch64 = _ORDINAL_1_64
    micros = np.round(days_from_epoch * 86_400_000_000).astype(np.int64)
    return (epoch64 + micros.astype("timedelta64[us]")).astype("datetime64[D]")


def _split_lengths(n: int, train_frac: float, val_frac: float, test_frac: float) -> tuple[int, int, int]:
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        raise ValueError("train_frac + val_frac + test_frac must equal 1")
    train_len = int(n * train_frac)
    val_len = int(n * val_frac)
    test_len = n - train_len - val_len
    return train_len, val_len, test_len


def assign_random_indices(
    n: int,
    *,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    split_seed: int,
) -> dict[str, list[int]]:
    """Legacy ``torch.random_split`` indices (seed-pinned)."""
    class _DS(torch.utils.data.Dataset):
        def __len__(self):
            return n

        def __getitem__(self, i):
            return i

    lengths = _split_lengths(n, train_frac, val_frac, test_frac)
    g = torch.Generator().manual_seed(int(split_seed))
    train_sub, val_sub, test_sub = random_split(_DS(), list(lengths), generator=g)
    return {
        "train": list(train_sub.indices),
        "val": list(val_sub.indices),
        "test": list(test_sub.indices),
    }


def assign_chronological_indices(
    juld: np.ndarray,
    *,
    dataset_tag: str,
    split_config: dict[str, dict[str, str]],
    v2_src: str | None = None,
    unassigned: str = "exclude",
) -> dict[str, list[int]]:
    """Assign samples by calendar date ranges in ``split_config``."""
    dates = sample_dates(juld, dataset_tag=dataset_tag, v2_src=v2_src)
    out: dict[str, list[int]] = {k: [] for k in ("train", "val", "test")}
    windows: dict[str, tuple[date, date]] = {}
    for split in ("train", "val", "test"):
        cfg = split_config.get(split) or {}
        if "start" not in cfg or "end" not in cfg:
            raise ValueError(f"chronological split requires {split}.start and {split}.end")
        windows[split] = (_parse_iso_date(cfg["start"]), _parse_iso_date(cfg["end"]))

    for i, d in enumerate(dates):
        day = date.fromisoformat(str(d)[:10])
        assigned = False
        for split, (start, end) in windows.items():
            if start <= day <= end:
                out[split].append(i)
                assigned = True
                break
        if not assigned and unassigned == "train":
            out["train"].append(i)

    for split in ("train", "val", "test"):
        out[split] = sorted(out[split])
    return out


def assign_chronological_fraction_indices(
    juld: np.ndarray,
    *,
    dataset_tag: str,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    v2_src: str | None = None,
) -> dict[str, list[int]]:
    """Sort by time, then take contiguous 70/15/15 (or configured) blocks."""
    order = np.argsort(sample_dates(juld, dataset_tag=dataset_tag, v2_src=v2_src))
    n = len(order)
    train_len, val_len, test_len = _split_lengths(n, train_frac, val_frac, test_frac)
    train_idx = order[:train_len].tolist()
    val_idx = order[train_len : train_len + val_len].tolist()
    test_idx = order[train_len + val_len :].tolist()
    return {"train": train_idx, "val": val_idx, "test": test_idx}


def build_split_indices(
    n: int,
    juld: np.ndarray | None,
    dl_cfg: dict[str, Any],
    *,
    dataset_tag: str = "unknown",
    v2_src: str | None = None,
) -> dict[str, list[int]]:
    mode = dl_cfg.get("split_mode", "random")
    if mode == "random":
        return assign_random_indices(
            n,
            train_frac=float(dl_cfg.get("train_frac", 0.7)),
            val_frac=float(dl_cfg.get("val_frac", 0.15)),
            test_frac=float(dl_cfg.get("test_frac", 0.15)),
            split_seed=int(dl_cfg.get("split_seed", 42)),
        )
    if mode == "chronological":
        if juld is None:
            raise ValueError("chronological split requires JULD in cache")
        split_config = dl_cfg.get("split_config")
        if split_config:
            return assign_chronological_indices(
                juld,
                dataset_tag=dataset_tag,
                split_config=split_config,
                v2_src=v2_src,
                unassigned=str(dl_cfg.get("unassigned", "exclude")),
            )
        return assign_chronological_fraction_indices(
            juld,
            dataset_tag=dataset_tag,
            train_frac=float(dl_cfg.get("train_frac", 0.7)),
            val_frac=float(dl_cfg.get("val_frac", 0.15)),
            test_frac=float(dl_cfg.get("test_frac", 0.15)),
            v2_src=v2_src,
        )
    raise ValueError(f"split_mode must be random|chronological, got {mode!r}")


def subsets_from_indices(dataset, indices: dict[str, list[int]]) -> dict[str, Subset]:
    return {k: Subset(dataset, v) for k, v in indices.items()}


def split_summary(
    indices: dict[str, list[int]],
    juld: np.ndarray | None = None,
    *,
    dataset_tag: str = "unknown",
    v2_src: str | None = None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {"counts": {k: len(v) for k, v in indices.items()}}
    if juld is None:
        return summary
    dates = sample_dates(juld, dataset_tag=dataset_tag, v2_src=v2_src)
    by_split: dict[str, dict[str, int]] = {}
    for split, idx in indices.items():
        if not idx:
            by_split[split] = {}
            continue
        yrs = [str(d)[:4] for d in dates[idx]]
        by_split[split] = dict(sorted({y: yrs.count(y) for y in set(yrs)}.items()))
    summary["by_year"] = by_split
    all_idx = sorted(set(indices["train"]) | set(indices["val"]) | set(indices["test"]))
    if all_idx:
        d0, d1 = dates[all_idx[0]], dates[all_idx[-1]]
        summary["date_range"] = {"min": str(d0), "max": str(d1)}
    return summary
