"""Sample observed SSH (ADT, SLA) from DUACS L4 files at profile locations."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_UTILS = Path(__file__).resolve().parents[2] / "utils"
if str(_UTILS) not in sys.path:
    sys.path.insert(0, str(_UTILS))

from retrieve_sat import retrieve_satellite_data  # noqa: E402

from base.split_utils import juld_to_datetime  # noqa: E402


def _juld_to_astropy_jd(juld: float, *, dataset_tag: str, v2_src: str | None) -> float:
    from astropy.time import Time

    dt = juld_to_datetime(juld, dataset_tag=dataset_tag, v2_src=v2_src)
    return float(Time(dt).jd)


def _center_value(arr: np.ndarray) -> float:
    a = np.asarray(arr, dtype=np.float64)
    a = np.squeeze(a)
    if a.size == 0:
        return float("nan")
    if a.ndim == 0:
        return float(a)
    flat = a.ravel()
    mid = flat.size // 2
    return float(flat[mid])


def sample_ssh_obs(
    lat: np.ndarray,
    lon: np.ndarray,
    juld: np.ndarray,
    *,
    dataset_tag: str = "argo_v2",
    v2_src: str | None = None,
    max_batch_size: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return ``(ssh_obs_adt, ssh_obs_sla)`` per sample from DUACS ``sla``/``adt``.

    Uses ``retrieve_satellite_data`` with astronomical Julian day queries.
    """
    n = len(lat)
    adt = np.full(n, np.nan, dtype=np.float32)
    sla = np.full(n, np.nan, dtype=np.float32)
    products = {"ssh": ["adt", "sla"]}

    for start in range(0, n, max_batch_size):
        end = min(start + max_batch_size, n)
        queries = [
            (
                float(lat[i]),
                float(lon[i]),
                _juld_to_astropy_jd(float(juld[i]), dataset_tag=dataset_tag, v2_src=v2_src),
            )
            for i in range(start, end)
        ]
        results = retrieve_satellite_data(queries, products, spatial_pad=0, temporal_pad=0)
        for local_i, global_i in enumerate(range(start, end)):
            prod = results.get(local_i, {}).get("ssh")
            if not prod:
                continue
            data = prod.get("data", {})
            if "adt" in data:
                adt[global_i] = _center_value(data["adt"])
            if "sla" in data:
                sla[global_i] = _center_value(data["sla"])

    valid = np.isfinite(sla).sum()
    print(f"  ssh_obs: {valid}/{n} samples with finite SLA from DUACS")
    return adt, sla
