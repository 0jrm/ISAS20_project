"""Time-causal neighbor pairs: one row is (target_i, source_j, deltas), not a cast."""

from __future__ import annotations

import numpy as np

EARTH_KM = 6371.0
PAIR_DTYPE = np.dtype(
    [
        ("target_i", np.int32),
        ("source_j", np.int32),
        ("dlat", np.float32),
        ("dlon", np.float32),
        ("dt_days", np.float32),
    ]
)
PAIR_IN_DIM = 9 + 64 + 3
MIX_IN_DIM = PAIR_IN_DIM + 1
LIVE_IN_DIM = 9 + 9 + 3
KM_EDGES = (0.0, 50.0, 100.0, 175.0, 250.0)
DT_EDGES = (3.0, 10.0, 20.0, 45.0)


def haversine_km(lat1, lon1, lat2, lon2):
    lat1 = np.radians(np.asarray(lat1, dtype=np.float64))
    lon1 = np.radians(np.asarray(lon1, dtype=np.float64))
    lat2 = np.radians(np.asarray(lat2, dtype=np.float64))
    lon2 = np.radians(np.asarray(lon2, dtype=np.float64))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _as_pairs(rows: list) -> np.ndarray:
    if not rows:
        return np.empty(0, dtype=PAIR_DTYPE)
    out = np.empty(len(rows), dtype=PAIR_DTYPE)
    for i, r in enumerate(rows):
        out[i] = r
    return out


def _bin_index(v, edges) -> int:
    e = np.asarray(edges, dtype=np.float64)
    b = int(np.searchsorted(e, float(v), side="right") - 1)
    return int(np.clip(b, 0, e.size - 2))


def _stratified_take(cand, km, dt, k, rng) -> np.ndarray:
    n_km = len(KM_EDGES) - 1
    n_dt = len(DT_EDGES) - 1
    buckets = [[] for _ in range(n_km * n_dt)]
    for j, kk, tt in zip(cand, km, dt):
        bi = _bin_index(kk, KM_EDGES)
        bj = _bin_index(tt, DT_EDGES)
        buckets[bi * n_dt + bj].append(int(j))
    for b in buckets:
        rng.shuffle(b)
    picked = []
    while len(picked) < int(k):
        progressed = False
        for b in buckets:
            if b and len(picked) < int(k):
                picked.append(b.pop())
                progressed = True
        if not progressed:
            break
    return np.asarray(picked, dtype=np.int32)


def _neighbors_for_target(
    i: int,
    lat: np.ndarray,
    lon: np.ndarray,
    juld: np.ndarray,
    *,
    max_km: float,
    min_dt: float,
    max_dt: float,
    k: int,
    later: bool = False,
    sample: str = "nearest",
) -> list:
    n = lat.shape[0]
    dt = juld[i] - juld
    lag = -dt if later else dt
    km = haversine_km(lat[i], lon[i], lat, lon)
    js = np.arange(n, dtype=np.int32)
    ok = (js != i) & (lag >= min_dt) & (lag <= max_dt) & (km <= max_km)
    cand = js[ok]
    if cand.size == 0 or k <= 0:
        return []
    take = min(int(k), cand.size)
    if sample == "stratified":
        rng = np.random.default_rng(10_007 + int(i))
        chosen = _stratified_take(cand, km[ok], dt[ok], take, rng)
    elif sample == "nearest":
        sc = (km[ok] / 50.0) ** 2 + (dt[ok] / 10.0) ** 2
        pick = np.argpartition(sc, take - 1)[:take]
        pick = pick[np.argsort(sc[pick], kind="stable")]
        chosen = cand[pick]
    else:
        raise ValueError(f"sample must be nearest|stratified, got {sample!r}")
    rows = []
    for j in chosen:
        j = int(j)
        rows.append((i, j, lat[i] - lat[j], lon[i] - lon[j], float(dt[j])))
    return rows


def build_pair_table(
    cache,
    split_indices,
    *,
    max_km=250,
    min_dt=3,
    max_dt=45,
    k_train=4,
    k_eval=1,
    later=False,
    sample_train="nearest",
) -> dict:
    lat = np.asarray(cache["LAT"], dtype=np.float64).ravel()
    lon = np.asarray(cache["LON"], dtype=np.float64).ravel()
    juld = np.asarray(cache["JULD"], dtype=np.float64).ravel()
    if not (lat.shape == lon.shape == juld.shape):
        raise ValueError("LAT, LON, JULD must be the same length")
    ks = {"train": int(k_train), "val": int(k_eval), "test": int(k_eval)}
    out = {}
    # ponytail: O(n_targets * n) scan; ceiling ~1e5 casts → ball tree / time-sorted window
    for split, k in ks.items():
        rows = []
        for i in split_indices.get(split, []):
            i = int(i)
            rows.extend(
                _neighbors_for_target(
                    i,
                    lat,
                    lon,
                    juld,
                    max_km=max_km,
                    min_dt=min_dt,
                    max_dt=max_dt,
                    k=k,
                    later=bool(later),
                    sample=str(sample_train) if split == "train" else "nearest",
                )
            )
        out[split] = _as_pairs(rows)
    return out


def nearest_eval_pairs(lat, lon, juld, *, max_km=250, min_dt=3, max_dt=45) -> np.ndarray:
    """One nearest legal earlier neighbor per target, or omit the target."""
    lat = np.asarray(lat, dtype=np.float64).ravel()
    lon = np.asarray(lon, dtype=np.float64).ravel()
    juld = np.asarray(juld, dtype=np.float64).ravel()
    if not (lat.shape == lon.shape == juld.shape):
        raise ValueError("lat, lon, juld must be the same length")
    rows = []
    for i in range(lat.shape[0]):
        rows.extend(
            _neighbors_for_target(
                i,
                lat,
                lon,
                juld,
                max_km=max_km,
                min_dt=min_dt,
                max_dt=max_dt,
                k=1,
                later=False,
                sample="nearest",
            )
        )
    return _as_pairs(rows)


def pack_pair_inputs(base_inputs, targets, pairs) -> np.ndarray:
    pairs = np.asarray(pairs)
    base = np.asarray(base_inputs, dtype=np.float32)
    pcs = np.asarray(targets, dtype=np.float32)
    n = int(pairs.shape[0])
    x = np.empty((n, PAIR_IN_DIM), dtype=np.float32)
    if n == 0:
        return x
    ti = pairs["target_i"]
    sj = pairs["source_j"]
    x[:, : base.shape[1]] = base[ti]
    x[:, base.shape[1] : base.shape[1] + pcs.shape[1]] = pcs[sj]
    off = base.shape[1] + pcs.shape[1]
    x[:, off] = pairs["dlat"]
    x[:, off + 1] = pairs["dlon"]
    x[:, off + 2] = np.asarray(pairs["dt_days"], dtype=np.float32) / 30.0
    return x


def pack_mix_inputs(base_inputs, argo_pcs, synth_pcs, pairs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each (i,j) twice: flag +1 Argo source PCs, flag -1 synth source PCs. y is Argo at i."""
    pairs = np.asarray(pairs)
    argo = np.asarray(argo_pcs, dtype=np.float32)
    synth = np.asarray(synth_pcs, dtype=np.float32)
    n = int(pairs.shape[0])
    xa = pack_pair_inputs(base_inputs, argo, pairs)
    xs = pack_pair_inputs(base_inputs, synth, pairs)
    x = np.empty((2 * n, MIX_IN_DIM), dtype=np.float32)
    y = np.empty((2 * n, argo.shape[1]), dtype=np.float32)
    idx = np.empty(2 * n, dtype=np.int64)
    if n == 0:
        return x, y, idx
    ti = pairs["target_i"].astype(np.int64)
    x[0::2, :PAIR_IN_DIM] = xa
    x[1::2, :PAIR_IN_DIM] = xs
    x[0::2, PAIR_IN_DIM] = 1.0
    x[1::2, PAIR_IN_DIM] = -1.0
    y[0::2] = argo[ti]
    y[1::2] = argo[ti]
    idx[0::2] = ti
    idx[1::2] = ti
    return x, y, idx


def pack_live_inputs(base_inputs, pairs) -> np.ndarray:
    """Target 9-d, source 9-d, deltas. Encoder runs on source inside the net."""
    pairs = np.asarray(pairs)
    base = np.asarray(base_inputs, dtype=np.float32)
    n = int(pairs.shape[0])
    x = np.empty((n, LIVE_IN_DIM), dtype=np.float32)
    if n == 0:
        return x
    if base.shape[1] != 9:
        raise ValueError(f"live pairs need 9-d base inputs, got {base.shape[1]}")
    ti = pairs["target_i"]
    sj = pairs["source_j"]
    x[:, :9] = base[ti]
    x[:, 9:18] = base[sj]
    x[:, 18] = pairs["dlat"]
    x[:, 19] = pairs["dlon"]
    x[:, 20] = np.asarray(pairs["dt_days"], dtype=np.float32) / 30.0
    return x


def _cols(arr, idx, n_hint: int) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim != 2:
        raise ValueError("profile array must be 2-d")
    if a.shape[1] == n_hint or a.shape[0] != n_hint:
        return a[:, idx]
    return np.asarray(a[idx]).T


def persistence_rmse(profiles, pairs) -> dict:
    pairs = np.asarray(pairs)
    if pairs.shape[0] == 0:
        nan = {"mean": float("nan"), "median": float("nan")}
        return {"T": dict(nan), "S": dict(nan)}
    n_hint = int(max(int(pairs["target_i"].max()), int(pairs["source_j"].max())) + 1)
    t_src = _cols(profiles["temperature"], pairs["source_j"], n_hint)
    t_tgt = _cols(profiles["temperature"], pairs["target_i"], n_hint)
    s_src = _cols(profiles["salinity"], pairs["source_j"], n_hint)
    s_tgt = _cols(profiles["salinity"], pairs["target_i"], n_hint)
    t_rmse = np.sqrt(np.nanmean((t_src - t_tgt) ** 2, axis=0))
    s_rmse = np.sqrt(np.nanmean((s_src - s_tgt) ** 2, axis=0))
    return {
        "T": {"mean": float(np.nanmean(t_rmse)), "median": float(np.nanmedian(t_rmse))},
        "S": {"mean": float(np.nanmean(s_rmse)), "median": float(np.nanmedian(s_rmse))},
    }
