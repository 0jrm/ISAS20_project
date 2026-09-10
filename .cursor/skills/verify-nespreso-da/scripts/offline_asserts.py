#!/usr/bin/env python3
"""CPU asserts for NeSPReSO vs xb residuals. Toy by default. Live nc is optional."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

BANDS = (("0-50", 0.0, 50.0), ("50-200", 50.0, 200.0), ("200-800", 200.0, 800.0), (">800", 800.0, np.inf))
AGE_BINS = (("(0,3]", 0.0, 3.0), ("(3,10]", 3.0, 10.0), ("(10,20]", 10.0, 20.0), ("(20,45]", 20.0, 45.0), (">45", 45.0, np.inf))
EARTH_KM = 6371.0
NEAR_KM = 100.0


def band_mask(z: np.ndarray, lo: float, hi: float) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    return (z >= lo) & (z < hi)


def error_corr(e_a: np.ndarray, e_b: np.ndarray, ok: np.ndarray) -> float:
    aa = e_a[ok]
    bb = e_b[ok]
    if aa.size < 3:
        return float("nan")
    if np.std(aa) == 0.0 or np.std(bb) == 0.0:
        return float("nan")
    return float(np.corrcoef(aa, bb)[0, 1])


def blend_weight(var_n: float, var_x: float, rho: float) -> float:
    cov = rho * np.sqrt(var_n * var_x)
    den = var_n + var_x - 2.0 * cov
    if den <= 0.0:
        return 0.0
    w = (var_x - cov) / den
    return float(np.clip(w, 0.0, 1.0))


def blend_rmse(e_n: np.ndarray, e_x: np.ndarray, ok: np.ndarray, w: float) -> float:
    mix = w * e_n[ok] + (1.0 - w) * e_x[ok]
    return float(np.sqrt(np.mean(mix * mix)))


def _debiased_rmse(rmse: float, bias: float) -> float:
    if not (np.isfinite(rmse) and np.isfinite(bias)):
        return float("nan")
    return float(np.sqrt(max(rmse * rmse - bias * bias, 0.0)))


def table_from_errors(z: np.ndarray, e_n: np.ndarray, e_x: np.ndarray, valid: np.ndarray) -> dict:
    rows = {}
    for name, lo, hi in BANDS:
        m = valid & band_mask(z, lo, hi)[None, :] & np.isfinite(e_n) & np.isfinite(e_x)
        rho = error_corr(e_n, e_x, m)
        vn = float(np.mean(e_n[m] ** 2)) if m.any() else float("nan")
        vx = float(np.mean(e_x[m] ** 2)) if m.any() else float("nan")
        bn = float(np.mean(e_n[m])) if m.any() else float("nan")
        bx = float(np.mean(e_x[m])) if m.any() else float("nan")
        rmse_n = float(np.sqrt(vn)) if np.isfinite(vn) else float("nan")
        rmse_x = float(np.sqrt(vx)) if np.isfinite(vx) else float("nan")
        w = blend_weight(vn, vx, rho) if np.isfinite(rho) else float("nan")
        rows[name] = {
            "n": int(m.sum()),
            "rho": rho,
            "rmse_nes": rmse_n,
            "rmse_xb": rmse_x,
            "bias_nes": bn,
            "bias_xb": bx,
            "rmse_nes_debiased": _debiased_rmse(rmse_n, bn),
            "rmse_xb_debiased": _debiased_rmse(rmse_x, bx),
            "w_nes": w,
            "rmse_blend": blend_rmse(e_n, e_x, m, w) if np.isfinite(w) else float("nan"),
        }
    return rows


def effective_dof(E: np.ndarray) -> float:
    E = np.asarray(E, dtype=np.float64)
    if E.ndim != 2 or E.shape[0] < 2 or E.shape[1] < 1:
        return float("nan")
    C = np.atleast_2d(np.cov(E, rowvar=False, ddof=1))
    ss = float(np.sum(C * C))
    if ss <= 0.0:
        return float("nan")
    tr = float(np.trace(C))
    return float((tr * tr) / ss)


def dof_table(z: np.ndarray, e_n: np.ndarray, e_x: np.ndarray, valid: np.ndarray) -> dict:
    rows = {}
    for name, lo, hi in BANDS:
        bm = band_mask(z, lo, hi)
        n_levels = int(bm.sum())
        if n_levels == 0:
            rows[name] = {"nes": float("nan"), "xb": float("nan"), "n_casts": 0, "n_levels": 0}
            continue
        sub = valid[:, bm] & np.isfinite(e_n[:, bm]) & np.isfinite(e_x[:, bm])
        complete = sub.all(axis=1)
        n_casts = int(complete.sum())
        rows[name] = {
            "nes": effective_dof(e_n[complete][:, bm]),
            "xb": effective_dof(e_x[complete][:, bm]),
            "n_casts": n_casts,
            "n_levels": n_levels,
        }
    return rows


def desroziers_table(z: np.ndarray, d_b: np.ndarray, d_a: np.ndarray, valid: np.ndarray) -> dict:
    rows = {}
    for name, lo, hi in BANDS:
        m = valid & band_mask(z, lo, hi)[None, :]
        if not m.any():
            rows[name] = {"n": 0, "R_hat": float("nan"), "OmB_mse": float("nan")}
            continue
        db = d_b[m]
        da = d_a[m]
        rows[name] = {
            "n": int(m.sum()),
            "R_hat": float(np.mean(db * da)),
            "OmB_mse": float(np.mean(db * db)),
        }
    return rows


def haversine_km(lat0: float, lon0: float, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    p0 = np.radians(lat0)
    l0 = np.radians(lon0)
    p = np.radians(lat)
    dlat = p0 - p
    dlon = l0 - np.radians(lon)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(p0) * np.cos(p) * np.sin(dlon / 2.0) ** 2
    return EARTH_KM * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def in_file_argo_age_days(lat: np.ndarray, lon: np.ndarray, times) -> np.ndarray:
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    tsec = np.array([ti.timestamp() for ti in times], dtype=np.float64)
    n = lat.size
    age = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        earlier = tsec < tsec[i]
        if not earlier.any():
            continue
        dist = haversine_km(float(lat[i]), float(lon[i]), lat, lon)
        ok = earlier & (dist <= NEAR_KM) & np.isfinite(dist)
        if not ok.any():
            continue
        age[i] = (tsec[i] - tsec[ok]).min() / 86400.0
    return age


def age_bin_rmse(age: np.ndarray, z: np.ndarray, e_n: np.ndarray, e_x: np.ndarray, valid: np.ndarray) -> dict:
    bm = band_mask(z, 50.0, 200.0)
    out = {}
    specs = list(AGE_BINS) + [("none", None, None)]
    for name, lo, hi in specs:
        if name == "none":
            pick = ~np.isfinite(age)
        else:
            pick = np.isfinite(age) & (age > lo) & (age <= hi)
        m = valid & pick[:, None] & bm[None, :] & np.isfinite(e_n) & np.isfinite(e_x)
        out[name] = {
            "n_casts": int(pick.sum()),
            "rmse_nes": float(np.sqrt(np.mean(e_n[m] ** 2))) if m.any() else float("nan"),
            "rmse_xb": float(np.sqrt(np.mean(e_x[m] ** 2))) if m.any() else float("nan"),
        }
    return out


def estimator_blend_hits(slices: dict) -> list:
    hits = []
    for sl, rows in slices.items():
        for band, r in rows.items():
            w, rb, rx = r["w_nes"], r["rmse_blend"], r["rmse_xb"]
            if np.isfinite(w) and w > 0.3 and np.isfinite(rb) and np.isfinite(rx) and rb < rx:
                hits.append({"slice": sl, "band": band})
    return hits


def complementarity_50_200(slices: dict) -> dict:
    rho = {}
    for sl, rows in slices.items():
        r = rows.get("50-200", {}).get("rho")
        if r is None or not np.isfinite(r):
            continue
        rho[sl] = float(r)
    vals = list(rho.values())
    return {
        "rho": rho,
        "high_ge_0p7": [s for s, v in rho.items() if v >= 0.7],
        "low_le_0p4": [s for s, v in rho.items() if v <= 0.4],
        "all_high": bool(vals) and all(v >= 0.7 for v in vals),
    }


def autopsy_2024(xb: dict, slices: dict) -> dict:
    m = slices["2024"]
    n = int(m.sum())
    dates = np.asarray(xb["analysis_date"])[m]
    uniq, counts = np.unique(dates, return_counts=True)
    return {
        "n": n,
        "n_unique_analysis_date": int(uniq.size),
        "casts_per_date": [{"date": str(d), "n": int(c)} for d, c in zip(uniq, counts)],
        "ood_frac": float(np.mean(slices["ood"][m])) if n else float("nan"),
        "lc_frac": float(np.mean(slices["lc"][m])) if n else float("nan"),
    }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(type(o).__name__)


def _dump(obj) -> None:
    print(json.dumps(obj, indent=2, default=_json_default))


def selfcheck() -> None:
    z = np.array([25.0, 100.0, 400.0, 1200.0])
    e_n = np.array([[1.0, 2.0, 0.5, 0.1], [1.2, 2.2, 0.4, 0.0]])
    e_x = e_n.copy()
    ok = np.ones_like(e_n, dtype=bool)
    rho = error_corr(e_n, e_x, ok)
    assert abs(rho - 1.0) < 1e-12, "identical errors must have rho=1"
    w = blend_weight(float(np.mean(e_n**2)), float(np.mean(e_x**2)), rho)
    assert w == 0.0 or abs(blend_rmse(e_n, e_x, ok, w) - float(np.sqrt(np.mean(e_x**2)))) < 1e-12

    rng = np.random.default_rng(0)
    a = rng.normal(size=(8, 4))
    b = rng.normal(size=(8, 4))
    rho0 = error_corr(a, b, np.ones_like(a, dtype=bool))
    vn = float(np.mean(a**2))
    vx = float(np.mean(b**2))
    w0 = blend_weight(vn, vx, rho0)
    rmse_b = blend_rmse(a, b, np.ones_like(a, dtype=bool), w0)
    assert rmse_b <= np.sqrt(vn) + 1e-12
    assert rmse_b <= np.sqrt(vx) + 1e-12

    rows = table_from_errors(z, e_n, e_x, ok)
    assert rows["50-200"]["n"] == 2
    bias = float(np.mean(e_n[:, 1]))
    assert abs(rows["50-200"]["bias_nes"] - bias) < 1e-12, "band bias is mean residual"

    u = rng.normal(size=40)
    E1 = np.repeat(u[:, None], 6, axis=1)
    assert abs(effective_dof(E1) - 1.0) < 1e-9, "rank-1 identical-depth errors have dof~1"
    E0 = rng.normal(size=(4000, 6))
    d0 = effective_dof(E0)
    assert abs(d0 - 6.0) < 0.4, "independent columns have dof~n_levels"

    db = np.array([[1.0, 2.0]])
    da = np.array([[3.0, 4.0]])
    z2 = np.array([100.0, 150.0])
    dz = desroziers_table(z2, db, da, np.ones((1, 2), dtype=bool))
    assert abs(dz["50-200"]["R_hat"] - 5.5) < 1e-12, "R_hat = mean(d_b * d_a)"
    assert abs(dz["50-200"]["OmB_mse"] - 2.5) < 1e-12, "OmB_mse = mean(d_b^2)"
    assert dz["50-200"]["n"] == 2

    lat = np.array([25.0, 25.0, 0.0])
    lon_off = 1.0 / (111.32 * np.cos(np.radians(25.0)))
    lon = np.array([-90.0, -90.0 + lon_off, 10.0])
    t0 = datetime(2024, 1, 1)
    times = [t0, t0 + timedelta(days=5), t0 + timedelta(days=2)]
    age = in_file_argo_age_days(lat, lon, times)
    assert abs(age[1] - 5.0) < 1e-9, "later of two 1km casts is 5 days"
    assert not np.isfinite(age[0]), "earliest near neighbor has no earlier cast"
    assert not np.isfinite(age[2]), "isolated cast is none"

    fake = {
        "all": {"50-200": {"rho": 0.8, "w_nes": 0.1, "rmse_blend": 1.0, "rmse_xb": 0.9}},
        "2024": {"50-200": {"rho": 0.2, "w_nes": 0.4, "rmse_blend": 0.7, "rmse_xb": 0.9}},
    }
    c = complementarity_50_200(fake)
    assert c["all_high"] is False
    assert c["low_le_0p4"] == ["2024"]
    assert c["high_ge_0p7"] == ["all"]
    assert estimator_blend_hits(fake) == [{"slice": "2024", "band": "50-200"}]

    print("selfcheck: offline_asserts ok")


def check_stats(path: Path, model: str) -> dict:
    stats = json.loads(path.read_text())
    sl = stats["slices"]["all"]
    if "heave_vs_shape" not in sl["xb"]:
        raise SystemExit("stats.json xb missing heave_vs_shape")
    if model not in sl["models"]:
        raise SystemExit(f"stats.json missing model {model}")
    if "heave_vs_shape" not in sl["models"][model]:
        raise SystemExit(f"stats.json {model} missing heave_vs_shape")
    if "2024" not in stats["slices"]:
        raise SystemExit("stats.json missing 2024 slice")
    digest = sha256_file(path)
    out = {"sha256": digest, "n": stats.get("n"), "n_ood": stats.get("n_ood"), "model": model, "slices": {}}
    for name in ("all", "2024", "ood", "lc"):
        if name not in stats["slices"]:
            continue
        s = stats["slices"][name]
        xb = s.get("xb", {})
        md = s.get("models", {}).get(model, {})
        out["slices"][name] = {
            "n": s.get("n"),
            "xb_T_rmse": xb.get("T", {}).get("rmse"),
            "model_T_rmse": md.get("T", {}).get("rmse"),
            "xb_heave_vs_shape": xb.get("heave_vs_shape"),
            "model_heave_vs_shape": md.get("heave_vs_shape"),
        }
    _dump(out)
    return out


def _load_compare():
    repo = Path(__file__).resolve().parents[4]
    api = repo / "NeSPReSO2_onTemplate"
    if str(api) not in sys.path:
        sys.path.insert(0, str(api))
    import importlib.util

    cmp_path = api / "scripts" / "compare_xb_argo.py"
    spec = importlib.util.spec_from_file_location("compare_xb_argo", cmp_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def live_blend(xb_path: Path, nes_path: Path, model: str) -> dict:
    mod = _load_compare()
    xb = mod.load_xb(xb_path)
    preds = mod.load_preds_nc(nes_path)
    if model not in preds:
        raise SystemExit(f"{nes_path} has no T_{model}")
    e_n = preds[model]["T"] - xb["T_argo"]
    e_x = xb["T_xb"] - xb["T_argo"]
    valid = xb["valid_T"]
    slices_m = mod._slice_masks(xb)
    slices = {}
    for name, mask in slices_m.items():
        slices[name] = table_from_errors(xb["z"], e_n, e_x, valid & mask[:, None])
    d_b = xb["T_argo"] - xb["T_xb"]
    d_a = xb["T_argo"] - xb["T_xa"]
    xa_ok = valid & np.isfinite(xb["T_xa"])
    payload = {
        "model": model,
        "bands": slices["all"],
        "slices": slices,
        "desroziers": desroziers_table(xb["z"], d_b, d_a, xa_ok),
        "dof": dof_table(xb["z"], e_n, e_x, valid),
        "in_file_argo_age_km100": age_bin_rmse(in_file_argo_age_days(xb["lat"], xb["lon"], xb["times"]), xb["z"], e_n, e_x, valid),
        "autopsy_2024": autopsy_2024(xb, slices_m),
        "complementarity_50_200": complementarity_50_200(slices),
        "estimator_blend_hits": estimator_blend_hits(slices),
    }
    _dump(payload)
    return payload


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selfcheck", action="store_true")
    p.add_argument("--stats", default="", help="hash stats.json and require heave_vs_shape keys")
    p.add_argument("--model", default="A_CRPS_z32")
    p.add_argument("--xb", default="")
    p.add_argument("--nes", default="")
    args = p.parse_args(argv)
    if args.selfcheck:
        selfcheck()
        return 0
    if args.stats:
        check_stats(Path(args.stats), args.model)
        return 0
    if args.xb and args.nes:
        live_blend(Path(args.xb), Path(args.nes), args.model)
        return 0
    p.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
