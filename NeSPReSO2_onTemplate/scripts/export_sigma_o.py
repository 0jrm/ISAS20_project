#!/usr/bin/env python3
"""Dai σ_o after H: 41-layer RMSE vs Argo, floored. Not 1 m RMSE and not CRPS-head σ."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_REPO = _ROOT.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evalphys.constants import LC_LAT_RANGE, LC_LON_RANGE
from preproc.h_operator import load_interfaces
from scripts.thermocline_scorecard import (
    DEFAULT_INTERFACES,
    _layer_rmse,
    _load_ckpt_pred,
    _remap_ts,
    _score_pair,
    _try_real_bundle,
)

FLOOR_T = 0.05
FLOOR_S = 0.02
# ponytail: GoM shelf is the 200 m isobath; skip the regime if bottom_depth is missing.
SHELF_M = 200.0


def _ckpt_a(seed: int) -> Path:
    return (
        _ROOT
        / "saved/phase5_matrix/A_CRPS_v2/models"
        / f"NeSPReSO2_ARGO_GoM_p5_A_CRPS_v2_p5_A_CRPS_v2_s{seed}_s2"
        / f"p5_A_CRPS_v2_s{seed}_s2"
        / "model_best.pth"
    )


def _heave_row(name, cfg_rel, model_dir, run_id):
    return {
        "name": name,
        "config_path": _ROOT / cfg_rel,
        "ckpts": [
            {
                "seed": 42,
                "path": _ROOT / "saved/models" / model_dir / run_id / "model_best.pth",
            }
        ],
    }


CANDIDATES = (
    {
        "name": "A_CRPS",
        "config_path": _ROOT / "saved/runs/phase5_matrix/configs/p5_A_CRPS_v2_s42.json",
        "ckpts": [{"seed": s, "path": _ckpt_a(s)} for s in (42, 43, 44)],
    },
    _heave_row(
        "HeaveFast",
        "config/argo/config_argo_heave_residual_fast.json",
        "NeSPReSO2_ARGO_GoM_heave_residual_fast",
        "heave_fast_s42",
    ),
    _heave_row(
        "conv3",
        "config/argo/config_argo_heave_fast_conv3.json",
        "NeSPReSO2_ARGO_GoM_heave_fast_conv3",
        "heave_conv3_s42",
    ),
    _heave_row(
        "ops",
        "config/argo/config_argo_heave_fast_ops.json",
        "NeSPReSO2_ARGO_GoM_heave_fast_ops",
        "heave_ops_s42",
    ),
    _heave_row(
        "bathy",
        "config/argo/config_argo_heave_fast_bathy.json",
        "NeSPReSO2_ARGO_GoM_heave_fast_bathy",
        "heave_bathy_s42",
    ),
    _heave_row(
        "bathy_wind",
        "config/argo/config_argo_heave_fast_bathy_wind.json",
        "NeSPReSO2_ARGO_GoM_heave_fast_bathy_wind",
        "heave_bathy_wind_s42",
    ),
)


def floor_sigma(rmse_t, rmse_s, floor_t=FLOOR_T, floor_s=FLOOR_S):
    rt = np.asarray(rmse_t, dtype=np.float64)
    rs = np.asarray(rmse_s, dtype=np.float64)
    sig_t = np.maximum(rt, floor_t)
    sig_s = np.maximum(rs, floor_s)
    hit_t = np.isfinite(rt) & (rt < floor_t)
    hit_s = np.isfinite(rs) & (rs < floor_s)
    return sig_t, sig_s, hit_t, hit_s


def _regime_masks(lat, lon, bottom_depth=None):
    n = int(lat.shape[0])
    lc = (
        (lat >= LC_LAT_RANGE[0])
        & (lat <= LC_LAT_RANGE[1])
        & (lon >= LC_LON_RANGE[0])
        & (lon <= LC_LON_RANGE[1])
    )
    out = {"all": np.ones(n, dtype=bool), "lc": lc, "complement": ~lc}
    if bottom_depth is not None:
        bd = np.asarray(bottom_depth, dtype=np.float64)
        if np.isfinite(bd).any():
            out["shelf"] = np.isfinite(bd) & (bd < SHELF_M)
    return out


def _as_list(a):
    return [None if not np.isfinite(x) else float(x) for x in np.asarray(a, dtype=np.float64)]


def _bool_list(a):
    return [bool(x) for x in np.asarray(a, dtype=bool)]


def _native_row(T_pred, S_pred, T_true, S_true, z, lat, lon, sla, mask):
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return {"T_50_200": None, "D26_rmse": None, "n": 0}
    sla_m = None if sla is None else np.asarray(sla)[m]
    scored = _score_pair(
        T_pred[m], S_pred[m], T_true[m], S_true[m], z, lat[m], lon[m], sla_m, ""
    )
    t_band = ((scored.get("ts_rmse") or {}).get("T") or {}).get("50-200")
    return {"T_50_200": t_band, "D26_rmse": scored.get("D26_rmse"), "n": int(m.sum())}


def _sigma_block(rmse_t, rmse_s, n_t, n_s, zmid, native):
    sig_t, sig_s, hit_t, hit_s = floor_sigma(rmse_t, rmse_s)
    return {
        "sigma_T": _as_list(sig_t),
        "sigma_S": _as_list(sig_s),
        "rmse_T": _as_list(rmse_t),
        "rmse_S": _as_list(rmse_s),
        "zmid_m": _as_list(zmid),
        "n": [int(x) for x in n_t],
        "n_S": [int(x) for x in n_s],
        "floor_T": _bool_list(hit_t),
        "floor_S": _bool_list(hit_s),
        "native": native,
    }


def _stack_stats(blocks, key):
    arr = np.stack([np.asarray(b[key], dtype=np.float64) for b in blocks])
    with np.errstate(all="ignore"):
        out = {
            f"{key}_mean": _as_list(np.nanmean(arr, axis=0)),
            f"{key}_min": _as_list(np.nanmin(arr, axis=0)),
            f"{key}_max": _as_list(np.nanmax(arr, axis=0)),
        }
        if arr.shape[0] > 1:
            out[f"{key}_std"] = _as_list(np.nanstd(arr, axis=0, ddof=1))
    return out


def _score_candidate(cand, packet, skipped):
    name = cand["name"]
    cfg_path = Path(cand["config_path"])
    bundle, err = _try_real_bundle(cfg_path)
    if bundle is None:
        skipped.append({"name": name, "reason": err or "no bundle"})
        return None
    T_true = bundle["T_true"]
    S_true = bundle["S_true"]
    z, lat, lon, sla = bundle["z"], bundle["lat"], bundle["lon"], bundle["sla"]
    bd = None
    if "bottom_depth" in bundle["cache"]:
        raw = np.asarray(bundle["cache"]["bottom_depth"], dtype=np.float64)
        if raw.size == bundle["cache"]["LAT"].shape[0]:
            bd = raw[bundle["idx"]]
    masks = _regime_masks(lat, lon, bd)
    seeds = {}
    for ck in cand["ckpts"]:
        path = Path(ck["path"])
        seed = ck["seed"]
        if not path.is_file():
            skipped.append({"name": name, "seed": seed, "reason": f"missing ckpt {path}"})
            continue
        pred = _load_ckpt_pred(path, bundle)
        if pred is None:
            skipped.append({"name": name, "seed": seed, "reason": f"decode failed {path}"})
            continue
        T_pred, S_pred = pred
        Ht_p, Hs_p, zmid = _remap_ts(packet, z, T_pred, S_pred, lat, lon)
        Ht_t, Hs_t, _ = _remap_ts(packet, z, T_true, S_true, lat, lon)
        regimes = {}
        for regime, mask in masks.items():
            if not mask.any():
                continue
            rmse_t, n_t = _layer_rmse(Ht_p[mask], Ht_t[mask])
            rmse_s, n_s = _layer_rmse(Hs_p[mask], Hs_t[mask])
            native = _native_row(T_pred, S_pred, T_true, S_true, z, lat, lon, sla, mask)
            regimes[regime] = _sigma_block(rmse_t, rmse_s, n_t, n_s, zmid, native)
        seeds[str(seed)] = regimes
    if not seeds:
        return None
    out = {"config": str(cfg_path), "cache": bundle["source"], "seeds": seeds}
    seed_ids = list(seeds)
    if len(seed_ids) > 1:
        agg = {}
        regimes = sorted(set().union(*(seeds[s].keys() for s in seed_ids)))
        for regime in regimes:
            blocks = [seeds[s][regime] for s in seed_ids if regime in seeds[s]]
            if len(blocks) < 2:
                continue
            stats = {}
            for key in ("sigma_T", "sigma_S", "rmse_T", "rmse_S"):
                stats.update(_stack_stats(blocks, key))
            stats["zmid_m"] = blocks[0]["zmid_m"]
            stats["n_seeds"] = len(blocks)
            agg[regime] = stats
        out["aggregate"] = agg
    return out


def _write_csv(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model", "seed", "regime", "k", "zmid_m",
        "sigma_T", "sigma_S", "rmse_T", "rmse_S", "floor_T", "floor_S", "n",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for model, rec in (payload.get("candidates") or {}).items():
            for seed, regimes in (rec.get("seeds") or {}).items():
                for regime, block in regimes.items():
                    zmid = block["zmid_m"]
                    for k, zm in enumerate(zmid):
                        w.writerow({
                            "model": model,
                            "seed": seed,
                            "regime": regime,
                            "k": k,
                            "zmid_m": zm,
                            "sigma_T": block["sigma_T"][k],
                            "sigma_S": block["sigma_S"][k],
                            "rmse_T": block["rmse_T"][k],
                            "rmse_S": block["rmse_S"][k],
                            "floor_T": block["floor_T"][k],
                            "floor_S": block["floor_S"][k],
                            "n": block["n"][k],
                        })


def _fmt(x, nd=3):
    if x is None:
        return "n/a"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(v):
        return "n/a"
    return f"{v:.{nd}f}"


def _layer_table(zmid, cols, band=None):
    zmid = np.asarray(zmid, dtype=np.float64)
    idx = range(len(zmid))
    if band is not None:
        lo, hi = band
        idx = [k for k in idx if lo <= zmid[k] <= hi]
    header = "| k | zmid_m | " + " | ".join(c[0] for c in cols) + " |"
    sep = "|---:|---:|" + "|".join("---:" for _ in cols) + "|"
    lines = [header, sep]
    for k in idx:
        cells = " | ".join(_fmt(c[1][k]) for c in cols)
        lines.append(f"| {k} | {_fmt(zmid[k], 1)} | {cells} |")
    return lines


def _headline_sigma_t(rec):
    if rec.get("aggregate") and "all" in rec["aggregate"]:
        block = rec["aggregate"]["all"]
        sig = block.get("sigma_T_mean") or block.get("sigma_T")
        return sig, block.get("zmid_m"), "mean"
    if rec.get("seeds"):
        seed = next(iter(rec["seeds"]))
        block = rec["seeds"][seed].get("all") or {}
        return block.get("sigma_T"), block.get("zmid_m"), f"s{seed}"
    return None, None, None


def _write_md(path: Path, payload: dict):
    cands = payload.get("candidates") or {}
    order = [c["name"] for c in CANDIDATES if c["name"] in cands]
    cols = []
    zmid = []
    for name in order:
        sig, zm, tag = _headline_sigma_t(cands[name])
        if sig is None:
            continue
        if not zmid and zm:
            zmid = zm
        label = f"{name} {tag}" if tag == "mean" else name
        cols.append((f"{label} σ_T", sig))

    lines = [
        "# 41-layer Dai σ_o after H",
        "",
        "A×CRPS is the frozen Phase 6 cell. HeaveFast is the v2 challenger. "
        "conv3, ops, bathy, and bathy_wind use the same H, floors, and chrono test "
        "on their own caches. They are ablations, not ingest, unless they beat HeaveFast "
        "on thermocline σ_T. The ingest file is this 41-layer table, not 1 m RMSE and not dense Σ. "
        "Markdown shows σ_T. σ_S is in the csv/json.",
        "",
        "H is reference-H from the 2024-01-05 18Z drifted GOMb0.04 background "
        "(9 GDAC columns plus mean p_ifc). It is not live thknss. Label is `h_kind=reference`.",
        "",
        "Floors are 0.05 °C and 0.02 psu (Argo analysis limits). "
        "σ = max(layer RMSE, floor).",
        "",
        "CRPS-as-σ_o is deferred until ENCE < 0.20 by band. "
        "A physical ENCE(T) = 0.236. HeaveFast ENCE(σ_D26) = 0.52.",
        "",
        f"Interfaces: `{payload.get('interfaces')}`.",
        "",
        "## Thermocline layers (zmid 50–200 m)",
        "",
    ]
    if cols and zmid:
        lines += _layer_table(zmid, cols, band=(50.0, 200.0))
    else:
        lines.append("No scored candidates.")
    lines += [
        "",
        "## Loop Current vs complement (zmid 50–200 m)",
        "",
        "LC is the evalphys box 24–28°N, 88–84°W. Complement is the rest of the chrono test. "
        "Shelf is omitted. This cache has no `bottom_depth`.",
        "",
    ]
    regime_note = False
    for model in order:
        rec = cands[model]
        seeds = rec.get("seeds") or {}
        for seed, regimes in seeds.items():
            for rg in ("lc", "complement"):
                block = regimes.get(rg)
                if not block:
                    continue
                nat = block.get("native") or {}
                zm = block["zmid_m"]
                st = block["sigma_T"]
                ks = [k for k, z in enumerate(zm) if z is not None and 50.0 <= float(z) <= 200.0]
                if not ks:
                    continue
                mean_t = sum(st[k] for k in ks) / len(ks)
                lines.append(
                    f"- {model} s{seed} {rg} n={nat.get('n')}: "
                    f"thermocline mean σ_T {_fmt(mean_t)} °C "
                    f"(native 50–200 T {_fmt(nat.get('T_50_200'))} °C)"
                )
                regime_note = True
    if not regime_note:
        lines.append("No LC/complement rows.")
    lines += ["", "## Full 41-layer σ_o", ""]
    if cols and zmid:
        lines += _layer_table(zmid, cols)
    skipped = payload.get("skipped") or []
    if skipped:
        lines += ["", "## Skipped", ""]
        for row in skipped:
            extra = f" seed {row['seed']}" if "seed" in row else ""
            lines.append(f"- {row.get('name')}{extra}: {row.get('reason')}")
    lines += [
        "",
        "## Native 1 m hydrography (not R)",
        "",
        "These numbers compare hydrography to the DA σ_o table. They are not the ingest file.",
        "",
    ]
    for model in order:
        rec = cands[model]
        for seed, regimes in (rec.get("seeds") or {}).items():
            nat = (regimes.get("all") or {}).get("native") or {}
            lines.append(
                f"- {model} s{seed}: 50–200 m T RMSE {_fmt(nat.get('T_50_200'))} °C, "
                f"D26 RMSE {_fmt(nat.get('D26_rmse'), 2)} m, n={nat.get('n')}"
            )
    deep_notes = []
    zm = np.asarray(zmid, dtype=np.float64) if zmid else np.array([])
    for name in order:
        rec = cands[name]
        seed0 = next(iter(rec.get("seeds") or {}), None)
        block = rec["seeds"][seed0].get("all") if seed0 else None
        hits = (block or {}).get("floor_T") or []
        if zm.size and hits and any(hits[k] for k in range(len(zm)) if zm[k] > 800.0):
            extra = " It is not the random-split v1 0.013." if name == "A_CRPS" else ""
            deep_notes.append(
                f"{name} deep (zmid>800) σ_T is at the 0.05 °C floor on at least one layer.{extra}"
            )
    lines += ["", "## Deep layers", ""]
    if deep_notes:
        lines.extend(f"- {n}" for n in deep_notes)
    else:
        lines.append("Deep σ_T was not scored (missing candidate).")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", default=None, help="ignored; candidates use the registry")
    parser.add_argument("--out-json", default=str(_REPO / "reports/sigma_o_hycom.json"))
    parser.add_argument("--out-csv", default=str(_REPO / "reports/sigma_o_hycom.csv"))
    parser.add_argument("--out-md", default=str(_REPO / "reports/sigma_o_hycom.md"))
    parser.add_argument("--interfaces", default=str(DEFAULT_INTERFACES))
    args = parser.parse_args(argv)

    packet = load_interfaces(args.interfaces)
    skipped = []
    candidates = {}
    for cand in CANDIDATES:
        rec = _score_candidate(cand, packet, skipped)
        if rec is not None:
            candidates[cand["name"]] = rec
    payload = {
        "h_kind": "reference",
        "interfaces": str(Path(args.interfaces).resolve()),
        "floors": {"T": FLOOR_T, "S": FLOOR_S},
        "lc_box": {"lat": LC_LAT_RANGE, "lon": LC_LON_RANGE},
        "candidates": candidates,
        "skipped": skipped,
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n")
    _write_csv(Path(args.out_csv), payload)
    _write_md(Path(args.out_md), payload)
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_csv}")
    print(f"wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
