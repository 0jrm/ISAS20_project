#!/usr/bin/env python3
"""Test-split RMSE + thermocline metrics for HeaveFast ablations. Pair each ckpt with its cache."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_REPO = _ROOT.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from base.util import read_json
from parse_config import ConfigParser, validate_config
from eval_run import main as eval_main, write_report
from scripts.thermocline_scorecard import _load_ckpt_pred, _score_pair, _try_real_bundle

RUNS = (
    ("HeaveFast", "config/argo/config_argo_heave_residual_fast.json",
     "saved/models/NeSPReSO2_ARGO_GoM_heave_residual_fast/heave_fast_s42/model_best.pth",
     _REPO / "reports/eval_heave_fast_s42.json"),
    ("conv3", "config/argo/config_argo_heave_fast_conv3.json",
     "saved/models/NeSPReSO2_ARGO_GoM_heave_fast_conv3/heave_conv3_1deg_s42/model_best.pth",
     _REPO / "reports/eval_heave_conv3_1deg_s42.json"),
    ("ops", "config/argo/config_argo_heave_fast_ops.json",
     "saved/models/NeSPReSO2_ARGO_GoM_heave_fast_ops/heave_ops_s42/model_best.pth",
     _REPO / "reports/eval_heave_ops_s42.json"),
    ("bathy", "config/argo/config_argo_heave_fast_bathy.json",
     "saved/models/NeSPReSO2_ARGO_GoM_heave_fast_bathy/heave_bathy_s42/model_best.pth",
     _REPO / "reports/eval_heave_bathy_s42.json"),
    ("bathy_wind", "config/argo/config_argo_heave_fast_bathy_wind.json",
     "saved/models/NeSPReSO2_ARGO_GoM_heave_fast_bathy_wind/heave_bathy_wind_s42/model_best.pth",
     _REPO / "reports/eval_heave_bathy_wind_s42.json"),
)


def main() -> int:
    phys = {}
    evals = {}
    for label, cfg_rel, ckpt_rel, out_json in RUNS:
        cfg_path = _ROOT / cfg_rel
        ckpt = _ROOT / ckpt_rel
        cfg = read_json(str(cfg_path))
        validate_config(cfg)
        config = ConfigParser(cfg, run_id="")
        report = eval_main(config, str(ckpt), split="test")
        write_report(report, out_json)
        evals[label] = report
        bundle, err = _try_real_bundle(cfg_path)
        if bundle is None:
            phys[label] = {"error": err}
            continue
        pred = _load_ckpt_pred(ckpt, bundle)
        if pred is None:
            phys[label] = {"error": "decode failed"}
            continue
        sla = bundle["sla"]
        scored = _score_pair(
            pred[0], pred[1], bundle["T_true"], bundle["S_true"],
            bundle["z"], bundle["lat"], bundle["lon"], sla, label,
        )
        stab = (scored.get("evalphys") or {}).get("static_stability_pred") or {}
        if isinstance(stab, dict) and "violation_rate_profile" not in stab:
            stab = stab.get("0") or (next(iter(stab.values())) if stab else {})
        phys[label] = {
            "D26_rmse": scored.get("D26_rmse"),
            "mld_rmse": scored.get("mld_rmse"),
            "ts_rmse": scored.get("ts_rmse"),
            "evalphys": {
                "static_stability_pred": {
                    "violation_rate_profile": stab.get("violation_rate_profile") if isinstance(stab, dict) else None,
                    "violation_rate_level": stab.get("violation_rate_level") if isinstance(stab, dict) else None,
                }
            },
        }

    a_json = _REPO / "reports/thermocline_scorecard.json"
    a_phys = {}
    if a_json.is_file():
        a_phys = json.loads(a_json.read_text()).get("models", {}).get("A_CRPS") or {}

    lines = [
        "# HeaveFast ablations vs A×CRPS and HeaveFast s42",
        "",
        "Chronological test, n=623, native ARGO z. Each ablation paired with its own cache.",
        "Conv is 3×3 at 1°, T=t−2…t0, local SST/SSS/SSH in `n_enc`. Point extras are operators, GEBCO bathy, NBS wind.",
        "",
        "## Train",
        "",
        "| run | epochs | wall | s/epoch | best val |",
        "|---|---:|---:|---:|---:|",
        "| HeaveFast s42 | 2091 | 130.5 min | 3.745 | — |",
        "| conv3 | 836 | 20.9 min | 1.501 | 3.762 |",
        "| ops | 714 | 18.0 min | 1.514 | 3.678 |",
        "| bathy | 704 | 16.8 min | 1.431 | 3.865 |",
        "| bathy+wind | 698 | 25.0 min | 2.148 | 3.717 |",
        "",
        "## Test RMSE (`eval_run` native z)",
        "",
        "| run | T RMSE | S RMSE |",
        "|---|---:|---:|",
        f"| A_CRPS | 0.562 | **0.091** |",
    ]
    t_best = "HeaveFast"
    s_best = "A_CRPS"
    rows = [("HeaveFast", evals["HeaveFast"]["raw_profile_rmse"])]
    for k in ("conv3", "ops", "bathy", "bathy_wind"):
        rows.append((k, evals[k]["raw_profile_rmse"]))
    t_vals = {k: v["temperature"] for k, v in rows}
    s_vals = {k: v["salinity"] for k, v in rows}
    t_min = min(t_vals.values())
    s_min = min(list(s_vals.values()) + [0.091])
    def fmt(name, t, s):
        ts = f"**{t:.3f}**" if abs(t - t_min) < 1e-9 else f"{t:.3f}"
        ss = f"**{s:.3f}**" if abs(s - s_min) < 1e-9 else f"{s:.3f}"
        return f"| {name} | {ts} | {ss} |"
    lines.append(fmt("HeaveFast", t_vals["HeaveFast"], s_vals["HeaveFast"]))
    for k, lab in (("conv3", "conv 3×3@1°"), ("ops", "current+ops"), ("bathy", "current+bathy"), ("bathy_wind", "current+bathy+wind")):
        lines.append(fmt(lab, t_vals[k], s_vals[k]))
    lines += ["", "## Thermocline", ""]
    def row(name, p):
        if not p or "D26_rmse" not in p:
            return f"- **{name}**: decode failed ({p})"
        ts = p.get("ts_rmse", {}).get("T", {})
        stab = ((p.get("evalphys") or {}).get("static_stability_pred") or {})
        if isinstance(stab, dict) and "violation_rate_profile" not in stab:
            stab = stab.get("0") or (next(iter(stab.values())) if stab else {})
        n2p = stab.get("violation_rate_profile") if isinstance(stab, dict) else None
        n2l = stab.get("violation_rate_level") if isinstance(stab, dict) else None
        n2s = ""
        if n2p is not None:
            n2s = f"; N² profile {n2p:.2f}, level {n2l:.4f}" if n2l is not None else f"; N² profile {n2p:.2f}"
        return (
            f"- **{name}**: D26 {p['D26_rmse']:.2f} m; MLD {p['mld_rmse']:.1f} m; "
            f"T 0–50/50–200/200–800 {ts.get('0-50', float('nan')):.3f}/"
            f"{ts.get('50-200', float('nan')):.3f}/{ts.get('200-800', float('nan')):.3f}{n2s}"
        )
    if a_phys:
        lines.append(row("A_CRPS", a_phys))
    for k, lab in (("HeaveFast", "HeaveFast"), ("conv3", "conv 3×3@1°"), ("ops", "current+ops"), ("bathy", "current+bathy"), ("bathy_wind", "current+bathy+wind")):
        lines.append(row(lab, phys.get(k) or {}))
    md = _REPO / "reports/heave_ablation_compare.md"
    md.write_text("\n".join(lines) + "\n")
    payload = {"eval": {k: v for k, v in evals.items()}, "physical": phys}
    (_REPO / "reports/heave_ablation_compare.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(md.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
