#!/usr/bin/env python3
"""Wait for three runs, then eval + scorecard + timing table. Stdlib poll."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_REPO = _ROOT.parent
PY = sys.executable

RUNS = [
    {
        "label": "Heave",
        "cfg": "config/argo/config_argo_heave_residual.json",
        "save": "saved/models/NeSPReSO2_ARGO_GoM_heave_residual/heave_residual_s42d",
        "out": _REPO / "reports" / "eval_heave_s42d.json",
        "mode": "heave",
    },
    {
        "label": "Latent",
        "cfg": "config/argo/config_argo_latent_profile.json",
        "save": "saved/models/NeSPReSO2_ARGO_GoM_latent_profile/latent_profile_s42",
        "out": _REPO / "reports" / "eval_latent_s42.json",
        "mode": "profile",
    },
    {
        "label": "Direct",
        "cfg": "config/argo/config_argo_profile_direct.json",
        "save": "saved/models/NeSPReSO2_ARGO_GoM_profile_direct/profile_direct_s42",
        "out": _REPO / "reports" / "eval_direct_s42.json",
        "mode": "profile",
    },
]


def _status(save: Path) -> dict | None:
    p = save / "status.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text())


def _timing(save: Path, st: dict | None) -> dict:
    out = {"epoch": None, "elapsed_sec": None, "sec_per_epoch": None, "state": None, "reason": None}
    if st:
        out["epoch"] = st.get("epoch")
        out["state"] = st.get("state")
        out["reason"] = st.get("reason")
        if st.get("elapsed_sec") is not None:
            out["elapsed_sec"] = float(st["elapsed_sec"])
            out["sec_per_epoch"] = float(st.get("sec_per_epoch") or (out["elapsed_sec"] / max(int(out["epoch"] or 1), 1)))
            return out
    start_p = save / "train_start.json"
    started = None
    if start_p.is_file():
        started = json.loads(start_p.read_text()).get("started_at")
    elif st and st.get("started_at"):
        started = st["started_at"]
    ended = (st or {}).get("updated_at")
    if started and ended:
        def _parse(s):
            s = str(s).replace("Z", "+00:00")
            return datetime.fromisoformat(s)

        elapsed = (_parse(ended) - _parse(started)).total_seconds()
        if elapsed > 0:
            out["elapsed_sec"] = elapsed
            out["sec_per_epoch"] = elapsed / max(int(out["epoch"] or 1), 1)
    return out


def _fmt_sec(s):
    if s is None:
        return "—"
    s = float(s)
    if s < 90:
        return f"{s:.1f} s"
    return f"{s / 60:.1f} min"


def wait_all(timeout_h=18, poll_s=60):
    t0 = time.time()
    while True:
        rows = []
        for r in RUNS:
            st = _status(_ROOT / r["save"])
            rows.append((r["label"], (st or {}).get("state"), (st or {}).get("epoch")))
        print("wait", rows, flush=True)
        if all(s in ("done", "failed") for _, s, _ in rows) and all(s is not None for _, s, _ in rows):
            return
        if time.time() - t0 > timeout_h * 3600:
            raise SystemExit("timeout waiting for runs")
        time.sleep(poll_s)


def _best(save: Path) -> Path:
    p = save / "model_best.pth"
    return p if p.is_file() else save / "checkpoint.pth"


def main() -> int:
    wait_all()
    md_rows = []
    ckpt_args = []
    for r in RUNS:
        save = _ROOT / r["save"]
        st = _status(save)
        ckpt = _best(save)
        out = r["out"]
        cmd = [PY, "eval_run.py", "-c", r["cfg"], "-r", str(ckpt), "--out", str(out)]
        print("RUN", " ".join(cmd), flush=True)
        subprocess.check_call(cmd, cwd=_ROOT)
        ev = json.loads(out.read_text())
        t = _timing(save, st)
        raw = ev.get("raw_profile_rmse") or {}
        md_rows.append(
            {
                "label": r["label"],
                "T": raw.get("temperature"),
                "S": raw.get("salinity"),
                **t,
                "ckpt": str(ckpt),
                "eval": ev,
            }
        )
        ckpt_args += ["--ckpt", f"{r['label']}={ckpt}"]
        if r["mode"] == "heave":
            npz = _REPO / "reports" / "heave_export_s42d.npz"
            subprocess.check_call(
                [PY, "scripts/export_heave_tsis.py", "-c", r["cfg"], "-r", str(ckpt), "--out", str(npz)],
                cwd=_ROOT,
            )
    sc_md = _REPO / "reports" / "thermocline_scorecard.md"
    sc_json = _REPO / "reports" / "thermocline_scorecard.json"
    subprocess.check_call(
        [
            PY, "scripts/thermocline_scorecard.py",
            "-c", "config/argo/config_argo_heave_residual.json",
            "--out-md", str(sc_md), "--out-json", str(sc_json),
            *ckpt_args,
        ],
        cwd=_ROOT,
    )
    sc = json.loads(sc_json.read_text())
    models = sc.get("models") or {}
    lines = [
        "# Three-way eval (same 11-d inputs, chrono test)",
        "",
        f"Wrote {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}. Cache `train_ready_3adcff404b0b.pkl`.",
        "",
        "Heave fix: exp MLD/D26 (raw=0 → 50/120 m), CRPS σ in metres, geom_scale=10, warp μ zero-init. Replaces s42b/s42c.",
        "",
        "## Training time",
        "",
        "| run | epochs | wall | s/epoch | state |",
        "|---|---:|---:|---:|---|",
    ]
    for row in md_rows:
        spe = row["sec_per_epoch"]
        spe_s = f"{spe:.3f}" if spe is not None else "—"
        lines.append(
            f"| {row['label']} | {row['epoch']} | {_fmt_sec(row['elapsed_sec'])} | "
            f"{spe_s} | {row['state']}/{row['reason']} |"
        )
    lines += ["", "## Test RMSE (eval_run native z)", "", "| run | T RMSE | S RMSE |", "|---|---:|---:|"]
    for row in md_rows:
        t = row["T"]
        s = row["S"]
        lines.append(f"| {row['label']} | {t:.3f} | {s:.3f} |" if t is not None else f"| {row['label']} | — | — |")
    lines += ["", "## Thermocline scorecard", ""]
    for name, rec in models.items():
        if "heave_vs_shape" not in rec:
            continue
        h = rec["heave_vs_shape"]
        ep = rec.get("evalphys") or {}
        t_bands = ((ep.get("ts_rmse") or {}).get("T")) or {}
        n2 = (ep.get("static_stability_pred") or {}).get("1e-08") or {}
        lines.append(
            f"- **{name}**: D26 {rec.get('D26_rmse')} m; MLD {rec.get('mld_rmse')} m; "
            f"T 0–50/50–200/200–800 {t_bands.get('0-50')}/{t_bands.get('50-200')}/{t_bands.get('200-800')}; "
            f"heave frac {h.get('heave_fraction')}; N² viol {n2.get('violation_rate_profile')}"
        )
    out_md = _REPO / "reports" / "heave_eval_compare.md"
    out_md.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
