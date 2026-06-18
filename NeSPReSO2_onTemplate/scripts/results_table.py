#!/usr/bin/env python3
"""Aggregate saved eval JSON into a reproducible results table (Phase 6.6)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ponytail: label map for known eval filenames — extend when new runs land.
EVAL_LABELS: dict[str, str] = {
    "eval_isas_patch16_pca_test": "ISAS patch16 PCA (prod)",
    "eval_patch16_test": "ISAS patch16 PCA",
    "eval_baseline15_test": "ISAS point 15-PC",
    "eval_argo16_test": "ARGO 16-PC (prod)",
    "eval_isas_decoder16_test": "ISAS decoder16_ae",
    "eval_isas_decoder32_test": "ISAS decoder32_ae",
    "eval_isas_decoder32_res_test": "ISAS decoder32_res_ae (immature)",
}


def _load_eval(path: Path) -> dict:
    text = path.read_text()
    # stdlib json accepts NaN; keep for decoder loss rows.
    return json.loads(text)


def _model_label(path: Path, data: dict) -> str:
    stem = path.stem
    if stem in EVAL_LABELS:
        return EVAL_LABELS[stem]
    ckpt = str(data.get("checkpoint", ""))
    if "decoder" in ckpt:
        return Path(ckpt).parent.name
    if "argo" in ckpt.lower():
        return "ARGO"
    if "patch" in ckpt.lower():
        return "ISAS patch"
    return stem


def collect_eval_rows(saved_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(saved_dir.glob("eval_*.json")):
        data = _load_eval(path)
        rmse = data.get("raw_profile_rmse") or {}
        rows.append(
            {
                "label": _model_label(path, data),
                "file": path.name,
                "dataset_tag": data.get("dataset_tag"),
                "split": data.get("split"),
                "n_samples": data.get("n_samples"),
                "temp_rmse": rmse.get("temperature"),
                "sal_rmse": rmse.get("salinity"),
                "loss": data.get("loss"),
                "checkpoint": data.get("checkpoint"),
            }
        )
    return rows


def _pct_delta(val: float | None, base: float | None) -> str:
    if val is None or base is None or base == 0:
        return "—"
    return f"{100.0 * (val - base) / base:+.1f}%"


def build_table(rows: list[dict]) -> dict:
    prod = next((r for r in rows if "prod" in r["label"].lower() and r["dataset_tag"] == "isas20"), None)
    base_t = prod["temp_rmse"] if prod else None
    base_s = prod["sal_rmse"] if prod else None

    table_rows = []
    for r in rows:
        table_rows.append(
            {
                **r,
                "vs_prod_temp": _pct_delta(r["temp_rmse"], base_t),
                "vs_prod_sal": _pct_delta(r["sal_rmse"], base_s),
            }
        )

    verdict = {
        "phase5_go_no_go_1": "FAILED — no decoder run within 5% of PCA prod on test raw_profile_rmse",
        "phase5_go_no_go_2": "not demonstrated — ISAS decoder ~3% slower than PCA at dim16",
        "production": "ISAS patch16_scales PCA-16",
    }
    if prod:
        verdict["production_rmse"] = {"temperature": prod["temp_rmse"], "salinity": prod["sal_rmse"]}

    return {"rows": table_rows, "verdict": verdict, "n_files": len(rows)}


def to_markdown(report: dict) -> str:
    lines = [
        "# NeSPReSO eval results (test split, within-tag)",
        "",
        "| Model | T RMSE | S RMSE | vs prod T | vs prod S | Notes |",
        "|-------|-------:|-------:|----------:|----------:|-------|",
    ]
    for r in report["rows"]:
        t = r["temp_rmse"]
        s = r["sal_rmse"]
        t_s = f"{t:.3f}" if isinstance(t, (int, float)) else "—"
        s_s = f"{s:.3f}" if isinstance(s, (int, float)) else "—"
        note = r["file"]
        if r["loss"] is None or (isinstance(r["loss"], float) and r["loss"] != r["loss"]):
            note += "; loss NaN"
        lines.append(
            f"| {r['label']} | {t_s} | {s_s} | {r['vs_prod_temp']} | {r['vs_prod_sal']} | {note} |"
        )
    lines.extend(
        [
            "",
            "## Phase 5 verdict",
            "",
            f"- Go/no-go #1: {report['verdict']['phase5_go_no_go_1']}",
            f"- Go/no-go #2: {report['verdict']['phase5_go_no_go_2']}",
            f"- Production: {report['verdict']['production']}",
        ]
    )
    if "production_rmse" in report["verdict"]:
        pr = report["verdict"]["production_rmse"]
        lines.append(f"- Production test RMSE: T **{pr['temperature']:.3f}** / S **{pr['salinity']:.3f}**")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saved-dir", default="saved", help="Directory with eval_*.json")
    parser.add_argument("--out-dir", default="saved/results", help="Output directory")
    args = parser.parse_args()

    saved_dir = Path(args.saved_dir)
    if not saved_dir.is_absolute():
        saved_dir = ROOT / saved_dir
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_eval_rows(saved_dir)
    report = build_table(rows)
    (out_dir / "eval_table.json").write_text(json.dumps(report, indent=2) + "\n")
    (out_dir / "eval_table.md").write_text(to_markdown(report))
    print(to_markdown(report))
    print(f"Wrote {out_dir / 'eval_table.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
