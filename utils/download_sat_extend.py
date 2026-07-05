#!/usr/bin/env python3
"""Extend local satellite archives through a target date (default 2026-05-31).

Phase 1 — Argo period (2015-01-01 .. 2022-03-01): audit only; downloads only if gaps
are found (excluding the whitelisted SSS 2022-03-01 endpoint).

Phase 2 — Extension (2022-03-02 .. target): OSTIA + SSS daily + SSH yearly files.

Usage:
  python download_sat_extend.py                 # audit argo, then extend to 2026-05-31
  python download_sat_extend.py --audit-only
  python download_sat_extend.py --extend-only --end 2026-05-31
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable

ARGO_START = "2015-01-01"
ARGO_END = "2022-03-01"
EXTEND_START = "2022-03-02"
DEFAULT_END = "2026-05-31"


def _run(cmd: list[str], log: Path | None = None) -> int:
    print("RUN", " ".join(cmd), flush=True)
    if log:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a") as fh:
            fh.write(f"\n=== {' '.join(cmd)} ===\n")
            fh.flush()
            return subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT)
    return subprocess.call(cmd)


def audit(argo_end: str, extend_end: str, json_out: Path) -> int:
    return _run(
        [
            PY,
            str(HERE / "audit_sat_coverage.py"),
            "--argo-end",
            argo_end,
            "--extend-end",
            extend_end,
            "--json-out",
            str(json_out),
        ]
    )


def extend_ostia(start: str, end: str, log: Path) -> int:
    return _run([PY, str(HERE / "download_OSTIA_range.py"), start, end], log=log)


def extend_sss(start: str, end: str, log: Path) -> int:
    return _run([PY, str(HERE / "download_SSS_range.py"), start, end], log=log)


def extend_ssh(end: str, log: Path) -> int:
    end_d = date.fromisoformat(end)
    years = list(range(2023, end_d.year))
    rc = 0
    if years:
        rc |= _run([PY, str(HERE / "download_SSH.py"), *[str(y) for y in years]], log=log)
    if end_d.year >= 2023:
        rc |= _run([PY, str(HERE / "download_SSH.py"), str(end_d.year)], log=log)
    return rc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--extend-only", action="store_true")
    parser.add_argument("--argo-end", default=ARGO_END)
    parser.add_argument("--start", default=EXTEND_START, help="extension start date")
    parser.add_argument("--end", default=DEFAULT_END, help="extension end date")
    parser.add_argument("--log-dir", type=Path, default=Path("/unity/g2/jmiranda/SubsurfaceFields/Data/logs/sat_extend"))
    args = parser.parse_args()

    report = args.log_dir / "coverage_audit.json"
    rc = 0

    if not args.extend_only:
        rc |= audit(args.argo_end, args.end, report)

    if args.audit_only:
        return rc

    args.log_dir.mkdir(parents=True, exist_ok=True)
    rc |= extend_ostia(args.start, args.end, args.log_dir / "ostia_extend.log")
    rc |= extend_sss(args.start, args.end, args.log_dir / "sss_extend.log")
    rc |= extend_ssh(args.end, args.log_dir / "ssh_extend.log")
    rc |= audit(args.argo_end, args.end, report)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
