#!/usr/bin/env python3
"""Audit local OSTIA / CMEMS SSS / CMEMS SSH coverage against target date ranges."""

from __future__ import annotations

import argparse
import json
import re
from datetime import date, timedelta
from pathlib import Path

OSTIA_DIR = Path("/unity/g2/jmiranda/SubsurfaceFields/Data/OISST/OSTIA")
SSS_DIR = Path("/unity/g2/jmiranda/SubsurfaceFields/Data/CMEMS/SSS")
SSH_DIR = Path("/unity/g2/jmiranda/SubsurfaceFields/Data/CMEMS/SSH")

# Argo export + 7-day tendency pad (matches cube_schema TIME_END slack)
ARGO_START = date(2015, 1, 1)
ARGO_END = date(2022, 3, 1)
EXTEND_END = date(2026, 5, 31)


def _parse_date(s: str) -> date:
    return date.fromisoformat(s)


def ostia_dates(root: Path = OSTIA_DIR) -> set[date]:
    pat = re.compile(r"^(\d{8})")
    out: set[date] = set()
    for path in root.glob("*.nc"):
        m = pat.match(path.name)
        if not m:
            continue
        s = m.group(1)
        out.add(date(int(s[:4]), int(s[4:6]), int(s[6:8])))
    return out


def sss_dates(root: Path = SSS_DIR) -> set[date]:
    pat = re.compile(r"SSS_(\d{8})\.nc")
    out: set[date] = set()
    for path in root.glob("SSS_*.nc"):
        m = pat.match(path.name)
        if not m:
            continue
        s = m.group(1)
        out.add(date(int(s[:4]), int(s[4:6]), int(s[6:8])))
    return out


def ssh_years(root: Path = SSH_DIR) -> set[int]:
    pat = re.compile(r"SSH_(\d{4})\.nc")
    return {int(m.group(1)) for path in root.glob("SSH_*.nc") if (m := pat.match(path.name))}


def daily_missing(have: set[date], start: date, end: date) -> list[str]:
    missing: list[str] = []
    d = start
    while d <= end:
        if d not in have:
            missing.append(d.isoformat())
        d += timedelta(days=1)
    return missing


def audit_range(label: str, start: date, end: date, ostia: set[date], sss: set[date], ssh: set[int]) -> dict:
    missing_o = daily_missing(ostia, start, end)
    missing_s = daily_missing(sss, start, end)
    need_ssh = list(range(start.year, end.year + 1))
    missing_ssh = [y for y in need_ssh if y not in ssh]
    return {
        "label": label,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "ostia": {"missing_count": len(missing_o), "missing_first": missing_o[:5], "missing_last": missing_o[-5:]},
        "sss": {"missing_count": len(missing_s), "missing_first": missing_s[:5], "missing_last": missing_s[-5:]},
        "ssh": {"missing_years": missing_ssh},
        "complete": len(missing_o) == 0 and len(missing_s) == 0 and len(missing_ssh) == 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--argo-end", default=ARGO_END.isoformat())
    parser.add_argument("--extend-end", default=EXTEND_END.isoformat())
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    ostia = ostia_dates()
    sss = sss_dates()
    ssh = ssh_years()

    argo_end = _parse_date(args.argo_end)
    extend_end = _parse_date(args.extend_end)
    extend_start = argo_end + timedelta(days=1)

    report = {
        "argo_period": audit_range("argo", ARGO_START, argo_end, ostia, sss, ssh),
        "extension": audit_range("extend", extend_start, extend_end, ostia, sss, ssh),
        "notes": {
            "argo_sss_whitelist": "2022-03-01 is intentionally missing (CMEMS SSS archive gap); cube ALLOWED_MISSING_DAYS",
        },
    }

    text = json.dumps(report, indent=2)
    print(text)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n")

    argo_ok = report["argo_period"]["ostia"]["missing_count"] == 0 and (
        report["argo_period"]["sss"]["missing_count"] <= 1
    ) and report["argo_period"]["ssh"]["missing_years"] == []
    extend_ok = report["extension"]["complete"]
    return 0 if argo_ok and extend_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
