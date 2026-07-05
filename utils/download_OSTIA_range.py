#!/usr/bin/env python3
"""Download OSTIA daily files for a date range into the shared OSTIA archive.

Uses PO.DAAC REP (OSTIA-UKMO-L4-GLOB-REP-v2.0) through 2023-12-31 and NRT
(OSTIA-UKMO-L4-GLOB-v2.0) from 2024-01-01 onward. Skips files already present.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

OSTIA_DIR = Path("/unity/g2/jmiranda/SubsurfaceFields/Data/OISST/OSTIA")
REP_COLLECTION = "OSTIA-UKMO-L4-GLOB-REP-v2.0"
NRT_COLLECTION = "OSTIA-UKMO-L4-GLOB-v2.0"
REP_THROUGH = date(2023, 12, 31)
DOWNLOADER = "podaac-data-downloader"


def _have_day(d: date, root: Path) -> bool:
    prefix = d.strftime("%Y%m%d")
    return any(p.name.startswith(prefix) for p in root.glob(f"{prefix}*.nc"))


def _missing_days(start: date, end: date, root: Path) -> list[date]:
    out: list[date] = []
    d = start
    while d <= end:
        if not _have_day(d, root):
            out.append(d)
        d += timedelta(days=1)
    return out


def _contiguous_ranges(days: list[date]) -> list[tuple[date, date]]:
    if not days:
        return []
    days = sorted(days)
    ranges: list[tuple[date, date]] = []
    lo = hi = days[0]
    for d in days[1:]:
        if d == hi + timedelta(days=1):
            hi = d
            continue
        ranges.append((lo, hi))
        lo = hi = d
    ranges.append((lo, hi))
    return ranges


def _collection_for_range(start: date, end: date) -> str:
    if end <= REP_THROUGH:
        return REP_COLLECTION
    if start > REP_THROUGH:
        return NRT_COLLECTION
    raise ValueError(f"range {start}..{end} crosses REP/NRT boundary; split first")


def _download_chunk(start: date, end: date, root: Path, dry_run: bool) -> int:
    collection = _collection_for_range(start, end)
    cmd = [
        DOWNLOADER,
        "-c",
        collection,
        "-d",
        str(root),
        "--start-date",
        f"{start.isoformat()}T00:00:00Z",
        "--end-date",
        f"{end.isoformat()}T00:00:00Z",
    ]
    if dry_run:
        cmd.append("--dry-run")
    print(f"RUN {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("start", nargs="?", default="2022-03-02")
    parser.add_argument("end", nargs="?", default="2026-05-31")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    root = OSTIA_DIR
    root.mkdir(parents=True, exist_ok=True)

    missing = _missing_days(start, end, root)
    print(f"OSTIA missing {len(missing)} days in {start}..{end}", flush=True)
    if not missing:
        return 0

    rc = 0
    for lo, hi in _contiguous_ranges(missing):
        # Split at REP/NRT boundary if needed
        if lo <= REP_THROUGH < hi:
            if lo <= REP_THROUGH:
                rc |= _download_chunk(lo, REP_THROUGH, root, args.dry_run)
            rc |= _download_chunk(REP_THROUGH + timedelta(days=1), hi, root, args.dry_run)
        else:
            rc |= _download_chunk(lo, hi, root, args.dry_run)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
