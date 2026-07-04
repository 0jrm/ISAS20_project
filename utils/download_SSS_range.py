#!/usr/bin/env python3
"""Download daily CMEMS SSS files for a date range (fills the post-2020 gap).

Usage: python download_SSS_range.py [START END]   (dates as YYYY-MM-DD)
Defaults to 2021-01-01 .. 2022-02-28, the stale window found 2026-07-03.
Skips files already on disk, so it is safe to re-run after interruption.
"""
import os
import sys
from datetime import date, timedelta

import copernicusmarine

OUTPUT_DIR = "/unity/g2/jmiranda/SubsurfaceFields/Data/CMEMS/SSS"
DATASET_ID = "cmems_obs-mob_glo_phy-sss_my_multi_P1D"
DATASET_VERSION = "202311"


def download_day(d: date) -> bool:
    fname = f"SSS_{d.strftime('%Y%m%d')}.nc"
    if os.path.isfile(os.path.join(OUTPUT_DIR, fname)):
        print(f"skip existing {fname}")
        return True
    ds = d.strftime("%Y-%m-%d")
    try:
        copernicusmarine.subset(
            dataset_id=DATASET_ID,
            dataset_version=DATASET_VERSION,
            minimum_longitude=-179.9375,
            maximum_longitude=179.9375,
            minimum_latitude=-89.9375,
            maximum_latitude=89.9375,
            start_datetime=f"{ds}T00:00:00",
            end_datetime=f"{ds}T00:00:00",
            minimum_depth=0,
            maximum_depth=0,
            disable_progress_bar=True,
            output_directory=OUTPUT_DIR,
            output_filename=fname,
        )
        print(f"downloaded {fname}")
        return True
    except Exception as e:
        print(f"FAILED {fname}: {e}")
        return False


def main():
    start = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 2 else date(2021, 1, 1)
    end = date.fromisoformat(sys.argv[2]) if len(sys.argv) > 2 else date(2022, 2, 28)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ok = fail = 0
    d = start
    while d <= end:
        if download_day(d):
            ok += 1
        else:
            fail += 1
        d += timedelta(days=1)
    print(f"\nSummary: {ok} ok, {fail} failed ({start}..{end})")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
