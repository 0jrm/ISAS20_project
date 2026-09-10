#!/usr/bin/env python3
"""Write the A2 opponent table into the figure directory. No new TSIS cycle."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from evalphys.opponent_report import format_a2_report, write_a2_report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cast", required=True)
    p.add_argument("--levels", required=True)
    p.add_argument("--perm-z", required=True)
    p.add_argument("--woa-long", required=True)
    p.add_argument("--woa-feat", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    stem = Path(args.cast).stem
    text = format_a2_report(
        pd.read_parquet(args.cast),
        pd.read_parquet(args.levels),
        pd.read_parquet(args.perm_z),
        pd.read_csv(args.woa_long),
        pd.read_parquet(args.woa_feat),
        stem,
    )
    dest = write_a2_report(text, Path(args.out))
    print(f"wrote {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
