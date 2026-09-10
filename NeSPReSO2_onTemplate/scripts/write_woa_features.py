#!/usr/bin/env python3
"""Turn a long WOA (cast_id, z, t_woa) table into one-row-per-cast shape features."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from evalphys.cast_table import woa_feature_table


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--levels", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    src = Path(args.levels)
    if src.suffix == ".parquet":
        long_df = pd.read_parquet(src)
    else:
        long_df = pd.read_csv(src)
    feat = woa_feature_table(long_df)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    feat.to_parquet(out, index=False)
    print(f"wrote {out} n={len(feat)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
