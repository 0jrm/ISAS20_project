#!/usr/bin/env python3
"""Write gitignored cast/level parquet and the permutation cache from profiles_xb/nes."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evalphys.cast_table import (
    RULE_VERSION,
    build_cast_and_level_tables,
    git_short_hash,
    load_profile_bundle,
    parquet_stem,
)
from evalphys.permutation_null import PERM_N_DEFAULT, permutation_cache


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--xb",
        default="/home/jrm22n/ISAS20_project/reports/xb_argo_compare/profiles_xb.nc",
    )
    p.add_argument(
        "--nes",
        default="/home/jrm22n/ISAS20_project/reports/xb_argo_compare/profiles_nes.nc",
    )
    p.add_argument("--out-dir", default="/home/jrm22n/work/gom-da-workspace/results/da_readiness")
    p.add_argument("--models", nargs="*", default=["A_CRPS", "persistence"])
    p.add_argument("--perm-n", type=int, default=PERM_N_DEFAULT)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--with-mld", action="store_true")
    args = p.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_profile_bundle(Path(args.xb), Path(args.nes))
    if args.limit:
        n = int(args.limit)
        bundle = replace(
            bundle,
            lon=bundle.lon[:n],
            lat=bundle.lat[:n],
            time_iso=bundle.time_iso[:n],
            analysis_date=bundle.analysis_date[:n],
            t_argo=bundle.t_argo[:n],
            t_xb=bundle.t_xb[:n],
            s_argo=bundle.s_argo[:n],
            s_xb=bundle.s_xb[:n],
            valid_t=bundle.valid_t[:n],
            ssh_sat=bundle.ssh_sat[:n],
            products={k: v[:n] for k, v in bundle.products.items()},
        )
    casts, levels = build_cast_and_level_tables(
        bundle, models=args.models, with_mld=args.with_mld
    )
    repo = _ROOT.parent
    stem = parquet_stem(git_short_hash(repo), RULE_VERSION)
    cast_path = out_dir / f"{stem}.parquet"
    level_path = out_dir / f"{stem}_levels.parquet"
    perm_path = out_dir / f"{stem}_perm.parquet"
    casts.to_parquet(cast_path, index=False)
    levels.to_parquet(level_path, index=False)
    perm_casts = casts.loc[casts["model"] != "persistence"]
    perm_levels = levels.loc[levels["model"] != "persistence"]
    perm = permutation_cache(perm_casts, perm_levels, n_perm=int(args.perm_n), seed=0)
    perm.to_parquet(perm_path, index=False)
    print(f"wrote {cast_path}", flush=True)
    print(f"wrote {level_path}", flush=True)
    print(f"wrote {perm_path}  cells={len(perm)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
