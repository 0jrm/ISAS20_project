#!/usr/bin/env python3
"""Build mask-native L3 processed samples around ARGO targets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from parse_config import validate_config
from playground import read_json
from preproc.export_l3_cache import build_l3_processed_batch, export_l3_train_cache


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rasterize L3 SSH + ERA5 wind around ARGO profiles")
    parser.add_argument("-c", "--config", required=True, help="config JSON (e.g. config_argo_l3_smoke.json)")
    parser.add_argument("--max-samples", type=int, default=None, help="limit number of profiles")
    parser.add_argument("--anchor-date", default="2020-01-15", help="pick nearest profiles to this date")
    parser.add_argument("--force", action="store_true", help="rebuild even if output exists")
    parser.add_argument("--export-train-cache", action="store_true", help="also write train_ready_l3_*.pkl")
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    if not config_path.is_file():
        config_path = _ROOT / args.config
    cfg = read_json(config_path)
    validate_config(cfg)
    if not cfg.get("io", {}).get("l3", {}).get("enabled"):
        raise ValueError("io.l3.enabled must be true")

    processed = build_l3_processed_batch(
        cfg,
        max_samples=args.max_samples,
        anchor_date=args.anchor_date if args.max_samples else None,
        force=args.force,
    )
    if args.export_train_cache:
        export_l3_train_cache(cfg, processed, force=args.force)
    print(processed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
