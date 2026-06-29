#!/usr/bin/env python3
"""Idempotent L3/L4 surface product download scaffolding (Phase 3).

Records manifest entries under data/manifests/download_manifest.jsonl.
Requires external credentials: ``copernicusmarine login``, CDS API for ERA5.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_REPO = _ROOT.parent

PRODUCTS = {
    "ssh_l3_historical": {
        "tool": "copernicusmarine",
        "product_id": "SEALEVEL_GLO_PHY_L3_MY_008_062",
        "output_subdir": "altimetry_l3",
        "description": "Copernicus L3 along-track SSH (historical)",
    },
    "ssh_l3_nrt": {
        "tool": "copernicusmarine",
        "product_id": "SEALEVEL_GLO_PHY_L3_NRT_008_044",
        "output_subdir": "altimetry_l3_nrt",
        "description": "Copernicus L3 along-track SSH (NRT, 2022+)",
    },
    "ssh_l4_aux": {
        "tool": "copernicusmarine",
        "product_id": "SEALEVEL_GLO_PHY_L4_MY_008_047",
        "output_subdir": "altimetry_l4_aux",
        "description": "DUACS L4 SSH for augmentation/baseline only",
    },
    "sst_viirs_n20": {
        "tool": "podaac",
        "collection": "VIIRS_N20-STAR-L3U-v2.80",
        "output_subdir": "sst_viirs_n20_l3u",
        "description": "GHRSST VIIRS N20 L3U SST",
    },
    "sst_viirs_npp": {
        "tool": "podaac",
        "collection": "VIIRS_NPP-STAR-L3U-v2.80",
        "output_subdir": "sst_viirs_npp_l3u",
        "description": "GHRSST VIIRS NPP L3U SST",
    },
    "sss_smap_8day": {
        "tool": "podaac",
        "collection": "SMAP_RSS_L3_SSS_SMI_8DAY-RUNNINGMEAN_V5",
        "output_subdir": "smap_sss_l3_8day",
        "description": "RSS SMAP L3 8-day SSS (optional)",
    },
    "swot_l2_basic": {
        "tool": "podaac",
        "collection": "SWOT_L2_LR_SSH_BASIC_2.0",
        "output_subdir": "swot_l2_basic",
        "description": "SWOT L2 SSH (2023+ validation extension)",
    },
}

# ponytail: GoM default bbox [N, W, S, E]
DEFAULT_BBOX = (35.0, -100.0, 15.0, -75.0)


def _manifest_path(data_root: Path) -> Path:
    p = data_root / "manifests" / "download_manifest.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _append_manifest(manifest: Path, entry: dict) -> None:
    entry.setdefault("timestamp", datetime.utcnow().isoformat() + "Z")
    with open(manifest, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")


def _download_copernicus_l3(
    product_id: str,
    out_dir: Path,
    day: date,
    *,
    force: bool,
) -> dict:
    import copernicusmarine

    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = [
        d.dataset_id
        for d in copernicusmarine.describe(product_id=product_id).products[0].datasets
    ]
    date_filter = f"*/{day.strftime('%Y/%m')}/*_{day.strftime('%Y%m%d')}*.nc"
    downloaded = []
    for dataset_id in datasets:
        copernicusmarine.get(
            dataset_id=dataset_id,
            filter=date_filter,
            output_directory=str(out_dir),
            skip_existing=not force,
        )
        downloaded.append(dataset_id)
    return {
        "product_id": product_id,
        "date": day.isoformat(),
        "datasets": downloaded,
        "output_directory": str(out_dir),
        "filter": date_filter,
    }


def _download_podaac(
    collection: str,
    out_dir: Path,
    start: str,
    end: str,
    bbox: tuple[float, float, float, float],
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    n, w, s, e = bbox
    cmd = [
        "podaac-data-subscriber",
        "-c",
        collection,
        "-d",
        str(out_dir),
        "--start-date",
        start,
        "--end-date",
        end,
        "-b",
        f"{w},{s},{e},{n}",
    ]
    subprocess.run(cmd, check=True)
    return {
        "collection": collection,
        "command": cmd,
        "output_directory": str(out_dir),
        "bbox": list(bbox),
        "start": start,
        "end": end,
    }


def _download_era5_month(
    out_dir: Path,
    year: int,
    month: int,
    bbox: tuple[float, float, float, float],
) -> dict:
    import cdsapi

    out_dir.mkdir(parents=True, exist_ok=True)
    n, w, s, e = bbox
    out_file = out_dir / f"era5_u10_v10_{year}{month:02d}_gom.nc"
    if out_file.is_file():
        return {"skipped": True, "path": str(out_file)}
    client = cdsapi.Client()
    client.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": ["reanalysis"],
            "variable": ["10m_u_component_of_wind", "10m_v_component_of_wind"],
            "year": [str(year)],
            "month": [f"{month:02d}"],
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": [f"{h:02d}:00" for h in range(24)],
            "area": [n, w, s, e],
            "format": "netcdf",
        },
        str(out_file),
    )
    return {"path": str(out_file), "year": year, "month": month, "bbox": list(bbox)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download L3/L4 surface products")
    parser.add_argument(
        "--product",
        choices=list(PRODUCTS.keys()) + ["era5_wind", "all_scaffold"],
        required=True,
        help="product key to download",
    )
    parser.add_argument("--data-root", default="../data/raw", help="raw data root")
    parser.add_argument("--date", help="YYYY-MM-DD (Copernicus daily filter)")
    parser.add_argument("--start", help="ISO start (PO.DAAC subscriber)")
    parser.add_argument("--end", help="ISO end (PO.DAAC subscriber)")
    parser.add_argument("--year", type=int, help="ERA5 year")
    parser.add_argument("--month", type=int, help="ERA5 month")
    parser.add_argument(
        "--bbox",
        default=",".join(str(x) for x in DEFAULT_BBOX),
        help="N,W,S,E bounding box",
    )
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    args = parser.parse_args(argv)

    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = (_ROOT / data_root).resolve()
    bbox = tuple(float(x) for x in args.bbox.split(","))
    manifest = _manifest_path(_REPO / "data")

    if args.product == "all_scaffold":
        print("Available products:")
        for k, v in PRODUCTS.items():
            print(f"  {k}: {v['description']}")
        print("  era5_wind: ERA5 hourly u10/v10 (requires cdsapi + ~/.cdsapirc)")
        return 0

    if args.product == "era5_wind":
        if args.year is None or args.month is None:
            parser.error("era5_wind requires --year and --month")
        out_dir = data_root / "era5"
        result = _download_era5_month(out_dir, args.year, args.month, bbox)
        _append_manifest(
            manifest,
            {"product": "era5_wind", "tool": "cdsapi", **result},
        )
        print(json.dumps(result, indent=2))
        return 0

    spec = PRODUCTS[args.product]
    out_dir = data_root / spec["output_subdir"]

    if spec["tool"] == "copernicusmarine":
        if not args.date:
            parser.error(f"{args.product} requires --date YYYY-MM-DD")
        day = date.fromisoformat(args.date)
        result = _download_copernicus_l3(
            spec["product_id"], out_dir, day, force=args.force
        )
        _append_manifest(
            manifest,
            {"product": args.product, "tool": "copernicusmarine", **result},
        )
    else:
        if not args.start or not args.end:
            parser.error(f"{args.product} requires --start and --end ISO timestamps")
        result = _download_podaac(
            spec["collection"], out_dir, args.start, args.end, bbox
        )
        _append_manifest(
            manifest,
            {"product": args.product, "tool": "podaac", **result},
        )

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
