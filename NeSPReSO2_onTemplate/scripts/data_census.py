#!/usr/bin/env python3
"""Phase 0: ARGO/profile data census and temporal split design reports."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from base.split_utils import (
    assign_chronological_fraction_indices,
    assign_chronological_indices,
    build_split_indices,
    split_summary,
)
from playground import read_json, write_json


def _git_commit() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_ROOT.parent,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _season(month: int) -> str:
    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    return "SON"


def _gom_region(lat: float, lon: float) -> str:
    """ponytail: coarse GoM shelf/slope/deep bins for census only."""
    if lat >= 27.0:
        return "north_gom"
    if lat >= 24.0:
        return "central_gom"
    if lat >= 21.0:
        return "southern_gom"
    return "caribbean_edge"


def _depth_coverage(profiles: dict, pres: np.ndarray | None) -> dict[str, int]:
    """Count profiles by fraction of valid (non-NaN) depth levels."""
    out: Counter[str] = Counter()
    for name, prof in profiles.items():
        if name not in ("temperature", "salinity"):
            continue
        arr = np.asarray(prof)
        if arr.ndim != 2:
            continue
        for col in arr.T:
            valid = np.isfinite(col).sum()
            frac = valid / max(1, arr.shape[0])
            if frac >= 0.9:
                out["ge90pct"] += 1
            elif frac >= 0.5:
                out["50_90pct"] += 1
            else:
                out["lt50pct"] += 1
        break  # ponytail: temp/sal share same mask in v2 caches
    return dict(out)


def load_argo_source(config_path: Path) -> dict:
    cfg = read_json(config_path)
    io = cfg["io"]
    tag = io.get("dataset_tag", "argo_v2")
    v2_src = io.get("v2_src")
    cache_dir = Path(io.get("cache_dir", "../data/cache"))

    # Prefer existing train cache (has JULD + profiles)
    from preproc.preproc_isas_sat import config_hash

    chash = config_hash(cfg)
    cache_path = cache_dir / f"train_ready_{chash}.pkl"
    if cache_path.is_file():
        import pickle

        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        return {"source": str(cache_path), "cache": cache, "config": cfg, "tag": tag, "v2_src": v2_src}

    pickle_path = io.get("v2_pickle")
    if not pickle_path or not Path(pickle_path).is_file():
        raise FileNotFoundError(f"No cache or v2_pickle found for {config_path}")

    if v2_src:
        sys.path.insert(0, str(v2_src))
    from nespreso.data.pickle_compat import load_dataset_pickle

    data = load_dataset_pickle(pickle_path)
    ds = data["full_dataset"]
    cache = {
        "LAT": np.asarray(ds.LAT, dtype=np.float32),
        "LON": np.asarray(ds.LON, dtype=np.float32),
        "JULD": np.asarray(ds.TIME, dtype=np.float32),
        "profiles": {"temperature": np.asarray(ds.TEMP), "salinity": np.asarray(ds.SAL)},
        "dataset_tag": tag,
        "min_depth": int(ds.min_depth),
        "max_depth": int(ds.max_depth),
    }
    return {"source": pickle_path, "cache": cache, "config": cfg, "tag": tag, "v2_src": v2_src}


def _l3_raw_coverage(config: dict) -> dict[str, Any]:
    """Inspect downloaded L3/ERA5 raw files when data/raw exists."""
    io = config.get("io", {})
    l3 = io.get("l3") or {}
    raw_root = Path(io.get("l3", {}).get("raw_root", "../data/raw"))
    if not raw_root.is_absolute():
        raw_root = (_ROOT / raw_root).resolve()
    if not raw_root.is_dir():
        return {
            "status": "missing_raw_root",
            "path": str(raw_root),
            "ssh_files": 0,
            "era5_files": 0,
        }
    ssh_files = sorted((raw_root / "altimetry_l3").glob("*.nc")) if (raw_root / "altimetry_l3").is_dir() else []
    era5_files = sorted((raw_root / "era5").glob("era5_u10_v10_*_gom.nc")) if (raw_root / "era5").is_dir() else []
    ssh_dates = sorted({p.stem[-8:] for p in ssh_files if len(p.stem) >= 8})
    era5_months = sorted({p.stem.split("_")[-2] for p in era5_files if "_" in p.stem})
    return {
        "status": "inspected",
        "path": str(raw_root),
        "ssh_files": len(ssh_files),
        "era5_files": len(era5_files),
        "ssh_date_span": [ssh_dates[0], ssh_dates[-1]] if ssh_dates else [],
        "era5_months": era5_months,
        "l3_enabled": bool(l3.get("enabled")),
    }


def build_census(payload: dict) -> dict:
    from base.split_utils import sample_dates

    cache = payload["cache"]
    tag = payload["tag"]
    v2_src = payload.get("v2_src")
    lat = np.asarray(cache["LAT"]).ravel()
    lon = np.asarray(cache["LON"]).ravel()
    juld = np.asarray(cache["JULD"]).ravel()
    n = len(juld)
    dates = sample_dates(juld, dataset_tag=tag, v2_src=v2_src)

    by_year: Counter[int] = Counter()
    by_month: Counter[str] = Counter()
    by_season: Counter[str] = Counter()
    by_region: Counter[str] = Counter()
    for i in range(n):
        d = date.fromisoformat(str(dates[i])[:10])
        by_year[d.year] += 1
        by_month[f"{d.year}-{d.month:02d}"] += 1
        by_season[_season(d.month)] += 1
        by_region[_gom_region(float(lat[i]), float(lon[i]))] += 1

    pres = cache.get("PRES")
    depth_cov = _depth_coverage(cache.get("profiles", {}), pres)

    census = {
        "n_profiles": n,
        "source": payload["source"],
        "dataset_tag": tag,
        "spatial_bounds": {
            "lat_min": float(lat.min()),
            "lat_max": float(lat.max()),
            "lon_min": float(lon.min()),
            "lon_max": float(lon.max()),
        },
        "depth_range_m": [int(cache.get("min_depth", 0)), int(cache.get("max_depth", 0))],
        "temporal_range": {"min": str(dates.min()), "max": str(dates.max())},
        "by_year": {str(k): v for k, v in sorted(by_year.items())},
        "by_month": dict(sorted(by_month.items())),
        "by_season": dict(by_season),
        "by_region": dict(by_region),
        "depth_coverage": depth_cov,
        "surface_inputs": {
            "note": "ARGO cache uses L4 gridded SST/SSH/SSS (v2 COAPS). L3 raw coverage below.",
            "l3_raw": _l3_raw_coverage(payload["config"]),
        },
        "git_commit": _git_commit(),
    }
    return census


def evaluate_split_candidates(census: dict, cache: dict, tag: str, v2_src: str | None) -> dict:
    juld = np.asarray(cache["JULD"])
    n = len(juld)

    candidates = {}

    # A: legacy 2002-2020 (expected empty for this GoM ARGO export)
    try:
        idx_a = assign_chronological_indices(
            juld,
            dataset_tag=tag,
            split_config={
                "train": {"start": "2002-01-01", "end": "2015-12-31"},
                "val": {"start": "2016-01-01", "end": "2017-12-31"},
                "test": {"start": "2018-01-01", "end": "2020-12-31"},
            },
            v2_src=v2_src,
        )
        candidates["A_simple_chronological_2002_2020"] = {
            "indices": idx_a,
            "summary": split_summary(idx_a, juld, dataset_tag=tag, v2_src=v2_src),
            "viable": sum(len(v) for v in idx_a.values()) > 0,
        }
    except Exception as exc:
        candidates["A_simple_chronological_2002_2020"] = {"error": str(exc), "viable": False}

    # E: common overlap era (actual ARGO span)
    idx_e = assign_chronological_indices(
        juld,
        dataset_tag=tag,
        split_config={
            "train": {"start": "2015-01-01", "end": "2019-12-31"},
            "val": {"start": "2020-01-01", "end": "2020-12-31"},
            "test": {"start": "2021-01-01", "end": "2021-12-31"},
        },
        v2_src=v2_src,
        unassigned="exclude",
    )
    candidates["E_common_overlap_2015_2021"] = {
        "indices": {k: len(v) for k, v in idx_e.items()},
        "summary": split_summary(idx_e, juld, dataset_tag=tag, v2_src=v2_src),
        "excluded_years": ["2022"],
        "viable": all(len(idx_e[s]) > 0 for s in ("train", "val", "test")),
    }

    # B: chronological 70/15/15 fractions
    idx_b = assign_chronological_fraction_indices(
        juld, dataset_tag=tag, train_frac=0.7, val_frac=0.15, test_frac=0.15, v2_src=v2_src
    )
    candidates["B_chronological_fraction_70_15_15"] = {
        "indices": {k: len(v) for k, v in idx_b.items()},
        "summary": split_summary(idx_b, juld, dataset_tag=tag, v2_src=v2_src),
        "viable": True,
    }

    # C: high-observation test (2020 peak year)
    idx_c = assign_chronological_indices(
        juld,
        dataset_tag=tag,
        split_config={
            "train": {"start": "2015-01-01", "end": "2019-12-31"},
            "val": {"start": "2021-01-01", "end": "2021-12-31"},
            "test": {"start": "2020-01-01", "end": "2020-12-31"},
        },
        v2_src=v2_src,
        unassigned="exclude",
    )
    candidates["C_high_observation_test_2020"] = {
        "indices": {k: len(v) for k, v in idx_c.items()},
        "summary": split_summary(idx_c, juld, dataset_tag=tag, v2_src=v2_src),
        "purpose": "best-case density year as test holdout",
        "viable": len(idx_c["test"]) > 0,
    }

    # D: low-observation stress (early sparse years as test)
    idx_d = assign_chronological_indices(
        juld,
        dataset_tag=tag,
        split_config={
            "train": {"start": "2019-01-01", "end": "2021-12-31"},
            "val": {"start": "2020-01-01", "end": "2020-06-30"},
            "test": {"start": "2015-01-01", "end": "2018-12-31"},
        },
        v2_src=v2_src,
        unassigned="exclude",
    )
    candidates["D_low_observation_stress_2015_2018"] = {
        "indices": {k: len(v) for k, v in idx_d.items()},
        "summary": split_summary(idx_d, juld, dataset_tag=tag, v2_src=v2_src),
        "purpose": "sparse early-era generalization stress test",
        "viable": len(idx_d["test"]) > 0,
    }

    recommendation = {
        "default_dissertation_split": "B_chronological_fraction_70_15_15",
        "rationale": (
            "GoM ARGO export spans 2015–2022 only (not 2002–2020). "
            "Candidate A is empty. Chronological 70/15/15 preserves temporal order "
            "and balanced counts without hand-tuning year boundaries."
        ),
        "high_observation_subset": "C_high_observation_test_2020",
        "low_observation_stress_subset": "D_low_observation_stress_2015_2018",
        "common_overlap_split": "E_common_overlap_2015_2021",
        "exclude_periods": ["2022 (sparse tail, n=89)"],
        "evaluation_subsets": {
            "high_observation": {"years": [2020], "split_candidate": "C_high_observation_test_2020"},
            "low_observation": {"years": [2015, 2016, 2017, 2018], "split_candidate": "D_low_observation_stress_2015_2018"},
            "common_overlap": {"years": list(range(2015, 2022)), "split_candidate": "E_common_overlap_2015_2021"},
        },
    }

    return {"candidates": candidates, "recommendation": recommendation, "n_profiles": n}


def write_markdown_census(census: dict, path: Path) -> None:
    lines = [
        "# Data census",
        "",
        f"- **Profiles:** {census['n_profiles']}",
        f"- **Source:** `{census['source']}`",
        f"- **Temporal range:** {census['temporal_range']['min']} → {census['temporal_range']['max']}",
        f"- **Spatial bounds:** lat [{census['spatial_bounds']['lat_min']:.2f}, {census['spatial_bounds']['lat_max']:.2f}], "
        f"lon [{census['spatial_bounds']['lon_min']:.2f}, {census['spatial_bounds']['lon_max']:.2f}]",
        f"- **Depth (m):** {census['depth_range_m'][0]}–{census['depth_range_m'][1]}",
        "",
        "## Profiles by year",
        "",
        "| Year | Count |",
        "|------|------:|",
    ]
    for y, c in census["by_year"].items():
        lines.append(f"| {y} | {c} |")
    lines.extend(["", "## By season", ""])
    for s, c in census["by_season"].items():
        lines.append(f"- **{s}:** {c}")
    lines.extend(["", "## By region (coarse GoM bins)", ""])
    for r, c in census["by_region"].items():
        lines.append(f"- **{r}:** {c}")
    lines.extend(["", "## Depth coverage", ""])
    for k, v in census.get("depth_coverage", {}).items():
        lines.append(f"- **{k}:** {v}")
    lines.extend(["", "## Surface inputs", "", census["surface_inputs"]["note"], ""])
    l3_raw = census["surface_inputs"].get("l3_raw", {})
    if l3_raw:
        lines.append(f"- **L3 raw root:** `{l3_raw.get('path', '?')}` ({l3_raw.get('status', '?')})")
        lines.append(f"- **SSH files:** {l3_raw.get('ssh_files', 0)}")
        lines.append(f"- **ERA5 files:** {l3_raw.get('era5_files', 0)}")
        if l3_raw.get("ssh_date_span"):
            lines.append(f"- **SSH dates:** {l3_raw['ssh_date_span'][0]} → {l3_raw['ssh_date_span'][-1]}")
    path.write_text("\n".join(lines) + "\n")


def write_markdown_split(design: dict, path: Path) -> None:
    rec = design["recommendation"]
    lines = [
        "# Split design",
        "",
        f"**Default dissertation split:** `{rec['default_dissertation_split']}`",
        "",
        rec["rationale"],
        "",
        "## Recommendations",
        "",
        f"1. **Default:** {rec['default_dissertation_split']}",
        f"2. **High-observation test:** {rec['high_observation_subset']}",
        f"3. **Low-observation stress:** {rec['low_observation_stress_subset']}",
        f"4. **Common-overlap era:** {rec['common_overlap_split']}",
        f"5. **Exclude:** {', '.join(rec['exclude_periods'])}",
        "",
        "## Candidate summaries",
        "",
    ]
    for name, cand in design["candidates"].items():
        lines.append(f"### {name}")
        if "error" in cand:
            lines.append(f"- **Error:** {cand['error']}")
            lines.append(f"- **Viable:** {cand['viable']}")
        else:
            sm = cand.get("summary", {})
            counts = sm.get("counts") or cand.get("indices", {})
            lines.append(f"- **Counts:** {counts}")
            if "by_year" in sm:
                for split, yrs in sm["by_year"].items():
                    lines.append(f"- **{split} years:** {yrs}")
            lines.append(f"- **Viable:** {cand.get('viable')}")
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ARGO data census and split design")
    parser.add_argument("-c", "--config", default="config_argo.json", help="ARGO config JSON")
    parser.add_argument(
        "--reports-dir",
        default="../reports",
        help="output directory for census/split reports",
    )
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    if not config_path.is_file():
        config_path = _ROOT / args.config
    reports_dir = Path(args.reports_dir)
    if not reports_dir.is_absolute():
        reports_dir = (_ROOT / reports_dir).resolve()
    reports_dir.mkdir(parents=True, exist_ok=True)

    payload = load_argo_source(config_path)
    census = build_census(payload)
    design = evaluate_split_candidates(
        census, payload["cache"], payload["tag"], payload.get("v2_src")
    )

    write_json(census, reports_dir / "data_census.json")
    write_json(design, reports_dir / "split_design.json")
    write_markdown_census(census, reports_dir / "data_census.md")
    write_markdown_split(design, reports_dir / "split_design.md")

    print(f"census: {reports_dir / 'data_census.json'}")
    print(f"split:  {reports_dir / 'split_design.json'}")
    print(f"default split: {design['recommendation']['default_dissertation_split']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
