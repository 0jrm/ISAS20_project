#!/usr/bin/env python3
"""Tier A EDA for NeSPReSO v1 global satellite HDF5 (Phase 6).

Reads station coordinates and scalar SST/SSS/SSH from the global satellite HDF5,
optionally filters by config ``io.BBox``, and writes summary JSON + coverage plots.
No ML — safe to run even when ``profiles_NeSPReSO_v1_global.h5`` is unreadable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preproc.preproc_isas_sat import get_bbox_mask


def _resolve_data_path(io: dict) -> Path:
    data_path = Path(io["data_path"])
    if not data_path.is_absolute():
        data_path = (ROOT / data_path).resolve()
    return data_path


def _scalar_field(ds: h5py.Dataset) -> np.ndarray:
    arr = ds[:]
    if arr.ndim == 4:
        return arr[:, 0, 0, 0]
    if arr.ndim == 1:
        return arr
    return arr.reshape(arr.shape[0], -1)[:, 0]


def coverage_stats(values: np.ndarray) -> dict[str, float | int]:
    finite = np.isfinite(values)
    return {
        "n": int(values.size),
        "n_finite": int(finite.sum()),
        "frac_finite": float(finite.mean()) if values.size else 0.0,
        "mean": float(np.nanmean(values)) if finite.any() else float("nan"),
        "std": float(np.nanstd(values)) if finite.sum() > 1 else float("nan"),
    }


def run_eda(config_path: Path, out_dir: Path) -> dict:
    cfg = json.loads(config_path.read_text())
    io = cfg["io"]
    sat_path = _resolve_data_path(io) / io.get("sat_file", "satellite_NeSPReSO_v1_global.h5")
    bbox = io.get("BBox")
    groups = io.get(
        "groups",
        {
            "sss": {"h5_group": "sss", "h5_var": "sos"},
            "sst": {"h5_group": "ostia", "h5_var": "analysed_sst"},
            "ssh": {"h5_group": "ssh", "h5_var": "adt"},
        },
    )

    with h5py.File(sat_path, "r") as sf:
        lat = sf["stations/latitude"][:].astype(np.float64)
        lon = sf["stations/longitude"][:].astype(np.float64)
        mask = get_bbox_mask(lat, lon, bbox)
        fields: dict[str, np.ndarray] = {}
        lengths: dict[str, int] = {}
        for key, spec in groups.items():
            ds = sf[f"{spec['h5_group']}/{spec['h5_var']}"]
            lengths[key] = int(ds.shape[0])
            vals = _scalar_field(ds)
            if vals.shape[0] == lat.shape[0]:
                fields[key] = vals[mask]
            else:
                # ponytail: shorter SSS row count — only keep in-bounds station indices.
                idx = np.where(mask)[0]
                in_bounds = idx < vals.shape[0]
                fields[key] = vals[idx[in_bounds]]

    lat_r = lat[mask]
    lon_r = lon[mask]

    summary = {
        "config": str(config_path),
        "sat_file": str(sat_path),
        "bbox": bbox,
        "n_stations_global": int(lat.size),
        "n_stations_region": int(mask.sum()),
        "field_row_counts": lengths,
        "lat_range": [float(lat_r.min()), float(lat_r.max())] if lat_r.size else [],
        "lon_range": [float(lon_r.min()), float(lon_r.max())] if lon_r.size else [],
        "fields": {k: coverage_stats(v) for k, v in fields.items()},
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist2d(lon, lat, bins=120, cmap="viridis")
    axes[0].set_title("Global station density")
    axes[0].set_xlabel("longitude")
    axes[0].set_ylabel("latitude")
    if bbox is not None:
        min_lat, max_lat, min_lon, max_lon = bbox
        axes[0].add_patch(
            plt.Rectangle(
                (min_lon, min_lat),
                max_lon - min_lon,
                max_lat - min_lat,
                fill=False,
                edgecolor="red",
                linewidth=1.5,
            )
        )

    axes[1].hist2d(lon_r, lat_r, bins=40, cmap="magma")
    axes[1].set_title(f"Regional subset (N={mask.sum()})")
    axes[1].set_xlabel("longitude")
    axes[1].set_ylabel("latitude")
    fig.tight_layout()
    fig.savefig(out_dir / "station_density.png", dpi=120)
    plt.close(fig)

    if fields:
        ncols = min(3, len(fields))
        fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))
        if ncols == 1:
            axes = [axes]
        for ax, (name, vals) in zip(axes, fields.items()):
            finite = np.isfinite(vals)
            ax.hist(vals[finite], bins=50, color="steelblue", alpha=0.85)
            ax.set_title(f"{name} (finite {finite.sum()}/{vals.size})")
        fig.tight_layout()
        fig.savefig(out_dir / "field_histograms.png", dpi=120)
        plt.close(fig)

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-c",
        "--config",
        default="config_isas_global_gom.json",
        help="Config JSON with io.data_path and optional io.BBox",
    )
    parser.add_argument(
        "--out-dir",
        default="saved/plots/global_eda",
        help="Output directory for summary.json and plots",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    summary = run_eda(config_path, Path(args.out_dir))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
