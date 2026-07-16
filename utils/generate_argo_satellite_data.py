#!/usr/bin/env python3
"""Generate L4 satellite HDF5 aligned 1:1 with the v2 ARGO profile pickle.

Batch files are written under ``<data_path>/argo_sat_batches/`` so retrieval can
resume after interruption. When every batch exists, batches are merged into the
final ``sat_file`` from config.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
from astropy.time import Time
from tqdm import tqdm

from retrieve_sat import retrieve_satellite_data

_UTILS = Path(__file__).resolve().parent
_PROJECT = _UTILS.parent
_TEMPLATE = _PROJECT / "NeSPReSO2_onTemplate"
if str(_TEMPLATE) not in sys.path:
    sys.path.insert(0, str(_TEMPLATE))

STATIC_PRODUCTS = {"bathymetry"}
BATCH_PREFIX = "argo_sat_batch_"


def _load_argo_stations(v2_pickle: str, v2_src: str | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if v2_src:
        sys.path.insert(0, str(v2_src))
    from nespreso.data.pickle_compat import load_dataset_pickle
    from nespreso.utils.time import datenum_to_datetime

    ds = load_dataset_pickle(v2_pickle)["full_dataset"]
    lat = np.asarray(ds.LAT, dtype=np.float64)
    lon = np.asarray(ds.LON, dtype=np.float64)
    juld = np.array([Time(datenum_to_datetime(float(t))).jd for t in ds.TIME], dtype=np.float64)
    return lat, lon, juld


def _expected_shape(spatial_pad: int, temporal_pad: int, is_static: bool) -> tuple[int, ...]:
    grid = 2 * spatial_pad + 1
    if is_static:
        if spatial_pad == 0:
            return ()
        return (grid, grid)
    if spatial_pad == 0:
        return (temporal_pad + 1, 1, 1)
    return (temporal_pad + 1, grid, grid)


def _products_for_config(
    io_groups: dict | None,
    error_groups: dict | None = None,
) -> dict[str, list[str]]:
    groups = dict(
        io_groups
        or {
            "sss": {"h5_group": "sss", "h5_var": "sos"},
            "sst": {"h5_group": "ostia", "h5_var": "analysed_sst"},
            "ssh": {"h5_group": "ssh", "h5_var": "adt"},
        }
    )
    # Phase 2.2: optional error vars under the same HDF5 product groups
    if error_groups:
        groups.update(error_groups)
    h5_to_product = {"sss": "sss", "ostia": "ostia", "ssh": "ssh", "bathymetry": "bathymetry"}
    products: dict[str, list[str]] = {"bathymetry": ["elevation"]}
    for spec in groups.values():
        prod = h5_to_product.get(spec["h5_group"])
        if prod is None:
            continue
        products.setdefault(prod, [])
        var = spec["h5_var"]
        if var not in products[prod]:
            products[prod].append(var)
    return products


def _station_tuple(lat, lon, juld, idx: int) -> tuple:
    return (float(lat), float(lon), float(juld), b"argo_v2", int(idx))


def batch_path(batch_dir: str, batch_start: int, batch_end: int, batch_size: int) -> str:
    return os.path.join(
        batch_dir,
        f"{BATCH_PREFIX}b{batch_size:04d}_{batch_start:06d}_{batch_end:06d}.h5",
    )


def manifest_path(batch_dir: str) -> str:
    return os.path.join(batch_dir, "manifest.json")


def _write_manifest(
    batch_dir: str,
    *,
    n_total: int,
    batch_size: int,
    spatial_pad: int,
    temporal_pad: int,
    completed: list[list[int]],
) -> None:
    os.makedirs(batch_dir, exist_ok=True)
    payload = {
        "n_total": n_total,
        "batch_size": batch_size,
        "spatial_pad": spatial_pad,
        "temporal_pad": temporal_pad,
        "completed_batches": completed,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(manifest_path(batch_dir), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_batch_h5(
    path: str,
    batch_start: int,
    stations: list[tuple],
    results: dict[int, dict],
    products: dict,
    spatial_pad: int,
    temporal_pad: int,
) -> None:
    """Write one resumable batch file (station indices are global)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    n = len(stations)
    with h5py.File(path, "w") as f:
        meta = f.create_group("metadata")
        meta.attrs["spatial_padding"] = spatial_pad
        meta.attrs["temporal_padding"] = temporal_pad
        meta.attrs["batch_start"] = batch_start
        meta.attrs["batch_end"] = batch_start + n
        meta.attrs["n_stations"] = n
        meta.attrs["dataset_tag"] = "argo_l4"

        for product_name, var_names in products.items():
            product_group = f.create_group(product_name)
            is_static = product_name in STATIC_PRODUCTS
            shape_tail = _expected_shape(spatial_pad, temporal_pad, is_static)
            sample = None
            for res in results.values():
                if product_name in res and res[product_name].get("data"):
                    sample = res[product_name]
                    break
            if sample is None:
                continue
            coords = sample["coordinates"]
            product_group.create_dataset("latitude_grid", data=coords["latitude"])
            product_group.create_dataset("longitude_grid", data=coords["longitude"])
            for var_name in var_names:
                full_shape = (n,) + shape_tail
                ds = product_group.create_dataset(
                    var_name,
                    shape=full_shape,
                    dtype=np.float32,
                    compression="gzip",
                    compression_opts=4,
                    fillvalue=np.nan,
                )
                for local_i, station_data in results.items():
                    if product_name not in station_data:
                        continue
                    pdata = station_data[product_name]["data"]
                    if var_name not in pdata:
                        continue
                    arr = np.asarray(pdata[var_name], dtype=np.float32)
                    if arr.shape == shape_tail or (not shape_tail and arr.shape == ()):
                        ds[local_i] = arr

        station_group = f.create_group("stations")
        station_group.create_dataset("latitude", data=[s[0] for s in stations])
        station_group.create_dataset("longitude", data=[s[1] for s in stations])
        station_group.create_dataset("julian_date", data=[s[2] for s in stations])
        station_group.create_dataset("source_file", data=np.array([s[3] for s in stations], dtype="S"))
        station_group.create_dataset("profile_index", data=[s[4] for s in stations])


def list_batch_files(batch_dir: str, batch_size: int) -> list[str]:
    pattern = os.path.join(batch_dir, f"{BATCH_PREFIX}b{batch_size:04d}_*.h5")
    return sorted(glob.glob(pattern))


def assert_batch_dir_matches_products(
    batch_dir: str,
    batch_size: int,
    products: dict[str, list[str]],
) -> None:
    """Refuse resume when on-disk batches lack datasets required by ``products``.

    Typical failure: v3 ``error_groups`` against a v2-era batch directory.
    """
    files = list_batch_files(batch_dir, batch_size)
    if not files:
        return
    missing: list[str] = []
    with h5py.File(files[0], "r") as f:
        for prod, vars_ in products.items():
            if prod not in f:
                missing.append(f"{prod}/")
                continue
            for var in vars_:
                if var not in f[prod]:
                    missing.append(f"{prod}/{var}")
    if missing:
        raise RuntimeError(
            "Batch directory schema mismatch "
            f"(missing {missing} in {files[0]}). "
            "Regenerate from scratch or use v2 config."
        )


def combine_argo_batches(
    batch_files: list[str],
    output_path: str,
    *,
    spatial_pad: int,
    temporal_pad: int,
) -> int:
    """Merge batch HDF5s into a single station-major file. Returns total N."""
    if not batch_files:
        raise ValueError("no batch files to combine")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    n_total = 0
    for bf in batch_files:
        with h5py.File(bf, "r") as f:
            n_total += int(f["metadata"].attrs["n_stations"])

    with h5py.File(output_path, "w") as out:
        meta = out.create_group("metadata")
        meta.attrs["spatial_padding"] = spatial_pad
        meta.attrs["temporal_padding"] = temporal_pad
        meta.attrs["n_stations"] = n_total
        meta.attrs["dataset_tag"] = "argo_l4"
        meta.attrs["n_batches"] = len(batch_files)

        product_names = []
        with h5py.File(batch_files[0], "r") as f0:
            product_names = [g for g in f0.keys() if g not in ("metadata", "stations")]

        out_stations = out.create_group("stations")
        lat_parts, lon_parts, jd_parts, src_parts, idx_parts = [], [], [], [], []

        for product_name in product_names:
            out_group = out.create_group(product_name)
            with h5py.File(batch_files[0], "r") as f0:
                out_group.create_dataset("latitude_grid", data=f0[product_name]["latitude_grid"][:])
                out_group.create_dataset("longitude_grid", data=f0[product_name]["longitude_grid"][:])
                var_names = [k for k in f0[product_name].keys() if k not in ("latitude_grid", "longitude_grid")]

            for var_name in var_names:
                sample_shape = None
                is_static = product_name in STATIC_PRODUCTS
                shape_tail = _expected_shape(spatial_pad, temporal_pad, is_static)
                full_shape = (n_total,) + shape_tail
                ds_out = out_group.create_dataset(
                    var_name,
                    shape=full_shape,
                    dtype=np.float32,
                    compression="gzip",
                    compression_opts=4,
                    fillvalue=np.nan,
                )
                offset = 0
                for bf in batch_files:
                    with h5py.File(bf, "r") as f:
                        n_batch = int(f["metadata"].attrs["n_stations"])
                        if var_name not in f[product_name]:
                            offset += n_batch
                            continue
                        chunk = f[product_name][var_name][:]
                        if sample_shape is None:
                            sample_shape = chunk.shape[1:]
                        ds_out[offset : offset + n_batch] = chunk
                        offset += n_batch

        offset = 0
        for bf in batch_files:
            with h5py.File(bf, "r") as f:
                n_batch = int(f["metadata"].attrs["n_stations"])
                st = f["stations"]
                lat_parts.append(st["latitude"][:])
                lon_parts.append(st["longitude"][:])
                jd_parts.append(st["julian_date"][:])
                src_parts.append(st["source_file"][:])
                idx_parts.append(st["profile_index"][:])
                offset += n_batch

        out_stations.create_dataset("latitude", data=np.concatenate(lat_parts))
        out_stations.create_dataset("longitude", data=np.concatenate(lon_parts))
        out_stations.create_dataset("julian_date", data=np.concatenate(jd_parts))
        out_stations.create_dataset("source_file", data=np.concatenate(src_parts))
        out_stations.create_dataset("profile_index", data=np.concatenate(idx_parts))

    return n_total


def _batch_ranges(n: int, batch_size: int) -> list[tuple[int, int]]:
    return [(start, min(start + batch_size, n)) for start in range(0, n, batch_size)]


def generate_argo_satellite_h5(
    config: dict,
    *,
    force: bool = False,
    max_samples: int | None = None,
    batch_size: int = 100,
    skip_existing: bool = False,
    resume: bool = True,
    combine_only: bool = False,
) -> str:
    io_cfg = config["io"]
    data_path = io_cfg["data_path"]
    sat_file = io_cfg.get("sat_file", "satellite_NeSPReSO_v2_ARGO_GoM.h5")
    output_path = os.path.join(data_path, sat_file)
    batch_dir = os.path.join(data_path, io_cfg.get("batch_dir", "argo_sat_batches"))

    if os.path.isfile(output_path) and not force and skip_existing:
        print(f"Final HDF5 exists, skipping: {output_path}")
        return output_path

    spatial_pad = int(io_cfg.get("spatial_pad", 2))
    temporal_pad = int(io_cfg.get("temporal_pad", 6))
    products = _products_for_config(io_cfg.get("groups"), io_cfg.get("error_groups"))

    if force:
        pattern = os.path.join(batch_dir, f"{BATCH_PREFIX}b{batch_size:04d}_*.h5")
        for bf in glob.glob(pattern):
            os.remove(bf)
        mp = manifest_path(batch_dir)
        if os.path.isfile(mp):
            os.remove(mp)
        if os.path.isfile(output_path):
            os.remove(output_path)
    elif resume and not combine_only:
        # ponytail: ceiling = first-batch-only check; upgrade to all batches if mixed schemas appear
        assert_batch_dir_matches_products(batch_dir, batch_size, products)

    lat, lon, juld = _load_argo_stations(io_cfg["v2_pickle"], io_cfg.get("v2_src"))
    n = len(lat)
    if max_samples is not None:
        n = min(n, int(max_samples))
        lat, lon, juld = lat[:n], lon[:n], juld[:n]

    ranges = _batch_ranges(n, batch_size)
    completed: list[list[int]] = []

    if not combine_only:
        for batch_start, batch_end in ranges:
            bpath = batch_path(batch_dir, batch_start, batch_end, batch_size)
            if resume and os.path.isfile(bpath) and not force:
                print(f"Resume: skip existing batch {batch_start}-{batch_end}")
                completed.append([batch_start, batch_end])
                continue

            print(f"Retrieving batch {batch_start}-{batch_end} of {n} stations")
            queries = [
                (float(lat[i]), float(lon[i]), float(juld[i]))
                for i in range(batch_start, batch_end)
            ]
            batch_results = retrieve_satellite_data(
                queries,
                products,
                spatial_pad,
                temporal_pad,
            )
            n_batch = batch_end - batch_start
            stations = [
                _station_tuple(lat[i], lon[i], juld[i], i)
                for i in range(batch_start, batch_end)
            ]
            reindexed: dict[int, dict] = {}
            for local_i in range(n_batch):
                reindexed[local_i] = batch_results.get(local_i, {})
            save_batch_h5(bpath, batch_start, stations, reindexed, products, spatial_pad, temporal_pad)
            completed.append([batch_start, batch_end])
            _write_manifest(batch_dir, n_total=n, batch_size=batch_size, spatial_pad=spatial_pad, temporal_pad=temporal_pad, completed=completed)
            print(f"Wrote batch {bpath}")

    batch_files = list_batch_files(batch_dir, batch_size)
    expected = len(ranges)
    if len(batch_files) < expected:
        missing = expected - len(batch_files)
        print(f"Incomplete: {len(batch_files)}/{expected} batches on disk ({missing} missing). Re-run to resume.")
        return output_path

    if os.path.isfile(output_path) and not force and resume:
        print(f"Final HDF5 already exists: {output_path}")
        return output_path

    print(f"Combining {len(batch_files)} batches -> {output_path}")
    n_written = combine_argo_batches(batch_files, output_path, spatial_pad=spatial_pad, temporal_pad=temporal_pad)
    print(f"ARGO satellite HDF5 saved to {output_path} (N={n_written})")
    return output_path


def main(argv: list[str] | None = None) -> int:
    from base.util import read_json

    parser = argparse.ArgumentParser(description="Generate ARGO-aligned L4 satellite HDF5 (resumable batches)")
    parser.add_argument("-c", "--config", required=True, help="NeSPReSO config JSON (argo_l4)")
    parser.add_argument("--force", action="store_true", help="redo all batches and final HDF5")
    parser.add_argument("--skip-existing", action="store_true", help="skip if final HDF5 exists")
    parser.add_argument("--no-resume", action="store_true", help="re-fetch batches even if batch files exist")
    parser.add_argument("--combine-only", action="store_true", help="only merge existing batch files")
    parser.add_argument("--max-samples", type=int, default=None, help="limit stations (smoke)")
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args(argv)
    cfg = read_json(args.config)
    generate_argo_satellite_h5(
        cfg,
        force=args.force,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        skip_existing=args.skip_existing,
        resume=not args.no_resume,
        combine_only=args.combine_only,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
