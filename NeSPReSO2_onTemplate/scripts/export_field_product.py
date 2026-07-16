#!/usr/bin/env python3
"""Export gridded T/S netCDF from a trained field U-Net (dissertation deliverable)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import xarray as xr

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import model.model as module_arch
from model.loss import reconstruct_physical_profiles
from parse_config import ConfigParser, validate_config
from base.split_utils import dates_to_juld
from base.util import prepare_device, read_json
from preproc.climatology import eval_climatology
from train import ensure_cache, set_seed


def main(config, checkpoint_path: str, out_path: str, date_start: str | None = None, date_end: str | None = None):
    set_seed(config.config.get("seed", 42))
    ensure_cache(config)
    import pickle

    cache_path = config.config["data_loader"]["args"]["cache_path"]
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)

    device, _ = prepare_device(config["n_gpu"])
    model = config.init_obj("arch", module_arch).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)))
    model.eval()

    pca_models = ckpt.get("pca_models", cache["pca_models"])
    outputs = ckpt.get("outputs", cache["outputs"])
    clim = cache.get("climatology")
    fields = np.asarray(cache["fields"], dtype=np.float32)
    dates = cache["dates"]
    lats = cache["grid_lats"]
    lons = cache["grid_lons"]
    land = fields[0, 2] < 0.5

    from data_loader.data_loaders import _time_channels_for_doy
    from datetime import date

    date_strs = {str(d)[:10] for d in dates}
    if date_start:
        date_strs = {d for d in date_strs if d >= date_start}
    if date_end:
        date_strs = {d for d in date_strs if d <= date_end}
    if not date_strs:
        raise ValueError(
            f"no dates in cache within [{date_start or '-inf'}, {date_end or '+inf'}]; "
            f"cache spans {min(str(d)[:10] for d in dates)}..{max(str(d)[:10] for d in dates)}"
        )

    # Climatology JULD convention comes from the climatology's own source tag, not the
    # field cache's "argo_field". eval_climatology -> design_matrix decodes juld with
    # clim.meta["dataset_tag"], so the encode must use the same tag or the day-of-year
    # (and therefore the whole seasonal cycle) is silently wrong.
    clim_tag = (clim.meta.get("dataset_tag", "argo_v2") if clim is not None else "argo_v2")

    def juld_for_date(ds: str) -> float:
        return float(dates_to_juld([ds], dataset_tag=clim_tag)[0])

    # Grid is fixed across dates — build it once, not once per date.
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")

    pres = np.asarray(cache["PRES"], dtype=np.float32)
    temp_cube = []
    sal_cube = []
    out_dates = []

    with torch.no_grad():
        for i, d in enumerate(dates):
            ds = str(d)[:10]
            if ds not in date_strs:
                continue
            base = fields[i]
            doy = date.fromisoformat(ds).timetuple().tm_yday
            tch = _time_channels_for_doy(doy)
            tmap = np.broadcast_to(tch[:, None, None], (4, base.shape[1], base.shape[2])).copy()
            field = np.concatenate([base, tmap], axis=0)
            x = torch.tensor(field[None], dtype=torch.float32, device=device)
            out = model(x)[0].cpu().numpy()
            h, w = base.shape[1], base.shape[2]
            pcs_flat = out.reshape(sum(outputs.values()), -1).T
            prof = reconstruct_physical_profiles(pcs_flat, pca_models, outputs)
            if clim is not None:
                flat_lat = lat_grid.ravel().astype(np.float32)
                flat_lon = lon_grid.ravel().astype(np.float32)
                # JULD must be in the convention eval_climatology decodes against — that is
                # clim.meta's dataset_tag (the *source* cache, e.g. argo_v2 MATLAB datenum),
                # not the field cache's own "argo_field" tag.
                juld = np.full(flat_lat.shape[0], float(juld_for_date(ds)), dtype=np.float64)
                clim_prof = eval_climatology(clim, flat_lat, flat_lon, juld)
                for name in outputs:
                    prof[name] = prof[name] + clim_prof[name]
            nz = prof["temperature"].shape[0]
            temp_cube.append(prof["temperature"].T.reshape(h, w, nz))
            sal_cube.append(prof["salinity"].T.reshape(h, w, nz))
            out_dates.append(ds)

    temp_arr = np.stack(temp_cube, axis=0)
    sal_arr = np.stack(sal_cube, axis=0)
    # arrays are (time, lat, lon, depth) and `land` is (lat, lon): it must index axes 1-2.
    # `[..., land]` would consume the LAST two axes (lon, depth) instead and raise.
    temp_arr[:, land, :] = np.nan
    sal_arr[:, land, :] = np.nan

    ds_out = xr.Dataset(
        {
            "temperature": (["time", "lat", "lon", "depth"], temp_arr),
            "salinity": (["time", "lat", "lon", "depth"], sal_arr),
        },
        coords={
            "time": np.asarray(out_dates, dtype="datetime64[D]"),
            "lat": lats,
            "lon": lons,
            "depth": pres,
        },
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    ds_out.to_netcdf(out_path)
    print(f"Wrote {out_path}")
    report_plausibility(temp_arr, sal_arr)


# Physical bounds for the Gulf of Mexico. Deliberately generous: this is a "did the
# pipeline emit an ocean?" tripwire, not a science check.
GOM_T_RANGE = (2.0, 34.0)
GOM_S_RANGE = (30.0, 38.0)


def report_plausibility(temp_arr: np.ndarray, sal_arr: np.ndarray) -> bool:
    """Print T/S ranges and flag unphysical output.

    A netCDF that opens cleanly is not a netCDF that contains an ocean: this script wrote
    `ord("2020-01-01")` as a date for its whole life and nothing noticed, because nothing
    ever looked at the numbers. Ranges, not existence.
    """
    ok = True
    for name, arr, (lo, hi) in (
        ("temperature", temp_arr, GOM_T_RANGE),
        ("salinity", sal_arr, GOM_S_RANGE),
    ):
        finite = np.isfinite(arr)
        if not finite.any():
            print(f"  {name}: NO FINITE VALUES — the product is empty")
            ok = False
            continue
        amin, amax = float(np.nanmin(arr)), float(np.nanmax(arr))
        n_bad = int(((arr < lo) | (arr > hi)).sum())
        frac = n_bad / max(1, int(finite.sum()))
        status = "ok" if n_bad == 0 else "OUT OF RANGE"
        print(f"  {name}: {amin:.2f}..{amax:.2f} (expect {lo}..{hi}) [{status}]"
              + (f" — {n_bad} pts ({100*frac:.2f}% of finite) outside" if n_bad else ""))
        if n_bad:
            ok = False
    prof = np.nanmean(temp_arr, axis=(0, 1, 2))
    if np.isfinite(prof[0]) and np.isfinite(prof[-1]) and prof[0] <= prof[-1]:
        print(f"  WARNING: mean T does not decrease with depth ({prof[0]:.2f} -> {prof[-1]:.2f})")
        ok = False
    if not ok:
        print("  ⚠️  Output is NOT physically plausible — do not ship this as a product.\n"
              "      Known cause: the climatology is a ridge fit on the ARGO sampling hull and\n"
              "      extrapolates badly outside it; the export grid is larger than that hull.\n"
              "      Clip the grid to the hull (or refit/regularize) before trusting the edges.")
    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-r", "--resume", required=True)
    parser.add_argument("-o", "--out", required=True)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    args = parser.parse_args()
    cfg = read_json(args.config)
    validate_config(cfg)
    main(ConfigParser(cfg), args.resume, args.out, date_start=args.start, date_end=args.end)
