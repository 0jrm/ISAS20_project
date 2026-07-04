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
from base.util import prepare_device, read_json
from preproc.climatology import eval_climatology
from train import ensure_cache, set_seed


def main(config, checkpoint_path: str, out_path: str, date_start: str | None = None, date_end: str | None = None):
    set_seed(config.config.get("seed", 42))
    ensure_cache(config)
    import pickle

    cache_path = config.config["data_loader"]["args"]["cache_path"]
    with open(cache_path, "rb") as f:
        cache = pickle.load(cache)

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

    date_strs = [str(d)[:10] for d in dates]
    if date_start:
        date_strs = [d for d in date_strs if d >= date_start]
    if date_end:
        date_strs = [d for d in date_strs if d <= date_end]

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
                lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
                flat_lat = lat_grid.ravel().astype(np.float32)
                flat_lon = lon_grid.ravel().astype(np.float32)
                juld = np.full(flat_lat.shape[0], float(ord(ds)), dtype=np.float32)
                clim_prof = eval_climatology(clim, flat_lat, flat_lon, juld)
                for name in outputs:
                    prof[name] = prof[name] + clim_prof[name]
            nz = prof["temperature"].shape[0]
            temp_cube.append(prof["temperature"].T.reshape(h, w, nz))
            sal_cube.append(prof["salinity"].T.reshape(h, w, nz))
            out_dates.append(ds)

    temp_arr = np.stack(temp_cube, axis=0)
    sal_arr = np.stack(sal_cube, axis=0)
    temp_arr[..., land] = np.nan
    sal_arr[..., land] = np.nan

    ds_out = xr.Dataset(
        {
            "temperature": (["time", "lat", "lon", "depth"], temp_arr),
            "salinity": (["time", "lat", "lon", "depth"], sal_arr),
        },
        coords={"time": out_dates, "lat": lats, "lon": lons, "depth": pres},
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    ds_out.to_netcdf(out_path)
    print(f"Wrote {out_path}")


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
