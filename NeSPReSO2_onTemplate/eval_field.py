#!/usr/bin/env python3
"""Evaluate field U-Net on test dates (gather at ARGO pixels)."""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

import data_loader.data_loaders as module_data
import model.model as module_arch
from eval_run import raw_profile_rmse, write_report
from model.loss import make_loss, reconstruct_physical_profiles
from parse_config import ConfigParser, validate_config
from base.util import prepare_device, read_json
from train import ensure_cache, set_seed, surface_residual_layout_from_cache


def main(config, checkpoint_path: str, split: str = "test"):
    set_seed(config.config.get("seed", 42))
    ensure_cache(config)

    dl_args = dict(config["data_loader"]["args"])
    dl_args["split"] = split
    dl_args["shuffle"] = False
    data_loader = getattr(module_data, config["data_loader"]["type"])(**dl_args)

    device, _ = prepare_device(config["n_gpu"])
    model = config.init_obj("arch", module_arch).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state_dict)
    model.eval()

    pca_models = ckpt.get("pca_models", data_loader.pca_models)
    outputs = OrderedDict(ckpt.get("outputs", dict(config["outputs"])))

    from types import SimpleNamespace

    loss_fn = make_loss(
        pca_models=pca_models,
        outputs=outputs,
        weights=data_loader.weights,
        device=device,
        steric_config=config.config.get("steric"),
        steric_meta=SimpleNamespace(
            LAT=data_loader.LAT,
            LON=data_loader.LON,
            PRES=data_loader.PRES,
            ssh_obs_sla=data_loader.cache.get("ssh_obs_sla"),
            clim_steric=data_loader.cache.get("clim_steric"),
            steric_calibration=data_loader.cache.get("steric_calibration"),
        ),
        clim_profiles=data_loader.cache.get("clim_profiles"),
    )

    pcs_list, idx_list = [], []
    total_loss = 0.0
    n_pix = 0
    with torch.no_grad():
        for batch in data_loader:
            fields, targets, indices, pixel_rc, batch_of_pixel = batch
            fields = fields.to(device)
            targets = targets.to(device)
            indices = indices.to(device)
            pixel_rc = pixel_rc.to(device)
            batch_of_pixel = batch_of_pixel.to(device)
            out = model(fields)
            rows = pixel_rc[:, 0].long()
            cols = pixel_rc[:, 1].long()
            pred = out[batch_of_pixel, :, rows, cols]
            loss = loss_fn(pred, targets, indices)
            total_loss += loss.item() * pred.size(0)
            n_pix += pred.size(0)
            pcs_list.append(pred.cpu().numpy())
            idx_list.append(indices.cpu().numpy())

    pcs = np.vstack(pcs_list)
    indices = np.concatenate(idx_list)
    report = {
        "checkpoint": str(checkpoint_path),
        "cache": str(data_loader.cache_path),
        "dataset_tag": data_loader.dataset_tag,
        "split": split,
        "n_samples": int(len(indices)),
        "loss": total_loss / max(1, n_pix),
        "raw_profile_rmse": raw_profile_rmse(
            pcs,
            data_loader.profiles,
            pca_models,
            outputs,
            indices,
            clim_profiles=data_loader.cache.get("clim_profiles"),
        ),
    }
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Field U-Net test evaluation")
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-r", "--resume", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    cfg = read_json(args.config)
    validate_config(cfg)
    config = ConfigParser(cfg)
    report = main(config, args.resume, split=args.split)
    if args.out:
        write_report(report, args.out)
