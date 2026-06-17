#!/usr/bin/env python3
"""Evaluate a checkpoint on the test split (raw profile RMSE + PCA loss)."""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

import data_loader.data_loaders as module_data
import model.model as module_arch
from model.loss import make_loss, sklearn_inverse_transform_pcs
from model.metric import per_variable_rmse
from parse_config import ConfigParser, validate_config
from playground import prepare_device, read_json
from train import ensure_cache, set_seed


def raw_profile_rmse(pred_pcs, true_profiles, pca_models, outputs, indices):
    """RMSE in physical space vs cache ``profiles`` (not PCA-reconstructed targets)."""
    pred = sklearn_inverse_transform_pcs(pred_pcs, pca_models, outputs)
    out = {}
    idx = np.asarray(indices, dtype=int)
    for name in outputs:
        diff = pred[name] - true_profiles[name][:, idx]
        out[name] = float(np.sqrt(np.nanmean(diff ** 2)))
    return out


def main(config, checkpoint_path: str, split: str = "test"):
    set_seed(config.config.get("seed", 42))
    ensure_cache(config)

    dl_args = dict(config["data_loader"]["args"])
    dl_args["split"] = split
    dl_args["shuffle"] = False
    data_loader = getattr(module_data, config["data_loader"]["type"])(**dl_args)

    if not data_loader.profiles:
        raise ValueError("cache missing 'profiles'; rebuild with build_train_cache --force or export_v2_cache")

    device, _ = prepare_device(config["n_gpu"])
    model = config.init_obj("arch", module_arch).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state_dict)
    model.eval()

    pca_models = ckpt.get("pca_models", data_loader.pca_models)
    outputs = OrderedDict(ckpt.get("outputs", dict(data_loader.outputs)))

    from types import SimpleNamespace

    loss_fn = make_loss(
        pca_models=pca_models,
        outputs=outputs,
        weights=data_loader.weights,
        device=device,
        density_config=config.config.get("density"),
        density_meta=SimpleNamespace(
            LAT=data_loader.LAT,
            LON=data_loader.LON,
            PRES=data_loader.PRES,
            min_depth=data_loader.min_depth,
            max_depth=data_loader.max_depth,
        ),
    )

    total_loss = 0.0
    pcs_list, idx_list = [], []
    with torch.no_grad():
        for data, target, indices in data_loader:
            data = data.to(device)
            target = target.to(device)
            indices = indices.to(device)
            output = model(data)
            total_loss += loss_fn(output, target, indices).item() * data.size(0)
            pcs_list.append(output.cpu().numpy())
            idx_list.append(indices.cpu().numpy())

    n = len(data_loader.dataset)
    pcs = np.vstack(pcs_list)
    indices = np.concatenate(idx_list)
    tgt_pcs = data_loader.cache["targets"][indices].astype(np.float64)

    report = {
        "checkpoint": str(checkpoint_path),
        "cache": str(data_loader.cache_path),
        "dataset_tag": data_loader.dataset_tag,
        "split": split,
        "n_samples": int(n),
        "loss": total_loss / n,
        "pca_target_rmse": per_variable_rmse(pcs, tgt_pcs, pca_models, outputs),
        "raw_profile_rmse": raw_profile_rmse(pcs, data_loader.profiles, pca_models, outputs, indices),
    }
    text = json.dumps(report, indent=2)
    print(text)
    return report


def write_report(report, out_path):
    Path(out_path).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeSPReSO test-split evaluation")
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-r", "--resume", required=True, help="checkpoint .pth")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("-d", "--device", default=None)
    parser.add_argument("--out", default=None, help="write JSON report to this path")
    args = parser.parse_args()
    if args.device:
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    cfg = read_json(args.config)
    validate_config(cfg)
    config = ConfigParser(cfg)
    report = main(config, args.resume, split=args.split)
    if args.out:
        write_report(report, args.out)
