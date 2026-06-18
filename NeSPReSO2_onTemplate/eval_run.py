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
from model.loss import decode_latent_profiles, load_decoders_from_dir, make_loss, sklearn_inverse_transform_pcs
from model.metric import per_variable_rmse
from parse_config import ConfigParser, validate_config
from playground import prepare_device, read_json
from train import ensure_cache, set_seed


def _resolve_eval_batch_size(dl_args: dict, split: str) -> None:
    """``batch_size=0`` means one batch over the whole split (inference-only)."""
    if int(dl_args.get("batch_size", 512)) > 0:
        return
    import pickle

    with open(dl_args["cache_path"], "rb") as f:
        n = int(pickle.load(f)["inputs"].shape[0])
    train_frac = float(dl_args.get("train_frac", 0.7))
    val_frac = float(dl_args.get("val_frac", 0.15))
    test_frac = float(dl_args.get("test_frac", 0.15))
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        raise ValueError("train_frac + val_frac + test_frac must equal 1")
    train_len = int(n * train_frac)
    val_len = int(n * val_frac)
    test_len = n - train_len - val_len
    split_lens = {"train": train_len, "val": val_len, "test": test_len}
    dl_args["batch_size"] = max(1, split_lens[split])


def _latent_block_rmse(pred, tgt, outputs):
    out = {}
    start = 0
    for name, k in outputs.items():
        block = pred[:, start : start + k] - tgt[:, start : start + k]
        out[name] = float(np.sqrt(np.mean(block**2)))
        start += k
    return out


def raw_profile_rmse(pred_pcs, true_profiles, pca_models, outputs, indices, *, decoders=None, device=None):
    """RMSE in physical space vs cache ``profiles`` (not PCA-reconstructed targets)."""
    idx = np.asarray(indices, dtype=int)
    if decoders is not None:
        pred_t = torch.tensor(pred_pcs, dtype=torch.float32, device=device)
        pred = decode_latent_profiles(pred_t, decoders, outputs)
        out = {}
        for name in outputs:
            # decode_latent_profiles is (N, n_depth); cache profiles are (n_depth, N).
            diff = pred[name].detach().cpu().numpy() - true_profiles[name][:, idx].T
            out[name] = float(np.sqrt(np.nanmean(diff ** 2)))
        return out

    pred = sklearn_inverse_transform_pcs(pred_pcs, pca_models, outputs)
    out = {}
    for name in outputs:
        diff = pred[name] - true_profiles[name][:, idx]
        out[name] = float(np.sqrt(np.nanmean(diff ** 2)))
    return out


def main(config, checkpoint_path: str, split: str = "test"):
    set_seed(config.config.get("seed", 42))
    ensure_cache(config)

    dl_args = dict(config["data_loader"]["args"])
    target_key = dl_args.get("target_key", "targets")
    weight_key = dl_args.get("weight_key", "weights")
    loss_outputs = OrderedDict(config["outputs"])
    dl_args["split"] = split
    dl_args["shuffle"] = False
    _resolve_eval_batch_size(dl_args, split)
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
    outputs = OrderedDict(ckpt.get("outputs", dict(loss_outputs)))

    from types import SimpleNamespace

    loss_cfg = config.config.get("loss_config") or {}
    decoders = None
    if loss_cfg.get("mode") == "decoder":
        decoders = load_decoders_from_dir(loss_cfg["decoder_dir"], outputs, device)

    true_profiles = data_loader.cache.get("true_profiles")
    if true_profiles is None and data_loader.profiles:
        n = data_loader.cache["inputs"].shape[0]
        true_profiles = {}
        for name in outputs:
            arr = np.asarray(data_loader.profiles[name], dtype=np.float32)
            if arr.shape[1] == n:
                arr = arr.T
            true_profiles[name] = arr

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
        loss_scales=config.config.get("loss_scales"),
        loss_config=loss_cfg,
        targets=data_loader.cache["targets"],
        true_profiles=true_profiles,
        ae_targets=data_loader.cache.get(target_key),
        ae_weights=data_loader.cache.get(weight_key),
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
    tgt_pcs = data_loader.cache[target_key][indices].astype(np.float64)
    pca_tgt = data_loader.cache.get("pca_targets")
    if pca_tgt is not None:
        pca_tgt = pca_tgt[indices].astype(np.float64)

    if loss_cfg.get("mode") == "decoder":
        latent_rmse = _latent_block_rmse(pcs, tgt_pcs, outputs)
    else:
        latent_rmse = per_variable_rmse(pcs, tgt_pcs, pca_models, outputs)

    report = {
        "checkpoint": str(checkpoint_path),
        "cache": str(data_loader.cache_path),
        "dataset_tag": data_loader.dataset_tag,
        "split": split,
        "n_samples": int(n),
        "loss": total_loss / n,
        "latent_target_rmse": latent_rmse,
        "raw_profile_rmse": raw_profile_rmse(
            pcs,
            data_loader.profiles,
            pca_models,
            outputs,
            indices,
            decoders=decoders,
            device=device,
        ),
    }
    if loss_cfg.get("mode") != "decoder" and pca_tgt is not None:
        pca_outputs = OrderedDict(data_loader.cache["outputs"])
        report["pca_target_rmse"] = per_variable_rmse(pcs, pca_tgt, pca_models, pca_outputs)
    elif loss_cfg.get("mode") != "decoder":
        report["pca_target_rmse"] = report["latent_target_rmse"]
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
