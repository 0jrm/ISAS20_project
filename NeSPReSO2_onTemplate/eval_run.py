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
from model.loss import HEAVE_LOSS_MODES, decode_latent_profiles, load_decoders_from_dir, make_loss, reconstruct_physical_profiles, sklearn_inverse_transform_pcs
from train import surface_residual_layout_from_cache
from model.metric import per_variable_rmse
from parse_config import ConfigParser, validate_config
from base.pairing import assert_cache_checkpoint_pair
from base.util import prepare_device, read_json
from train import ensure_cache, set_seed


def _resolve_eval_batch_size(dl_args: dict, split: str) -> None:
    """``batch_size=0`` means one batch over the whole split (inference-only)."""
    if int(dl_args.get("batch_size", 512)) > 0:
        return
    import pickle

    with open(dl_args["cache_path"], "rb") as f:
        cache = pickle.load(f)
    n = int(cache["inputs"].shape[0])
    from base.split_utils import build_split_indices

    dl_cfg = {
        "split_mode": dl_args.get("split_mode", "random"),
        "split_config": dl_args.get("split_config"),
        "train_frac": float(dl_args.get("train_frac", 0.7)),
        "val_frac": float(dl_args.get("val_frac", 0.15)),
        "test_frac": float(dl_args.get("test_frac", 0.15)),
        "split_seed": int(dl_args.get("split_seed", 42)),
        "unassigned": dl_args.get("unassigned", "exclude"),
    }
    indices = build_split_indices(
        n,
        cache.get("JULD"),
        dl_cfg,
        dataset_tag=cache.get("dataset_tag", "unknown"),
        v2_src=dl_args.get("v2_src"),
    )
    dl_args["batch_size"] = max(1, len(indices[split]))


def _latent_block_rmse(pred, tgt, outputs):
    out = {}
    start = 0
    for name, k in outputs.items():
        block = pred[:, start : start + k] - tgt[:, start : start + k]
        out[name] = float(np.sqrt(np.mean(block**2)))
        start += k
    return out


def raw_profile_rmse(
    pred_pcs,
    true_profiles,
    pca_models,
    outputs,
    indices,
    *,
    decoders=None,
    device=None,
    inputs=None,
    surface_residual_layout=None,
    bottom_depth=None,
    pres_levels=None,
    clim_profiles=None,
    joint_eof_meta=None,
):
    """RMSE in physical space vs cache ``profiles`` (not PCA-reconstructed targets)."""
    idx = np.asarray(indices, dtype=int)
    depth_mask = None
    if bottom_depth is not None and pres_levels is not None:
        pres = np.asarray(pres_levels, dtype=np.float32)
        bd = np.asarray(bottom_depth, dtype=np.float32)[idx]
        depth_mask = pres[np.newaxis, :] <= bd[:, np.newaxis]

    def _masked_rmse(diff):
        if depth_mask is not None:
            masked = diff.T
            valid = depth_mask
            if masked.shape != valid.shape:
                valid = valid.T
            sel = masked[valid]
            return float(np.sqrt(np.nanmean(sel ** 2))) if sel.size else float("nan")
        return float(np.sqrt(np.nanmean(diff ** 2)))

    # ponytail: joint EOF is T/S via destandardize, not a 'joint' profile key
    if joint_eof_meta is not None and list(outputs.keys()) == ["joint"]:
        from model.joint_eof import reconstruct_joint_eof

        meta = {k: joint_eof_meta[k] for k in ("T_mean", "T_std", "S_mean", "S_std", "n_lev")}
        pred_t, pred_s = reconstruct_joint_eof(pred_pcs, meta, pca_models["joint"])
        out = {}
        for name, pred_nz in (("temperature", pred_t), ("salinity", pred_s)):
            # reconstruct_joint_eof → (N, n_z); cache profiles → (n_depth, N)
            diff = pred_nz.T - true_profiles[name][:, idx]
            out[name] = _masked_rmse(diff)
        return out

    if decoders is not None:
        pred_t = torch.tensor(pred_pcs, dtype=torch.float32, device=device)
        inp_t = None
        if inputs is not None:
            inp_t = torch.tensor(inputs, dtype=torch.float32, device=device)
        pred = decode_latent_profiles(
            pred_t,
            decoders,
            outputs,
            inputs=inp_t,
            surface_residual_layout=surface_residual_layout,
        )
        out = {}
        for name in outputs:
            # decode_latent_profiles is (N, n_depth); cache profiles are (n_depth, N).
            diff = pred[name].detach().cpu().numpy() - true_profiles[name][:, idx].T
            out[name] = _masked_rmse(diff)
        return out

    pred = reconstruct_physical_profiles(
        pred_pcs, pca_models, outputs, clim_profiles=clim_profiles, indices=idx
    )
    out = {}
    for name in outputs:
        diff = pred[name] - true_profiles[name][:, idx]
        out[name] = _masked_rmse(diff)
    return out


def main(config, checkpoint_path: str, split: str = "test"):
    set_seed(config.config.get("seed", 42))
    ensure_cache(config)

    dl_args = dict(config["data_loader"]["args"])
    dl_args["input_params"] = config.config.get("input_params") or dl_args.get("input_params")
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
    assert_cache_checkpoint_pair(
        config.config.get("io", {}).get("dataset_tag"),
        data_loader.dataset_tag,
        ckpt.get("pca_models"),
        data_loader.pca_models,
    )
    outputs = OrderedDict(ckpt.get("outputs", dict(loss_outputs)))

    from types import SimpleNamespace

    loss_cfg = config.config.get("loss_config") or {}
    decoders = None
    if loss_cfg.get("mode") == "decoder":
        decoders = load_decoders_from_dir(loss_cfg["decoder_dir"], outputs, device)

    joint_eof_meta = data_loader.cache.get("joint_eof_meta")
    true_profiles = data_loader.cache.get("true_profiles")
    if true_profiles is None and data_loader.profiles:
        n = data_loader.cache["inputs"].shape[0]
        # joint_eof scores T/S profiles, not a 'joint' profile array
        profile_names = (
            ("temperature", "salinity")
            if joint_eof_meta is not None or loss_cfg.get("mode") in (*HEAVE_LOSS_MODES, "profile_direct")
            else tuple(k for k in outputs if k != "warp")
        )
        true_profiles = {}
        for name in profile_names:
            arr = np.asarray(data_loader.profiles[name], dtype=np.float32)
            if arr.shape[1] == n:
                arr = arr.T
            true_profiles[name] = arr

    from base.split_utils import build_split_indices as _eval_split

    _train_idx = _eval_split(
        data_loader.cache["inputs"].shape[0],
        data_loader.cache.get("JULD"),
        dl_args,
        dataset_tag=data_loader.dataset_tag,
        v2_src=dl_args.get("v2_src"),
    )["train"]
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
        true_profiles=(
            true_profiles
            if loss_cfg.get("mode") not in (*HEAVE_LOSS_MODES, "profile_direct")
            else data_loader.profiles
        ),
        ae_targets=data_loader.cache.get(target_key),
        ae_weights=data_loader.cache.get(weight_key),
        surface_residual_layout=surface_residual_layout_from_cache(data_loader.cache),
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
        joint_eof_meta=joint_eof_meta,
        train_idx=_train_idx,
        lat=data_loader.LAT,
        lon=data_loader.LON,
        profiles=data_loader.profiles,
        pres_levels=data_loader.PRES,
    )

    total_loss = 0.0
    pcs_list, idx_list = [], []
    with torch.no_grad():
        for data, target, indices in data_loader:
            data = data.to(device)
            target = target.to(device)
            indices = indices.to(device)
            output = model(data)
            total_loss += loss_fn(output, target, indices, inputs=data).item() * data.size(0)
            pcs_list.append(output.cpu().numpy())
            idx_list.append(indices.cpu().numpy())

    n = len(data_loader.dataset)
    pcs = np.vstack(pcs_list)
    indices = np.concatenate(idx_list)
    d_out = int(sum(outputs.values()))
    mu_pcs = pcs[:, :d_out] if pcs.shape[-1] == 2 * d_out else pcs
    tgt_pcs = data_loader.cache[target_key][indices].astype(np.float64)
    pca_tgt = data_loader.cache.get("pca_targets")
    if pca_tgt is not None:
        pca_tgt = pca_tgt[indices].astype(np.float64)

    if loss_cfg.get("mode") in (*HEAVE_LOSS_MODES, "profile_direct"):
        mu_t = torch.tensor(mu_pcs, dtype=torch.float32, device=device)
        idx_t = torch.tensor(indices, dtype=torch.long, device=device)
        n_all = data_loader.cache["inputs"].shape[0]

        def _sm(arr):
            a = np.asarray(arr, dtype=np.float64)
            return a if a.shape[0] == n_all else a.T

        T_true = _sm(data_loader.profiles["temperature"])[indices]
        S_true = _sm(data_loader.profiles["salinity"])[indices]
        if loss_cfg.get("mode") in HEAVE_LOSS_MODES:
            T_hat, S_hat = loss_fn.physical_ts(mu_t, idx_t)
            T_hat = T_hat.detach().cpu().numpy()
            S_hat = S_hat.detach().cpu().numpy()
            decode = loss_cfg.get("mode")
            latent_rmse = {"note": "z-PCA targets unused; residual PCs live on the warped grid"}
        else:
            n_z = mu_pcs.shape[1] // 2
            T_hat, S_hat = mu_pcs[:, :n_z], mu_pcs[:, n_z:]
            decode = "profile_direct"
            latent_rmse = {"note": "native-z T/S; PCA cache targets unused"}
        raw = {
            "temperature": float(np.sqrt(np.nanmean((T_hat - T_true) ** 2))),
            "salinity": float(np.sqrt(np.nanmean((S_hat - S_true) ** 2))),
        }
        report = {
            "checkpoint": str(checkpoint_path),
            "cache": str(data_loader.cache_path),
            "dataset_tag": data_loader.dataset_tag,
            "split": split,
            "n_samples": int(n),
            "loss": total_loss / n,
            "latent_target_rmse": latent_rmse,
            "raw_profile_rmse": raw,
            "decode": decode,
        }
    else:
        if loss_cfg.get("mode") == "decoder":
            latent_rmse = _latent_block_rmse(mu_pcs, tgt_pcs, outputs)
        else:
            latent_rmse = per_variable_rmse(mu_pcs, tgt_pcs, pca_models, outputs)

        report = {
            "checkpoint": str(checkpoint_path),
            "cache": str(data_loader.cache_path),
            "dataset_tag": data_loader.dataset_tag,
            "split": split,
            "n_samples": int(n),
            "loss": total_loss / n,
            "latent_target_rmse": latent_rmse,
            "raw_profile_rmse": raw_profile_rmse(
                mu_pcs,
                data_loader.profiles,
                pca_models,
                outputs,
                indices,
                decoders=decoders,
                device=device,
                inputs=data_loader.cache["inputs"][indices],
                surface_residual_layout=surface_residual_layout_from_cache(data_loader.cache),
                bottom_depth=data_loader.cache.get("bottom_depth"),
                pres_levels=data_loader.cache.get("PRES"),
                clim_profiles=data_loader.cache.get("clim_profiles"),
                joint_eof_meta=joint_eof_meta,
            ),
        }
    if loss_cfg.get("mode") in (*HEAVE_LOSS_MODES, "profile_direct"):
        report["pca_target_rmse"] = None
    elif loss_cfg.get("mode") != "decoder" and pca_tgt is not None:
        pca_outputs = OrderedDict(data_loader.cache["outputs"])
        report["pca_target_rmse"] = per_variable_rmse(mu_pcs, pca_tgt, pca_models, pca_outputs)
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
    config = ConfigParser(cfg, run_id="")
    report = main(config, args.resume, split=args.split)
    if args.out:
        write_report(report, args.out)
