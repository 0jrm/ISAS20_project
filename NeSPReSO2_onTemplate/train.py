import argparse
import collections
import pickle
from types import SimpleNamespace

import numpy as np
import torch

import data_loader.data_loaders as module_data
import model.metric as module_metric
import model.model as module_arch
from model.loss import make_loss, true_profiles_numpy
from parse_config import ConfigParser, validate_config
from playground import prepare_device
from playground.batch_size import resolve_batch_size
from playground.performance import apply_backend_settings, build_optimizer, get_performance_config, maybe_compile_model, maybe_compile_module
from preproc.preproc_isas_sat import (
    build_train_cache,
    compute_input_dim,
    count_encoding_dims,
    count_sat_vars,
    sat_patch_shape,
)
from trainer import Trainer


def set_seed(seed, performance=None):
    perf = performance or {}
    apply_backend_settings(
        {
            "cudnn_deterministic": perf.get("cudnn_deterministic", True),
            "cudnn_benchmark": perf.get("cudnn_benchmark", False),
            "matmul_precision": perf.get("matmul_precision"),
        },
        seed=seed,
    )


def ensure_cache(config):
    """Build or locate train-ready pickle; wire cache_path into data_loader args."""
    validate_config(config.config)
    pinned = config.config["data_loader"]["args"].get("cache_path") or None
    io_cfg = config.config.get("io", {})
    l3_cfg = io_cfg.get("l3") or {}
    input_params = config["input_params"]

    if l3_cfg.get("enabled"):
        from preproc.export_l3_cache import build_argo_l3_train_cache
        from preproc.l3_input import compute_l3_input_dim, l3_n_channels, l3_sat_patch_shape

        cache_path = build_argo_l3_train_cache(config.config)
        spatial_pad = int(io_cfg.get("spatial_pad", 0))
        temporal_pad = int(io_cfg.get("temporal_pad", 0))
        expected_dim = compute_l3_input_dim(input_params, l3_cfg)
    elif io_cfg.get("dataset_tag", "isas20") == "argo_v2":
        from preproc.export_v2_cache import build_argo_cache

        cache_path = build_argo_cache(config.config)
        spatial_pad = int(io_cfg.get("spatial_pad", 0))
        temporal_pad = int(io_cfg.get("temporal_pad", 0))
        expected_dim = compute_input_dim(input_params, spatial_pad, temporal_pad)
    else:
        cache_path = build_train_cache(config.config)
        spatial_pad = int(io_cfg.get("spatial_pad", 0))
        temporal_pad = int(io_cfg.get("temporal_pad", 0))
        expected_dim = compute_input_dim(input_params, spatial_pad, temporal_pad)

    if pinned:
        cache_path = pinned
    config.config["data_loader"]["args"]["cache_path"] = cache_path

    if config["arch"]["args"].get("input_dim") != expected_dim:
        config.config["arch"]["args"]["input_dim"] = expected_dim

    expected_out = sum(config["outputs"].values())
    if config["arch"]["args"].get("output_dim") != expected_out:
        config.config["arch"]["args"]["output_dim"] = expected_out

    arch_type = config["arch"]["type"]
    if arch_type == "PatchConvMLP":
        arch_args = config.config["arch"]["args"]
        arch_args["n_enc"] = count_encoding_dims(input_params)
        if l3_cfg.get("enabled"):
            patch_shape = l3_sat_patch_shape(l3_cfg)
            arch_args["n_sat"] = l3_n_channels(l3_cfg)
            arch_args["patch_shape"] = list(patch_shape)
        else:
            n_sat = count_sat_vars(input_params)
            patch_shape = sat_patch_shape(spatial_pad, temporal_pad, n_sat)
            arch_args["n_sat"] = n_sat
            arch_args["patch_shape"] = list(patch_shape) if patch_shape else None

    import pickle as _pickle

    with open(cache_path, "rb") as f:
        cache_dim = _pickle.load(f)["inputs"].shape[1]
    if cache_dim != expected_dim:
        raise ValueError(
            f"cache input dim {cache_dim} != expected {expected_dim} "
            f"(spatial_pad={spatial_pad}, temporal_pad={temporal_pad}, l3={l3_cfg.get('enabled')})"
        )

    split_seed = config.config.get("seed", 42)
    config.config["data_loader"]["args"]["split_seed"] = split_seed
    if io_cfg.get("v2_src") and not config.config["data_loader"]["args"].get("v2_src"):
        config.config["data_loader"]["args"]["v2_src"] = io_cfg["v2_src"]
    return cache_path


def _load_cache_tensors(cache_path: str, target_key: str = "targets"):
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    inputs = torch.tensor(cache["inputs"], dtype=torch.float32)
    if target_key not in cache:
        raise KeyError(f"cache missing {target_key!r}")
    targets = torch.tensor(cache[target_key], dtype=torch.float32)
    return cache, inputs, targets


def resolve_dataloader_batch_size(config, model, criterion, cache_path, device, logger):
    """Resolve ``batch_size``; ``0`` probes the largest value that fits (VRAM and train set)."""
    dl_args = config.config["data_loader"]["args"]
    configured = int(dl_args.get("batch_size", 512))
    safety = float(dl_args.get("batch_size_safety", 0.95))

    target_key = dl_args.get("target_key", "targets")

    cache, inputs, targets = _load_cache_tensors(cache_path, target_key)
    from base.split_utils import build_split_indices

    split_indices = build_split_indices(
        inputs.shape[0],
        cache.get("JULD"),
        dl_args,
        dataset_tag=cache.get("dataset_tag", "unknown"),
        v2_src=dl_args.get("v2_src"),
    )
    n_train = max(1, len(split_indices["train"]))

    optimizer_factory = lambda: build_optimizer(
        config.config,
        model.parameters(),
        fused=bool(get_performance_config(config.config).get("fused_optimizer")),
        device=device,
    )

    resolved = resolve_batch_size(
        configured,
        n_train,
        model,
        criterion,
        inputs,
        targets,
        device,
        safety_fraction=safety,
        optimizer_factory=optimizer_factory,
    )
    if resolved != configured:
        logger.info(
            "Resolved batch_size=%s (configured=%s, n_train=%s, safety=%s)",
            resolved,
            configured,
            n_train,
            safety,
        )
    config.config["data_loader"]["args"]["batch_size"] = resolved
    return resolved


def surface_residual_layout_from_cache(cache: dict) -> dict | None:
    if not cache.get("input_params"):
        return None
    return {
        "spatial_pad": int(cache.get("spatial_pad", 0)),
        "temporal_pad": int(cache.get("temporal_pad", 0)),
        "input_params": cache["input_params"],
        "n_enc": count_encoding_dims(cache["input_params"]),
    }


def main(config):
    logger = config.get_logger("train")
    performance = get_performance_config(config.config)
    set_seed(config.config.get("seed", 123), performance=performance)

    ensure_cache(config)
    cache_path = config.config["data_loader"]["args"]["cache_path"]

    device, device_ids = prepare_device(config["n_gpu"])

    model = config.init_obj("arch", module_arch)
    logger.info(model)
    model = model.to(device)
    if performance.get("compile"):
        model = maybe_compile_model(model, True)
        logger.info("torch.compile enabled")
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    cache, _, _ = _load_cache_tensors(cache_path)
    dl_args = config.config["data_loader"]["args"]
    target_key = dl_args.get("target_key", "targets")
    weight_key = dl_args.get("weight_key", "weights")
    loss_outputs = collections.OrderedDict(config["outputs"])
    loss_cfg = config.config.get("loss_config") or {}
    ae_targets = cache.get(target_key) if loss_cfg.get("mode") == "decoder" else cache.get("ae_targets")
    ae_weights = cache.get(weight_key) if loss_cfg.get("mode") == "decoder" else cache.get("ae_weights")
    true_profiles = cache.get("true_profiles")
    if true_profiles is None and loss_cfg.get("mode") == "decoder":
        true_profiles = true_profiles_numpy(
            cache["targets"], cache["pca_models"], collections.OrderedDict(cache["outputs"])
        )
    density_meta = SimpleNamespace(
        LAT=cache["LAT"],
        LON=cache["LON"],
        PRES=cache.get("PRES"),
        min_depth=cache.get("min_depth", 0),
        max_depth=cache.get("max_depth", cache["targets"].shape[1] - 1 if hasattr(cache["targets"], "shape") else 0),
    )
    criterion = make_loss(
        pca_models=cache["pca_models"],
        outputs=loss_outputs,
        weights=cache.get(weight_key, cache["weights"]),
        device=device,
        density_config=config.config.get("density"),
        density_meta=density_meta,
        loss_scales=config.config.get("loss_scales"),
        loss_config=loss_cfg,
        targets=cache["targets"],
        true_profiles=true_profiles,
        ae_targets=ae_targets,
        ae_weights=ae_weights,
        surface_residual_layout=surface_residual_layout_from_cache(cache),
    )
    if performance.get("compile_loss"):
        criterion = maybe_compile_module(criterion, True)
        logger.info("torch.compile enabled on loss")

    resolve_dataloader_batch_size(config, model, criterion, cache_path, device, logger)

    data_loader = config.init_obj("data_loader", module_data)
    valid_data_loader = data_loader.split_validation()

    metrics = [getattr(module_metric, met) for met in config["metrics"]]

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = build_optimizer(
        config.config,
        trainable_params,
        fused=bool(performance.get("fused_optimizer")),
        device=device,
    )
    lr_scheduler = config.init_obj("lr_scheduler", torch.optim.lr_scheduler, optimizer)

    checkpoint_extra = {
        "pca_models": data_loader.pca_models,
        "input_params": data_loader.input_params,
        "outputs": dict(loss_outputs),
    }

    trainer = Trainer(
        model,
        criterion,
        metrics,
        optimizer,
        config=config,
        device=device,
        data_loader=data_loader,
        valid_data_loader=valid_data_loader,
        lr_scheduler=lr_scheduler,
        checkpoint_extra=checkpoint_extra,
        performance=performance,
    )

    trainer.train()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="NeSPReSO v2 training")
    args.add_argument("-c", "--config", default=None, type=str, help="config file path")
    args.add_argument("-r", "--resume", default=None, type=str, help="checkpoint path")
    args.add_argument("-d", "--device", default=None, type=str, help="GPU indices")
    args.add_argument("-id", "--run-id", default=None, type=str, help="checkpoint/log subdir name")

    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["--lr", "--learning_rate"], type=float, target="optimizer;args;lr"),
        CustomArgs(["--bs", "--batch_size"], type=int, target="data_loader;args;batch_size"),
        CustomArgs(["--log-interval", "--log_interval"], type=int, target="trainer;log_interval"),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
