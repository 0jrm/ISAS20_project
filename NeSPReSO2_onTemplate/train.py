import argparse
import collections
from types import SimpleNamespace

import numpy as np
import torch

import data_loader.data_loaders as module_data
import model.metric as module_metric
import model.model as module_arch
from model.loss import make_loss
from parse_config import ConfigParser, validate_config
from playground import prepare_device
from playground.performance import apply_backend_settings, build_optimizer, get_performance_config, maybe_compile_model
from preproc.preproc_isas_sat import build_train_cache, compute_input_dim
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
    io_cfg = config.config.get("io", {})
    if io_cfg.get("dataset_tag", "isas20") == "argo_v2":
        from preproc.export_v2_cache import build_argo_cache

        cache_path = build_argo_cache(config.config)
    else:
        cache_path = build_train_cache(config.config)
    config.config["data_loader"]["args"]["cache_path"] = cache_path
    expected_dim = compute_input_dim(config["input_params"])
    if config["arch"]["args"].get("input_dim") != expected_dim:
        config.config["arch"]["args"]["input_dim"] = expected_dim
    split_seed = config.config.get("seed", 42)
    config.config["data_loader"]["args"]["split_seed"] = split_seed
    return cache_path


def main(config):
    logger = config.get_logger("train")
    performance = get_performance_config(config.config)
    set_seed(config.config.get("seed", 123), performance=performance)

    ensure_cache(config)

    data_loader = config.init_obj("data_loader", module_data)
    valid_data_loader = data_loader.split_validation()

    model = config.init_obj("arch", module_arch)
    logger.info(model)

    device, device_ids = prepare_device(config["n_gpu"])
    model = model.to(device)
    if performance.get("compile"):
        model = maybe_compile_model(model, True)
        logger.info("torch.compile enabled")
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    density_meta = SimpleNamespace(
        LAT=data_loader.LAT,
        LON=data_loader.LON,
        PRES=data_loader.PRES,
        min_depth=data_loader.min_depth,
        max_depth=data_loader.max_depth,
    )
    criterion = make_loss(
        pca_models=data_loader.pca_models,
        outputs=data_loader.outputs,
        weights=data_loader.weights,
        device=device,
        density_config=config.config.get("density"),
        density_meta=density_meta,
    )

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
        "outputs": dict(data_loader.outputs),
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
