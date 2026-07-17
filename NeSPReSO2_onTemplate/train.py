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
from base.util import prepare_device
from base.batch_size import resolve_batch_size
from base.performance import apply_backend_settings, build_optimizer, get_performance_config, maybe_compile_model, maybe_compile_module
from preproc.preproc_isas_sat import (
    build_train_cache,
    count_encoding_dims,
    count_scalar_dims,
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


def _init_density_spice_clim_bias(model, config) -> None:
    """Zero δa bias; optional density weight shrink via trainer.density_weight_scale."""
    if not hasattr(model, "mu_out"):
        return
    k = int(config.config["outputs"]["density_ctrl"])
    scale = float((config.config.get("trainer") or {}).get("density_weight_scale", 1.0))
    with torch.no_grad():
        model.mu_out.bias[:k].zero_()
        if scale != 1.0:
            model.mu_out.weight[:k].mul_(scale)
        if getattr(model, "sigma_out", None) is not None:
            if model.sigma_out.bias is not None:
                model.sigma_out.bias[:k].zero_()
            if scale != 1.0:
                model.sigma_out.weight[:k].mul_(scale)
    print(f"density_spice residual-δa init (density_weight_scale={scale})", flush=True)


def ensure_cache(config):
    """Build or locate train-ready pickle; wire cache_path into data_loader args."""
    validate_config(config.config)
    pinned = config.config["data_loader"]["args"].get("cache_path") or None
    io_cfg = config.config.get("io", {})
    l3_cfg = io_cfg.get("l3") or {}

    if l3_cfg.get("enabled"):
        from preproc.export_l3_cache import build_argo_l3_train_cache

        cache_path = build_argo_l3_train_cache(config.config)
    elif io_cfg.get("dataset_tag", "isas20") == "argo_field":
        from preproc.export_field_cache import build_field_cache

        cache_path = build_field_cache(config.config)
    elif io_cfg.get("dataset_tag", "isas20") == "argo_l4":
        from preproc.export_argo_l4_cache import build_argo_l4_cache

        cache_path = build_argo_l4_cache(config.config)
    elif io_cfg.get("dataset_tag", "isas20") == "argo_residual":
        from preproc.export_argo_residual_cache import build_argo_residual_cache

        cache_path = build_argo_residual_cache(config.config)
    elif io_cfg.get("dataset_tag", "isas20") == "argo_v2":
        from preproc.export_v2_cache import build_argo_cache

        cache_path = build_argo_cache(config.config)
    elif io_cfg.get("dataset_tag", "isas20") == "argo_cube":
        from preproc.features.export_feature_cache import build_feature_cache, build_point_cube_cache

        if io_cfg.get("cache_kind") == "point_cube":
            cache_path = build_point_cube_cache(config.config)
        else:
            cache_path = build_feature_cache(config.config)
    else:
        cache_path = build_train_cache(config.config)

    if pinned:
        cache_path = pinned
    config.config["data_loader"]["args"]["cache_path"] = cache_path
    if io_cfg.get("use_error_channels"):
        config.config["data_loader"]["args"]["use_error_channels"] = True

    from preproc.l3_input import sync_arch_with_io, verify_l3_cache_layout

    expected_dim = sync_arch_with_io(config.config)

    import pickle as _pickle

    with open(cache_path, "rb") as f:
        cache = _pickle.load(f)
    if l3_cfg.get("enabled"):
        verify_l3_cache_layout(cache, l3_cfg)
    elif cache.get("dataset_tag") == "argo_field":
        fields = cache.get("fields")
        if fields is None or np.asarray(fields).ndim != 4:
            raise ValueError("argo_field cache must contain fields with ndim==4")
    else:
        cache_dim = cache["inputs"].shape[1]
        if cache_dim != expected_dim:
            raise ValueError(
                f"cache input dim {cache_dim} != expected {expected_dim} "
                f"(l3={l3_cfg.get('enabled')}, spatial_pad={io_cfg.get('spatial_pad')}, "
                f"temporal_pad={io_cfg.get('temporal_pad')})"
            )

    steric_cfg = config.config.get("steric") or {}
    if steric_cfg.get("enabled"):
        for key in ("ssh_obs_sla", "clim_steric", "steric_calibration"):
            if key not in cache:
                raise ValueError(f"steric.enabled requires cache key {key!r}")
        r_train = float((cache["steric_calibration"] or {}).get("r_train", float("nan")))
        min_r = float(steric_cfg.get("min_calibration_r", 0.5))
        if not np.isfinite(r_train) or r_train < min_r:
            raise ValueError(
                f"steric calibration r_train={r_train:.3f} < {min_r} — loss is mis-specified; "
                "fix the calibration (or lower steric.min_calibration_r) before enabling steric"
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
    if cache.get("dataset_tag") == "argo_field":
        if configured <= 0:
            configured = 4
        config.config["data_loader"]["args"]["batch_size"] = configured
        return configured

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
        "n_enc": count_scalar_dims(cache["input_params"]),
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
    arch_args = config.config.get("arch", {}).get("args", {})
    if hasattr(model, "set_freeze_base") and arch_args.get("freeze_base"):
        model.set_freeze_base(True)
        logger.info("Residual base encoder frozen (freeze_base=true)")
    loss_cfg = config.config.get("loss_config") or {}
    if hasattr(model, "set_sigma_trainable") and loss_cfg.get("freeze_sigma"):
        # Keep σ params in the optimizer (resume-safe). Loss ignores σ when freeze_sigma.
        logger.info("Phase 4 stage-1: σ ignored by loss (params stay in optimizer for stage-2 resume)")
    model = model.to(device)
    # density_spice: random a → softplus/cumsum explodes; bias-init μ to clim σ₀ encode.
    if (
        (config.config.get("io") or {}).get("representation") == "density_spice"
        or loss_cfg.get("mode") == "density_spice"
    ):
        _init_density_spice_clim_bias(model, config)
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
    steric_meta = SimpleNamespace(
        LAT=cache["LAT"],
        LON=cache["LON"],
        PRES=cache.get("PRES"),
        ssh_obs_sla=cache.get("ssh_obs_sla"),
        clim_steric=cache.get("clim_steric"),
        steric_calibration=cache.get("steric_calibration"),
    )
    criterion = make_loss(
        pca_models=cache["pca_models"],
        outputs=loss_outputs,
        weights=cache.get(weight_key, cache["weights"]),
        device=device,
        density_config=config.config.get("density"),
        density_meta=density_meta,
        steric_config=config.config.get("steric"),
        steric_meta=steric_meta,
        clim_profiles=cache.get("clim_profiles"),
        loss_scales=config.config.get("loss_scales"),
        loss_config=loss_cfg,
        targets=cache["targets"],
        true_profiles=true_profiles,
        ae_targets=ae_targets,
        ae_weights=ae_weights,
        surface_residual_layout=surface_residual_layout_from_cache(cache),
        bottom_depth=cache.get("bottom_depth"),
        pres_levels=cache.get("PRES"),
        density_spice_meta=cache.get("density_spice_meta"),
        joint_eof_meta=cache.get("joint_eof_meta"),
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
    if data_loader.l3_enabled:
        checkpoint_extra["l3_enabled"] = True
        checkpoint_extra["l3_channel_metadata"] = data_loader.l3_channel_metadata
    if data_loader.sat_patch_shape:
        checkpoint_extra["sat_patch_shape"] = data_loader.sat_patch_shape

    if cache.get("dataset_tag") == "argo_field":
        from trainer.field_trainer import FieldTrainer

        trainer = FieldTrainer(
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
    else:
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
