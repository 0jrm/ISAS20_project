"""Backend performance knobs for training and benchmarks."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

DEFAULT_PERFORMANCE: dict[str, Any] = {
    "cudnn_benchmark": False,
    "cudnn_deterministic": True,
    "matmul_precision": None,
    "autocast": False,
    "autocast_dtype": "bfloat16",
    "compile": False,
    "compile_loss": False,
    "fused_optimizer": False,
}

VALID_MATMUL_PRECISION = {None, "highest", "high", "medium"}
VALID_AUTOCAST_DTYPES = {"bfloat16", "float16"}


@dataclass
class VariantSpec:
    name: str
    cudnn_benchmark: bool = False
    cudnn_deterministic: bool = True
    matmul_precision: str | None = None
    autocast: bool = False
    autocast_dtype: str = "bfloat16"
    compile_model: bool = False
    compile_loss: bool = False
    fused_optimizer: bool = False
    loss_mode: str | None = None
    batch_size: int | None = None
    notes: str = ""


BENCHMARK_VARIANTS: dict[str, VariantSpec] = {
    "baseline": VariantSpec("baseline", notes="current train.py defaults"),
    "cudnn_benchmark": VariantSpec("cudnn_benchmark", cudnn_benchmark=True, cudnn_deterministic=False),
    "matmul_high": VariantSpec("matmul_high", matmul_precision="high"),
    "matmul_highest": VariantSpec("matmul_highest", matmul_precision="highest"),
    "matmul_medium": VariantSpec("matmul_medium", matmul_precision="medium"),
    "autocast_bf16": VariantSpec("autocast_bf16", autocast=True, autocast_dtype="bfloat16"),
    "autocast_fp16": VariantSpec("autocast_fp16", autocast=True, autocast_dtype="float16"),
    "compile_model": VariantSpec("compile_model", compile_model=True),
    "compile_loss": VariantSpec("compile_loss", compile_loss=True, notes="torch.compile on CombinedPCALoss"),
    "compile_both": VariantSpec(
        "compile_both",
        compile_model=True,
        compile_loss=True,
        notes="torch.compile on model + loss",
    ),
    "fused_adam": VariantSpec("fused_adam", fused_optimizer=True),
    "combo_best": VariantSpec(
        "combo_best",
        cudnn_benchmark=True,
        cudnn_deterministic=False,
        matmul_precision="high",
        autocast=True,
        autocast_dtype="bfloat16",
        fused_optimizer=True,
        notes="stack cudnn_benchmark + matmul_high + autocast_bf16 + fused_adam",
    ),
    "combo_phase4b_all": VariantSpec(
        "combo_phase4b_all",
        compile_model=True,
        compile_loss=True,
        loss_mode="pred_profile_cached",
        batch_size=0,
        notes="Phase 4b stack: max batch + compile model+loss + pred_profile_cached",
    ),
    "decoder_loss": VariantSpec(
        "decoder_loss",
        loss_mode="decoder",
        notes="Phase 5: frozen AE decoder profile loss (requires ae_targets + decoder_dir in config)",
    ),
}


def get_performance_config(config: dict[str, Any]) -> dict[str, Any]:
    perf = dict(DEFAULT_PERFORMANCE)
    perf.update(config.get("performance") or {})
    return perf


def validate_performance_config(perf: dict[str, Any]) -> None:
    if perf.get("matmul_precision") not in VALID_MATMUL_PRECISION:
        raise ValueError(f"performance.matmul_precision must be one of {VALID_MATMUL_PRECISION}")
    if perf.get("autocast_dtype") not in VALID_AUTOCAST_DTYPES:
        raise ValueError(f"performance.autocast_dtype must be one of {VALID_AUTOCAST_DTYPES}")


def variant_to_performance(spec: VariantSpec) -> dict[str, Any]:
    return {
        "cudnn_benchmark": spec.cudnn_benchmark,
        "cudnn_deterministic": spec.cudnn_deterministic,
        "matmul_precision": spec.matmul_precision,
        "autocast": spec.autocast,
        "autocast_dtype": spec.autocast_dtype,
        "compile": spec.compile_model,
        "compile_loss": spec.compile_loss,
        "fused_optimizer": spec.fused_optimizer,
        "loss_mode": spec.loss_mode,
        "batch_size": spec.batch_size,
    }


def apply_backend_settings(perf: dict[str, Any], *, seed: int | None = None) -> None:
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    torch.backends.cudnn.deterministic = bool(perf.get("cudnn_deterministic", True))
    torch.backends.cudnn.benchmark = bool(perf.get("cudnn_benchmark", False))

    precision = perf.get("matmul_precision")
    if precision is not None:
        torch.set_float32_matmul_precision(precision)


def autocast_dtype_from_name(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    return torch.bfloat16


def maybe_compile_model(model: torch.nn.Module, enabled: bool) -> torch.nn.Module:
    return maybe_compile_module(model, enabled)


def maybe_compile_module(module: torch.nn.Module, enabled: bool) -> torch.nn.Module:
    if not enabled:
        return module
    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile requires PyTorch 2.0+")
    return torch.compile(module)


def build_optimizer(
    config,
    params,
    *,
    fused: bool = False,
    device: torch.device | None = None,
) -> torch.optim.Optimizer:
    opt_cfg = copy.deepcopy(config["optimizer"])
    args = dict(opt_cfg["args"])
    if fused and device is not None and device.type == "cuda":
        args["fused"] = True
    opt_type = opt_cfg["type"]
    return getattr(torch.optim, opt_type)(params, **args)
