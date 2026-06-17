#!/usr/bin/env python3
"""Benchmark ML optimization variants on NeSPReSO training steps."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import data_loader.data_loaders as module_data
import model.model as module_arch
from data_loader.data_loaders import _collate_with_index
from model.loss import make_loss
from parse_config import ConfigParser, validate_config
from playground.performance import (
    BENCHMARK_VARIANTS,
    VariantSpec,
    apply_backend_settings,
    autocast_dtype_from_name,
    build_optimizer,
    maybe_compile_model,
    variant_to_performance,
)
from preproc.preproc_isas_sat import build_train_cache, compute_input_dim
from train import ensure_cache

WARMUP_EPOCHS = 10
TIMED_EPOCHS = 100
SINGLE_GPU_VARIANTS = list(BENCHMARK_VARIANTS.keys())


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _init_distributed() -> tuple[int, int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), int(os.environ.get("LOCAL_RANK", 0)), dist.get_world_size()
    if "RANK" in os.environ:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29501")
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    return 0, 0, 1


def _cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _make_train_loader(data_loader, *, sampler=None):
    return DataLoader(
        data_loader.dataset,
        batch_size=data_loader.batch_size,
        shuffle=sampler is None and data_loader._split == "train",
        sampler=sampler,
        num_workers=data_loader.num_workers,
        collate_fn=_collate_with_index,
        pin_memory=data_loader.pin_memory,
    )


def _attach_loader_metadata(dst, src):
    for attr in (
        "pca_models",
        "outputs",
        "weights",
        "LAT",
        "LON",
        "PRES",
        "profiles",
        "input_params",
        "dataset_tag",
        "min_depth",
        "max_depth",
        "cache_path",
        "train_subset",
        "val_subset",
        "test_subset",
        "split_indices",
    ):
        if hasattr(src, attr):
            setattr(dst, attr, getattr(src, attr))
    dst._split = getattr(src, "_split", "train")
    return dst


def build_stack(config, device, *, rank: int = 0, world_size: int = 1):
    ensure_cache(config)

    data_loader = config.init_obj("data_loader", module_data)
    valid_data_loader = data_loader.split_validation()

    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(data_loader.dataset, shuffle=True)
    train_loader = _make_train_loader(data_loader, sampler=sampler)
    _attach_loader_metadata(train_loader, data_loader)

    model = config.init_obj("arch", module_arch).to(device)
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
    return model, criterion, train_loader, valid_data_loader, sampler


def _train_step(model, criterion, optimizer, batch, device, perf, scaler):
    data, target, indices = batch
    data = data.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)
    indices = indices.to(device, non_blocking=True)

    optimizer.zero_grad(set_to_none=True)
    dtype = autocast_dtype_from_name(perf.get("autocast_dtype", "bfloat16"))
    with torch.autocast(device_type=device.type, dtype=dtype, enabled=bool(perf.get("autocast"))):
        output = model(data)
        loss = criterion(output, target, indices)

    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()
    return float(loss.item())


def run_epochs(
    model,
    criterion,
    train_loader,
    optimizer,
    device,
    perf,
    *,
    epochs: int,
    sampler=None,
    timed: bool = False,
) -> tuple[float | None, float]:
    scaler = None
    if perf.get("autocast") and perf.get("autocast_dtype") == "float16" and device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda")

    total_time = 0.0
    last_loss = 0.0
    epoch_times: list[float] = []

    model.train()
    for epoch in range(epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        if timed:
            _sync(device)
            t0 = time.perf_counter()

        epoch_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            last_loss = _train_step(model, criterion, optimizer, batch, device, perf, scaler)
            epoch_loss += last_loss
            n_batches += 1

        if timed:
            _sync(device)
            epoch_times.append(time.perf_counter() - t0)
            total_time += epoch_times[-1]

    sec_per_epoch = (total_time / epochs) if timed and epochs > 0 else None
    avg_train_loss = epoch_loss / max(n_batches, 1)
    return sec_per_epoch, avg_train_loss


@torch.no_grad()
def eval_val_loss(model, criterion, valid_loader, device, perf) -> float:
    model.eval()
    total = 0.0
    n_batches = 0
    dtype = autocast_dtype_from_name(perf.get("autocast_dtype", "bfloat16"))
    for data, target, indices in valid_loader:
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        indices = indices.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=bool(perf.get("autocast"))):
            output = model(data)
            loss = criterion(output, target, indices)
        total += float(loss.item())
        n_batches += 1
    return total / max(n_batches, 1)


def profile_top_ops(model, criterion, train_loader, device, perf, steps: int = 5) -> list[dict]:
    if device.type != "cuda":
        return []

    batch = next(iter(train_loader))
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    dtype = autocast_dtype_from_name(perf.get("autocast_dtype", "bfloat16"))

    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities, record_shapes=False) as prof:
        for _ in range(steps):
            data, target, indices = batch
            data = data.to(device)
            target = target.to(device)
            indices = indices.to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=bool(perf.get("autocast"))):
                output = model(data)
                loss = criterion(output, target, indices)
            loss.backward()
            optimizer.step()

    events = prof.key_averages().table(sort_by="cuda_time_total", row_limit=10)
    rows = []
    for line in events.splitlines():
        if "cuda_time" in line or line.startswith("-"):
            continue
        parts = [p for p in line.split(" ") if p]
        if len(parts) >= 2:
            rows.append({"op": parts[0], "line": line.strip()})
        if len(rows) >= 5:
            break
    return rows


def run_variant(
    config,
    spec: VariantSpec,
    *,
    device: torch.device,
    rank: int,
    world_size: int,
    seed: int,
    profile: bool = False,
) -> dict:
    perf = variant_to_performance(spec)
    apply_backend_settings(perf, seed=seed)

    model, criterion, train_loader, valid_loader, sampler = build_stack(
        config, device, rank=rank, world_size=world_size
    )
    if perf.get("compile"):
        model = maybe_compile_model(model, True)

    optimizer = build_optimizer(
        config.config,
        filter(lambda p: p.requires_grad, model.parameters()),
        fused=bool(perf.get("fused_optimizer")),
        device=device,
    )

    if world_size > 1:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    run_epochs(model, criterion, train_loader, optimizer, device, perf, epochs=WARMUP_EPOCHS, sampler=sampler)
    sec_per_epoch, train_loss = run_epochs(
        model,
        criterion,
        train_loader,
        optimizer,
        device,
        perf,
        epochs=TIMED_EPOCHS,
        sampler=sampler,
        timed=True,
    )
    val_loss = eval_val_loss(model, criterion, valid_loader, device, perf)

    result = {
        "variant": spec.name,
        "sec_per_epoch": sec_per_epoch,
        "final_train_loss": train_loss,
        "final_val_loss": val_loss,
        "notes": spec.notes,
        "settings": perf,
        "world_size": world_size,
    }
    if profile and rank == 0:
        raw_model = model.module if hasattr(model, "module") else model
        result["profile_top_ops"] = profile_top_ops(raw_model, criterion, train_loader, device, perf)
    return result


def run_ddp_variant(config, seed: int, rank: int, local_rank: int, world_size: int) -> dict:
    device = torch.device(f"cuda:{local_rank}")
    spec = VariantSpec("ddp_2gpu", notes=f"DDP over {world_size} GPUs")
    try:
        result = run_variant(config, spec, device=device, rank=rank, world_size=world_size, seed=seed)
        if rank == 0:
            return result
        return {}
    finally:
        pass


def print_table(results: list[dict], baseline_sec: float | None) -> None:
    header = f"{'variant':<16} {'sec/epoch':>10} {'speedup':>8} {'val_loss':>12} notes"
    print(header)
    print("-" * len(header))
    for row in results:
        sec = row.get("sec_per_epoch")
        speedup = (baseline_sec / sec) if baseline_sec and sec else None
        speedup_s = f"{speedup:.2f}x" if speedup is not None else "n/a"
        sec_s = f"{sec:.4f}" if sec is not None else "n/a"
        val_s = f"{row.get('final_val_loss', 0):.6f}"
        notes = row.get("notes", "")
        print(f"{row['variant']:<16} {sec_s:>10} {speedup_s:>8} {val_s:>12} {notes}")


def main():
    global WARMUP_EPOCHS, TIMED_EPOCHS

    parser = argparse.ArgumentParser(description="Benchmark ML optimization variants")
    parser.add_argument("-c", "--config", default="config_isas.json", type=str)
    parser.add_argument("--variant", default="all", type=str, help="all | ddp | <variant_name>")
    parser.add_argument("--warmup-epochs", type=int, default=WARMUP_EPOCHS)
    parser.add_argument("--timed-epochs", type=int, default=TIMED_EPOCHS)
    parser.add_argument("--profile", action="store_true", help="Collect torch.profiler top ops for baseline/combo")
    parser.add_argument("--out", default=None, type=str, help="JSON output path")
    parser.add_argument("-d", "--device", default=None, type=str, help="CUDA_VISIBLE_DEVICES")
    args = parser.parse_args()

    WARMUP_EPOCHS = args.warmup_epochs
    TIMED_EPOCHS = args.timed_epochs

    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config_dict = json.loads(config_path.read_text())
    validate_config(config_dict)
    bench_run_id = ""
    config = ConfigParser(config_dict, run_id=bench_run_id)

    seed = int(config.config.get("seed", 42))
    rank, local_rank, world_size = _init_distributed()
    is_ddp_job = world_size > 1

    if args.variant == "ddp" or (is_ddp_job and args.variant == "all"):
        result = run_ddp_variant(config, seed, rank, local_rank, world_size)
        if dist.is_initialized():
            dist.barrier()
        if rank == 0 and result:
            out = {
                "config": str(config_path),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "warmup_epochs": WARMUP_EPOCHS,
                "timed_epochs": TIMED_EPOCHS,
                "results": [result],
            }
            print_table(out["results"], None)
            if args.out:
                Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
        _cleanup_distributed()
        return

    if is_ddp_job:
        raise SystemExit("Single-GPU variants require one process; use --variant ddp with torchrun.")

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for ML optimization benchmarks.")

    device = torch.device("cuda:0")
    variants = SINGLE_GPU_VARIANTS if args.variant == "all" else [args.variant]
    results: list[dict] = []

    for name in variants:
        if name not in BENCHMARK_VARIANTS:
            raise SystemExit(f"Unknown variant: {name}. Choices: {', '.join(BENCHMARK_VARIANTS)}")
        spec = BENCHMARK_VARIANTS[name]
        print(f"Running {name}...", flush=True)
        do_profile = args.profile and name in {"baseline", "combo_best"}
        row = run_variant(config, spec, device=device, rank=0, world_size=1, seed=seed, profile=do_profile)
        results.append(row)

    baseline_sec = next((r["sec_per_epoch"] for r in results if r["variant"] == "baseline"), None)
    print()
    print_table(results, baseline_sec)

    for row in results:
        if baseline_sec and row["variant"] != "baseline":
            ref_val = next(r["final_val_loss"] for r in results if r["variant"] == "baseline")
            if ref_val > 0:
                drift = abs(row["final_val_loss"] - ref_val) / ref_val
                row["val_loss_drift_pct"] = round(100.0 * drift, 3)
                if drift > 0.01:
                    row["notes"] = (row.get("notes", "") + " val_loss_drift>1%").strip()

    out = {
        "config": str(config_path),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "warmup_epochs": WARMUP_EPOCHS,
        "timed_epochs": TIMED_EPOCHS,
        "baseline_sec_per_epoch": baseline_sec,
        "results": results,
    }
    out_path = Path(args.out) if args.out else ROOT / "saved" / "benchmarks" / f"ml_opts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
