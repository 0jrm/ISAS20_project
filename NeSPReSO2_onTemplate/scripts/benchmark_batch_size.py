#!/usr/bin/env python3
"""Sweep batch sizes: VRAM fit, throughput, and recommended settings."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import model.model as module_arch
from model.loss import make_loss
from parse_config import ConfigParser, validate_config
from playground.batch_size import (
    default_sweep_sizes,
    measure_throughput,
    pick_best_throughput,
    probe_max_batch_size,
    train_samples_from_cache,
)
from playground.performance import apply_backend_settings, build_optimizer, get_performance_config, maybe_compile_model
from train import ensure_cache


def _load_tensors(cache_path: str) -> tuple[torch.Tensor, torch.Tensor, int]:
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    inputs = torch.tensor(cache["inputs"], dtype=torch.float32)
    targets = torch.tensor(cache["targets"], dtype=torch.float32)
    return inputs, targets, inputs.shape[0]


def _build_stack(config: ConfigParser, device: torch.device):
    performance = get_performance_config(config.config)
    apply_backend_settings(performance, seed=int(config.config.get("seed", 42)))

    cache_path = ensure_cache(config)
    inputs, targets, n_total = _load_tensors(cache_path)
    dl_args = config.config["data_loader"]["args"]
    n_train = train_samples_from_cache(n_total, float(dl_args.get("train_frac", 0.7)))

    model = config.init_obj("arch", module_arch).to(device)
    if performance.get("compile"):
        model = maybe_compile_model(model, True)

    with open(cache_path, "rb") as f:
        cache = pickle.load(f)

    density_meta = SimpleNamespace(
        LAT=cache["LAT"],
        LON=cache["LON"],
        PRES=cache.get("PRES"),
        min_depth=cache.get("min_depth", 0),
        max_depth=cache.get("max_depth", targets.shape[1] - 1),
    )
    criterion = make_loss(
        pca_models=cache["pca_models"],
        outputs=cache["outputs"],
        weights=cache["weights"],
        device=device,
        density_config=config.config.get("density"),
        density_meta=density_meta,
        loss_scales=config.config.get("loss_scales"),
        loss_config=config.config.get("loss_config"),
        targets=cache["targets"],
        true_profiles=cache.get("true_profiles"),
    )

    optimizer_factory = lambda: build_optimizer(
        config.config,
        model.parameters(),
        fused=bool(performance.get("fused_optimizer")),
        device=device,
    )
    return model, criterion, inputs, targets, n_train, optimizer_factory, performance


def print_table(rows: list[dict], max_bs: int, best_bs: int) -> None:
    header = (
        f"{'batch':>8} {'samples/s':>12} {'ms/step':>10} {'peak_MiB':>10} "
        f"{'gpu_util':>9} note"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        note = []
        if row["batch_size"] == max_bs:
            note.append("max_fit")
        if row["batch_size"] == best_bs:
            note.append("best_throughput")
        ms = 1000.0 * row["sec_per_step"]
        print(
            f"{row['batch_size']:>8} {row['samples_per_sec']:>12.1f} {ms:>10.2f} "
            f"{row['peak_mem_mib']:>10.1f} {row.get('gpu_util_pct', 'n/a'):>9} "
            f"{' '.join(note)}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark batch sizes for GPU throughput")
    parser.add_argument("-c", "--config", default="config_isas_patch.json", type=str)
    parser.add_argument(
        "--sizes",
        default="",
        help="Comma-separated batch sizes to test (default: powers of 2 up to max fit)",
    )
    parser.add_argument("--steps", type=int, default=30, help="Timed steps per batch size")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup steps per batch size")
    parser.add_argument("--safety", type=float, default=0.95, help="Safety fraction for max-fit probe")
    parser.add_argument("--out", default=None, type=str, help="JSON output path")
    parser.add_argument("-d", "--device", default=None, type=str, help="CUDA_VISIBLE_DEVICES")
    args = parser.parse_args()

    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for batch-size benchmarks.")

    device = torch.device("cuda:0")
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path

    config_dict = json.loads(config_path.read_text())
    validate_config(config_dict)
    config = ConfigParser(config_dict, run_id="")

    model, criterion, inputs, targets, n_train, optimizer_factory, performance = _build_stack(
        config, device
    )

    max_fit = probe_max_batch_size(
        model,
        criterion,
        inputs,
        targets,
        device,
        n_train=n_train,
        safety_fraction=args.safety,
        optimizer_factory=optimizer_factory,
    )

    if args.sizes.strip():
        sweep = sorted({int(x) for x in args.sizes.split(",") if x.strip()})
        sweep = [min(s, n_train) for s in sweep]
    else:
        sweep = default_sweep_sizes(n_train, max_fit)

    rows: list[dict] = []
    for bs in sweep:
        if bs > max_fit:
            rows.append(
                {
                    "batch_size": bs,
                    "sec_per_step": None,
                    "samples_per_sec": 0.0,
                    "peak_mem_mib": None,
                    "skipped": True,
                    "reason": f"exceeds max_fit={max_fit}",
                }
            )
            continue
        row = measure_throughput(
            model,
            criterion,
            inputs,
            targets,
            device,
            bs,
            steps=args.steps,
            warmup=args.warmup,
            optimizer_factory=optimizer_factory,
        )
        row["skipped"] = False
        rows.append(row)

    ok_rows = [r for r in rows if not r.get("skipped")]
    best = pick_best_throughput(ok_rows) if ok_rows else None
    best_bs = best["batch_size"] if best else max_fit

    total_mem_mib = torch.cuda.get_device_properties(device).total_memory / (1024**2)
    peak_frac = (max(r["peak_mem_mib"] for r in ok_rows) / total_mem_mib * 100) if ok_rows else 0.0

    print(f"config: {config_path.name}")
    print(f"arch: {config.config['arch']['type']}")
    print(f"train samples: {n_train}")
    print(f"GPU: {torch.cuda.get_device_name(device)} ({total_mem_mib:.0f} MiB total)")
    print(f"max batch that fits (safety={args.safety}): {max_fit}")
    if best:
        print(
            f"best throughput: batch_size={best_bs} "
            f"({best['samples_per_sec']:.1f} samples/s, {best['peak_mem_mib']:.0f} MiB peak)"
        )
    print(f"peak VRAM use in sweep: {peak_frac:.1f}% of device memory")
    print()
    print_table(ok_rows, max_fit, best_bs)
    skipped = [r for r in rows if r.get("skipped")]
    for row in skipped:
        print(f"  skipped batch={row['batch_size']}: {row.get('reason')}")

    out = {
        "config": str(config_path),
        "arch": config.config["arch"]["type"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_train": n_train,
        "gpu": torch.cuda.get_device_name(device),
        "total_mem_mib": total_mem_mib,
        "max_batch_fit": max_fit,
        "recommended_batch_size": best_bs,
        "recommended_throughput_samples_per_sec": best["samples_per_sec"] if best else None,
        "performance": performance,
        "results": rows,
    }

    out_path = (
        Path(args.out)
        if args.out
        else ROOT / "saved" / "benchmarks" / f"batch_size_{config_path.stem}_{datetime.now():%Y%m%d_%H%M%S}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
