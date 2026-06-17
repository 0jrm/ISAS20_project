"""Batch-size probing and throughput helpers."""

from __future__ import annotations

import gc
import time
from typing import Any, Callable, Iterable

import torch


def train_samples_from_cache(n_total: int, train_frac: float) -> int:
    return max(1, int(n_total * train_frac))


def cap_batch_size(batch_size: int, n_train: int) -> int:
    """Cap a fixed batch size by the train-set size."""
    return max(1, min(batch_size, n_train))


def _is_oom(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda error" in msg and "memory" in msg


def _cuda_reset(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    gc.collect()


def _run_train_step(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data: torch.Tensor,
    target: torch.Tensor,
    indices: torch.Tensor,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    data = data.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)
    indices = indices.to(device, non_blocking=True)
    output = model(data)
    loss = criterion(output, target, indices)
    loss.backward()
    optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return float(loss.item())


def batch_fits(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    batch_size: int,
    *,
    optimizer_factory: Callable[[], torch.optim.Optimizer] | None = None,
) -> bool:
    """Return True if a full forward/backward/optim step succeeds at ``batch_size``."""
    if batch_size < 1:
        return False
    n = min(batch_size, inputs.shape[0])
    data = inputs[:n]
    target = targets[:n]
    indices = torch.arange(n, dtype=torch.long)

    if optimizer_factory is None:
        optimizer_factory = lambda: torch.optim.Adam(model.parameters(), lr=1e-3)

    _cuda_reset(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    optimizer = optimizer_factory()
    try:
        _run_train_step(model, criterion, data, target, indices, device, optimizer)
        return True
    except RuntimeError as exc:
        if _is_oom(exc):
            return False
        raise
    finally:
        del optimizer
        _cuda_reset(device)


def probe_max_batch_size(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    *,
    n_train: int,
    min_batch_size: int = 1,
    safety_fraction: float = 0.95,
    optimizer_factory: Callable[[], torch.optim.Optimizer] | None = None,
) -> int:
    """Find the largest batch size that fits in GPU memory (up to ``n_train``)."""
    upper = max(1, n_train)
    lower = max(1, min(min_batch_size, upper))

    if not batch_fits(
        model,
        criterion,
        inputs,
        targets,
        device,
        lower,
        optimizer_factory=optimizer_factory,
    ):
        raise RuntimeError(f"minimum batch_size={lower} does not fit on {device}")

    if lower == upper:
        return lower

    left, right = lower, upper
    best = lower
    while left <= right:
        mid = (left + right) // 2
        if batch_fits(
            model,
            criterion,
            inputs,
            targets,
            device,
            mid,
            optimizer_factory=optimizer_factory,
        ):
            best = mid
            left = mid + 1
        else:
            right = mid - 1

    if safety_fraction < 1.0 and best > lower:
        safe = max(lower, int(best * safety_fraction))
        while safe < best and not batch_fits(
            model,
            criterion,
            inputs,
            targets,
            device,
            safe,
            optimizer_factory=optimizer_factory,
        ):
            safe = max(lower, safe - max(1, safe // 16))
        best = safe

    return max(1, best)


def resolve_batch_size(
    configured: int,
    n_train: int,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    *,
    safety_fraction: float = 0.95,
    optimizer_factory: Callable[[], torch.optim.Optimizer] | None = None,
) -> int:
    """
    Resolve effective batch size for training.

    - ``configured > 0``: use that value, capped by ``n_train``.
    - ``configured == 0``: probe the largest batch that fits in VRAM (up to ``n_train``).
    """
    if configured > 0:
        return cap_batch_size(configured, n_train)

    if device.type != "cuda":
        return cap_batch_size(512, n_train)

    return probe_max_batch_size(
        model,
        criterion,
        inputs,
        targets,
        device,
        n_train=n_train,
        safety_fraction=safety_fraction,
        optimizer_factory=optimizer_factory,
    )


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_throughput(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    batch_size: int,
    *,
    steps: int = 30,
    warmup: int = 5,
    optimizer_factory: Callable[[], torch.optim.Optimizer] | None = None,
) -> dict[str, Any]:
    """Time repeated train steps; return throughput and peak GPU memory."""
    n = min(batch_size, inputs.shape[0])
    data = inputs[:n]
    target = targets[:n]
    indices = torch.arange(n, dtype=torch.long)

    if optimizer_factory is None:
        optimizer_factory = lambda: torch.optim.Adam(model.parameters(), lr=1e-3)

    optimizer = optimizer_factory()
    model.train()

    _cuda_reset(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    times: list[float] = []
    for step in range(warmup + steps):
        _sync(device)
        t0 = time.perf_counter()
        _run_train_step(model, criterion, data, target, indices, device, optimizer)
        _sync(device)
        elapsed = time.perf_counter() - t0
        if step >= warmup:
            times.append(elapsed)

    sec_per_step = sum(times) / len(times)
    samples_per_sec = n / sec_per_step if sec_per_step > 0 else 0.0
    peak_mem_mib = 0.0
    if device.type == "cuda":
        peak_mem_mib = torch.cuda.max_memory_allocated(device) / (1024**2)

    return {
        "batch_size": n,
        "sec_per_step": sec_per_step,
        "samples_per_sec": samples_per_sec,
        "peak_mem_mib": peak_mem_mib,
    }


def default_sweep_sizes(n_train: int, max_bs: int) -> list[int]:
    """Powers-of-two sweep up to ``max_bs``, always including ``n_train`` if smaller."""
    sizes = []
    bs = 64
    cap = min(n_train, max_bs) if max_bs > 0 else n_train
    while bs <= cap:
        sizes.append(bs)
        bs *= 2
    if cap not in sizes:
        sizes.append(cap)
    if n_train < cap and n_train not in sizes:
        sizes.append(n_train)
    return sorted(set(s for s in sizes if s <= cap))


def pick_best_throughput(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Pick batch size with highest samples/sec."""
    rows = list(results)
    if not rows:
        raise ValueError("no benchmark results")
    return max(rows, key=lambda r: r["samples_per_sec"])
