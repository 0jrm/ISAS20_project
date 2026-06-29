"""Locate surface-model checkpoints; train if missing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from parse_config import ConfigParser

TEMPLATE_ROOT = Path(__file__).resolve().parents[1]

# GoM production runs (see PLAN-patch-arch-handoff.md)
KNOWN_CHECKPOINTS: dict[str, str] = {
    "isas_point": "saved/models/NeSPReSO2_ISAS_GoM/baseline15pc/model_best.pth",
    "isas_patch": "saved/models/NeSPReSO2_ISAS_GoM_patch/patch16_scales/model_best.pth",
    "argo_point": "saved/models/NeSPReSO2_ARGO_GoM/argo16_scales/model_best.pth",
}

CONFIG_EXPER_NAME = {
    "isas_point": "NeSPReSO2_ISAS_GoM",
    "isas_patch": "NeSPReSO2_ISAS_GoM_patch",
    "argo_point": "NeSPReSO2_ARGO_GoM",
}


def _exists(path: Path | str | None) -> Path | None:
    if path is None:
        return None
    p = Path(path)
    return p if p.is_file() else None


def discover_checkpoint(
    key: str,
    cfg: ConfigParser | None = None,
    *,
    template_root: Path | None = None,
    explicit: Path | str | None = None,
) -> Path | None:
    """Find ``model_best.pth`` for a surface config key (newest match wins on glob)."""
    root = template_root or TEMPLATE_ROOT

    for candidate in (
        explicit,
        KNOWN_CHECKPOINTS.get(key),
        _checkpoint_from_config(cfg) if cfg is not None else None,
    ):
        if candidate is None:
            continue
        p = candidate if isinstance(candidate, Path) else root / candidate
        found = _exists(p)
        if found is not None:
            return found

    exper = CONFIG_EXPER_NAME.get(key) or (cfg.config["name"] if cfg else None)
    if exper is None:
        return None

    pattern = root / "saved" / "models" / exper
    if not pattern.is_dir():
        return None

    hits = sorted(pattern.glob("*/model_best.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
    return hits[0] if hits else None


def _checkpoint_from_config(cfg: ConfigParser) -> Path | None:
    save_dir = getattr(cfg, "save_dir", None)
    if save_dir is None:
        return None
    for name in ("model_best.pth", "checkpoint.pth"):
        p = Path(save_dir) / name
        if p.is_file():
            return p
    epoch_ckpts = sorted(Path(save_dir).glob("checkpoint-epoch*.pth"))
    return epoch_ckpts[-1] if epoch_ckpts else None


def checkpoint_epoch(path: Path | str) -> int | None:
    """Return completed epoch from a ``.pth`` checkpoint, or None if unreadable."""
    import torch

    try:
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        epoch = ckpt.get("epoch")
        return int(epoch) if epoch is not None else None
    except Exception:
        return None


def apply_training_epochs(cfg: ConfigParser, epochs: int, *, monitor: bool = True) -> None:
    """Set trainer epochs (and monitoring) before ``train.main``."""
    t = cfg.config["trainer"]
    t["epochs"] = int(epochs)
    if monitor:
        t["monitor"] = "min val_loss"
        t["early_stop"] = t.get("early_stop") or 500
    else:
        t["monitor"] = "off"
        t["early_stop"] = 0


def _checkpoint_after_train(
    key: str,
    cfg: ConfigParser,
    *,
    template_root: Path | None = None,
) -> Path:
    ckpt = discover_checkpoint(key, cfg, template_root=template_root)
    if ckpt is None:
        ckpt = _checkpoint_from_config(cfg)
    if ckpt is None or not ckpt.is_file():
        raise FileNotFoundError(f"{key}: training finished but no checkpoint under {cfg.save_dir}")
    return ckpt


def resolve_or_train(
    key: str,
    cfg: ConfigParser,
    *,
    train_fn,
    max_epochs: int = 100,
    template_root: Path | None = None,
    explicit: Path | str | None = None,
    force_train: bool = False,
) -> tuple[Path, str]:
    """
    Return (checkpoint_path, source) where source is ``found``, ``resumed``, or ``trained``.

    Uses an existing checkpoint when it already reached ``max_epochs``. Otherwise trains
  (resuming from a partial checkpoint when possible) up to ``max_epochs``.
    """
    target_epochs = int(max_epochs)
    found = None if force_train else discover_checkpoint(
        key, cfg, template_root=template_root, explicit=explicit
    )

    if found is not None and not force_train:
        done = checkpoint_epoch(found)
        if done is not None and done >= target_epochs:
            return found, "found"
        if done is None:
            return found, "found"
        cfg.resume = str(found)
        apply_training_epochs(cfg, target_epochs, monitor=True)
        print(f"{key}: resuming from epoch {done}, training to {target_epochs} …")
        train_fn(cfg)
        return _checkpoint_after_train(key, cfg, template_root=template_root), "resumed"

    cfg.resume = None
    apply_training_epochs(cfg, target_epochs, monitor=True)
    print(f"{key}: training up to {target_epochs} epochs …")
    train_fn(cfg)
    return _checkpoint_after_train(key, cfg, template_root=template_root), "trained"
