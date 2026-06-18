"""Inline notebook configs — smoke training overrides on top of production JSON."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from parse_config import ConfigParser, validate_config

TEMPLATE_ROOT = Path(__file__).resolve().parents[1]

SURFACE_CONFIG_KEYS = ("isas_point", "isas_patch", "argo_point")


def _load_json(name: str) -> dict[str, Any]:
    return json.loads((TEMPLATE_ROOT / name).read_text())


def _notebook_trainer_overrides(save_subdir: str) -> dict[str, Any]:
    return {
        "epochs": 2,
        "save_dir": f"saved/notebook_runs/{save_subdir}/",
        "save_period": 2,
        "log_interval": 1,
        "verbosity": 2,
        "monitor": "off",
        "early_stop": 0,
        "tensorboard": False,
    }


def _smoke_dataloader() -> dict[str, Any]:
    return {
        "batch_size": 128,
        "shuffle": True,
        "train_frac": 0.7,
        "val_frac": 0.15,
        "test_frac": 0.15,
        "split": "train",
        "num_workers": 0,
        "pin_memory": False,
    }


def build_config_dict(key: str, *, template_root: Path | None = None) -> dict[str, Any]:
    """Return a deep-copied config dict with notebook smoke overrides."""
    root = template_root or TEMPLATE_ROOT
    if key == "isas_point":
        cfg = _load_json("config_isas.json")
        cfg["data_loader"]["args"] = {**cfg["data_loader"]["args"], **_smoke_dataloader()}
        cfg["trainer"] = _notebook_trainer_overrides("isas_point")
    elif key == "isas_patch":
        cfg = _load_json("config_isas_patch.json")
        cfg["data_loader"]["args"] = {**cfg["data_loader"]["args"], **_smoke_dataloader(), "batch_size": 128}
        cfg["trainer"] = _notebook_trainer_overrides("isas_patch")
    elif key == "argo_point":
        cfg = _load_json("config_argo.json")
        cfg["data_loader"]["args"] = {**cfg["data_loader"]["args"], **_smoke_dataloader()}
        cfg["trainer"] = _notebook_trainer_overrides("argo_point")
    else:
        raise KeyError(f"unknown config key {key!r}; expected one of {SURFACE_CONFIG_KEYS}")

    cfg = copy.deepcopy(cfg)
    if "io" in cfg:
        io = cfg["io"]
        if "data_path" in io and not Path(io["data_path"]).is_absolute():
            io["data_path"] = str((root / io["data_path"]).resolve())
        if "cache_dir" in io and not Path(io["cache_dir"]).is_absolute():
            io["cache_dir"] = str((root / io["cache_dir"]).resolve())
    return cfg


def make_config_parser(key: str, *, run_id: str = "", template_root: Path | None = None) -> ConfigParser:
    cfg = build_config_dict(key, template_root=template_root)
    validate_config(cfg)
    return ConfigParser(cfg, run_id=run_id)


AE_DEFAULTS = {
    "arch": "Autoencoder",
    "encoding_dim": 16,
    "epochs": 50,
    "batch_size": 256,
    "lr": 1e-3,
    "val_frac": 0.15,
}
