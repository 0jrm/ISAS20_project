"""Notebook configs — smoke overrides (legacy) and encoding-compare registry."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from parse_config import ConfigParser, validate_config

TEMPLATE_ROOT = Path(__file__).resolve().parents[1]

SURFACE_CONFIG_KEYS = ("isas_point", "isas_patch", "argo_point")

PRODUCTION_ARGO_KEYS = ("argo_point", "argo_patch_l4")

PRODUCTION_ARGO_SPECS: dict[str, tuple[str, str]] = {
    "argo_point": ("ARGO-point", "config/argo/config_argo.json"),
    "argo_patch_l4": ("ARGO-L4-patch", "config/argo/config_argo_patch_l4.json"),
}

MAX_EPOCHS = 2000
PCA_DIM = 16
AE_DIM = 128
FORCE_RETRAIN_DEFAULT = False

COMPARE_DIR = TEMPLATE_ROOT / "config" / "compare"
MANIFEST_PATH = TEMPLATE_ROOT / "saved" / "compare_runs" / "manifest.json"


@dataclass(frozen=True)
class CompareSpec:
    label: str
    config_file: str
    group: Literal["isas", "argo"]
    encoding: Literal["pca", "ae"]
    is_decoder: bool


COMPARE_CONFIGS: dict[str, CompareSpec] = {
    "isas_pt_pca16": CompareSpec(
        "ISAS-pt-PCA16", "isas_point_pca16.json", "isas", "pca", False
    ),
    "isas_pch_pca16": CompareSpec(
        "ISAS-pch-PCA16", "isas_patch_pca16.json", "isas", "pca", False
    ),
    "isas_pt_ae128": CompareSpec(
        "ISAS-pt-AE128", "isas_point_ae128.json", "isas", "ae", True
    ),
    "isas_pch_ae128": CompareSpec(
        "ISAS-pch-AE128", "isas_patch_ae128.json", "isas", "ae", True
    ),
    "argo_pca15": CompareSpec(
        "ARGO-PCA15", "argo_point_pca15.json", "argo", "pca", False
    ),
    "argo_pca16": CompareSpec(
        "ARGO-PCA16", "argo_point_pca16.json", "argo", "pca", False
    ),
}

COMPARE_CONFIG_KEYS = tuple(COMPARE_CONFIGS.keys())


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _resolve_io_paths(cfg: dict[str, Any], root: Path) -> None:
    if "io" not in cfg:
        return
    io = cfg["io"]
    if "data_path" in io and not Path(io["data_path"]).is_absolute():
        io["data_path"] = str((root / io["data_path"]).resolve())
    if "cache_dir" in io and not Path(io["cache_dir"]).is_absolute():
        io["cache_dir"] = str((root / io["cache_dir"]).resolve())


def build_compare_config_dict(
    key: str,
    *,
    template_root: Path | None = None,
) -> dict[str, Any]:
    """Load an encoding-compare JSON config (relative io paths — matches cache hashing)."""
    if key not in COMPARE_CONFIGS:
        raise KeyError(f"unknown compare key {key!r}; expected one of {COMPARE_CONFIG_KEYS}")
    root = template_root or TEMPLATE_ROOT
    path = COMPARE_DIR / COMPARE_CONFIGS[key].config_file
    # ponytail: do not resolve io paths here; config_hash includes io and must match
    # train.py / export_ae_latents.py (both run with cwd=NeSPReSO2_onTemplate).
    return copy.deepcopy(_load_json(path))


def make_compare_config_parser(
    key: str,
    *,
    run_id: str = "",
    template_root: Path | None = None,
) -> ConfigParser:
    cfg = build_compare_config_dict(key, template_root=template_root)
    validate_config(cfg)
    return ConfigParser(cfg, run_id=run_id)


def load_manifest(path: Path | None = None) -> dict[str, Any] | None:
    """Return compare training manifest JSON if present."""
    p = path or MANIFEST_PATH
    if not p.is_file():
        return None
    return json.loads(p.read_text())


def compare_matrix_row(key: str, cfg: ConfigParser) -> dict[str, Any]:
    spec = COMPARE_CONFIGS[key]
    outputs = cfg.config["outputs"]
    latent = sum(outputs.values())
    return {
        "key": key,
        "label": spec.label,
        "tag": cfg.config["io"]["dataset_tag"],
        "arch": cfg.config["arch"]["type"],
        "encoding": spec.encoding,
        "latent_dim": latent,
        "outputs": dict(outputs),
        "epochs": cfg.config["trainer"]["epochs"],
        "group": spec.group,
        "is_decoder": spec.is_decoder,
    }


# --- legacy smoke configs (unchanged API) ---


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
        "split_mode": "random",
        "split_seed": 42,
        "split": "train",
        "num_workers": 0,
        "pin_memory": False,
    }


def build_config_dict(key: str, *, template_root: Path | None = None) -> dict[str, Any]:
    """Return a deep-copied config dict with notebook smoke overrides."""
    root = template_root or TEMPLATE_ROOT
    if key == "isas_point":
        cfg = _load_json(root / "config/isas/config_isas.json")
        cfg["data_loader"]["args"] = {**cfg["data_loader"]["args"], **_smoke_dataloader()}
        cfg["trainer"] = _notebook_trainer_overrides("isas_point")
    elif key == "isas_patch":
        cfg = _load_json(root / "config/isas/config_isas_patch.json")
        cfg["data_loader"]["args"] = {
            **cfg["data_loader"]["args"],
            **_smoke_dataloader(),
            "batch_size": 128,
        }
        cfg["trainer"] = _notebook_trainer_overrides("isas_patch")
    elif key == "argo_point":
        cfg = _load_json(root / "config/argo/config_argo.json")
        cfg["data_loader"]["args"] = {**cfg["data_loader"]["args"], **_smoke_dataloader()}
        cfg["trainer"] = _notebook_trainer_overrides("argo_point")
    else:
        raise KeyError(f"unknown config key {key!r}; expected one of {SURFACE_CONFIG_KEYS}")

    cfg = copy.deepcopy(cfg)
    _resolve_io_paths(cfg, root)
    return cfg


def make_config_parser(key: str, *, run_id: str = "", template_root: Path | None = None) -> ConfigParser:
    cfg = build_config_dict(key, template_root=template_root)
    validate_config(cfg)
    return ConfigParser(cfg, run_id=run_id)


def build_production_config_dict(
    key: str,
    *,
    template_root: Path | None = None,
) -> dict[str, Any]:
    """Load a full production config (no notebook smoke overrides)."""
    if key not in PRODUCTION_ARGO_SPECS:
        raise KeyError(
            f"unknown production key {key!r}; expected one of {PRODUCTION_ARGO_KEYS}"
        )
    root = template_root or TEMPLATE_ROOT
    _, config_file = PRODUCTION_ARGO_SPECS[key]
    cfg = copy.deepcopy(_load_json(root / config_file))
    _resolve_io_paths(cfg, root)
    return cfg


def make_production_config_parser(
    key: str,
    *,
    run_id: str = "",
    template_root: Path | None = None,
) -> ConfigParser:
    cfg = build_production_config_dict(key, template_root=template_root)
    validate_config(cfg)
    return ConfigParser(cfg, run_id=run_id)


def production_argo_matrix_row(key: str, cfg: ConfigParser) -> dict[str, Any]:
    label, _ = PRODUCTION_ARGO_SPECS[key]
    outputs = cfg.config["outputs"]
    dl = cfg.config["data_loader"]["args"]
    return {
        "key": key,
        "label": label,
        "tag": cfg.config["io"]["dataset_tag"],
        "arch": cfg.config["arch"]["type"],
        "latent_dim": sum(outputs.values()),
        "outputs": dict(outputs),
        "split_mode": dl.get("split_mode", "random"),
        "group": "argo",
    }


AE_DEFAULTS = {
    "arch": "Autoencoder",
    "encoding_dim": AE_DIM,
    "epochs": 50,
    "batch_size": 256,
    "lr": 1e-3,
    "val_frac": 0.15,
}
