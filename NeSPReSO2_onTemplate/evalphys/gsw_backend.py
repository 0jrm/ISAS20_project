"""Explicit GSW backend switch — headline metrics always use reference ``gsw``."""

from __future__ import annotations

import importlib
from typing import Any

_ACCEPTED = frozenset({"gsw", "gsw_torch"})
_DEFAULT = "gsw"
_config_backend: str | None = None
_headline_frozen: bool = True  # frozen-headline mode: force reference gsw


def set_config_backend(backend: str | None) -> None:
    """Set backend from ``io.gsw_backend`` (or None to clear)."""
    global _config_backend
    if backend is None:
        _config_backend = None
        return
    b = str(backend).strip()
    if b not in _ACCEPTED:
        raise ValueError(f"io.gsw_backend must be one of {sorted(_ACCEPTED)}, got {b!r}")
    _config_backend = b


def set_headline_frozen(flag: bool) -> None:
    """When True (default), metric paths assert backend resolves to reference ``gsw``."""
    global _headline_frozen
    _headline_frozen = bool(flag)


def resolve_backend(backend: str | None = None) -> str:
    if backend is not None:
        b = str(backend).strip()
    elif _config_backend is not None:
        b = _config_backend
    else:
        b = _DEFAULT
    if b not in _ACCEPTED:
        raise ValueError(f"gsw backend must be one of {sorted(_ACCEPTED)}, got {b!r}")
    return b


def get_gsw(backend: str | None = None, *, allow_torch_for_training: bool = False) -> Any:
    """Return the GSW module.

    Resolution: explicit ``backend`` > config > default ``\"gsw\"``.
    Headline/frozen metric mode rejects ``gsw_torch`` unless ``allow_torch_for_training``.
    """
    b = resolve_backend(backend)
    if _headline_frozen and not allow_torch_for_training and b != "gsw":
        raise RuntimeError(
            f"frozen headline metrics require reference gsw backend, got {b!r} "
            "(set io.gsw_backend='gsw' or pass backend='gsw'; use allow_torch_for_training "
            "only for training losses / equivalence tests)"
        )
    return importlib.import_module(b)


def package_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in ("gsw", "gsw_torch"):
        try:
            mod = importlib.import_module(name)
            out[name] = getattr(mod, "__version__", None) or getattr(mod, "VERSION", "?")
        except ImportError:
            out[name] = None
    return out
