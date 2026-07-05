"""Feature operator registry (Component B)."""

from __future__ import annotations

from preproc.features.operators import (
    OPERATOR_REGISTRY,
    apply_operator,
    list_operators,
    register,
)

__all__ = ["OPERATOR_REGISTRY", "apply_operator", "list_operators", "register"]
