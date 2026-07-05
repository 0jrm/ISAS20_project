"""Pytest configuration for NeSPReSO2_onTemplate."""

from __future__ import annotations

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "s0b_gate: S0b release gate — residual init must match point_cube baseline",
    )
    config.addinivalue_line(
        "markers",
        "golden_repro: golden reproduces 0.416 (random split) / 0.514 (chronological split)",
    )
