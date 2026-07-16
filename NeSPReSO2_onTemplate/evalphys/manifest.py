"""Versioned metrics manifest — edit only with version bump + changelog entry."""

from __future__ import annotations

import json
import subprocess
from datetime import date
from pathlib import Path

from evalphys.constants import ENCE_MAX, GSW_BACKEND_HEADLINE, N2_TOL, SIGMA0_TOL, VERSION
from evalphys.gsw_backend import package_versions

_PKG_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = _PKG_DIR / "METRICS_MANIFEST.json"


def _git_sha() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_PKG_DIR.parents[1],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def default_manifest() -> dict:
    vers = package_versions()
    return {
        "version": VERSION,
        "frozen_date": date.today().isoformat(),
        "N2_TOL": N2_TOL,
        "SIGMA0_TOL": SIGMA0_TOL,
        "gsw_backend_headline": GSW_BACKEND_HEADLINE,
        "gsw_versions": vers,
        "thresholds": {
            "ence_max": ENCE_MAX,
            "rc1_note": (
                "hard constraint guarantees σ₀ monotonicity on the control grid; "
                "residual N² violations are expected to be small and must be reported "
                "(see PLAN §3.2 note). Report cost in RMSE/sharpness."
            ),
        },
        "git_sha": _git_sha(),
        "changelog": [
            {
                "version": "1.0.0",
                "date": "2026-07-16",
                "note": "Initial frozen evalphys package (PLAN-v2-recovery Phase 0).",
            },
            {
                "version": "1.1.0",
                "date": date.today().isoformat(),
                "note": (
                    "Additive: σ₀-monotonicity violation metric; configurable gsw backend "
                    "(headline pinned to reference gsw); exclude_top_m semantics fixed."
                ),
            },
        ],
    }


def write_manifest(path: Path | None = None) -> Path:
    path = path or MANIFEST_PATH
    data = default_manifest()
    path.write_text(json.dumps(data, indent=2) + "\n")
    return path


def load_manifest(path: Path | None = None) -> dict:
    path = path or MANIFEST_PATH
    return json.loads(path.read_text())
