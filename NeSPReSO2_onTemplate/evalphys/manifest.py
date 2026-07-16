"""Versioned metrics manifest — edit only with version bump + changelog entry."""

from __future__ import annotations

import json
import subprocess
from datetime import date
from pathlib import Path

from evalphys.constants import ENCE_MAX, N2_TOL, VERSION

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
    return {
        "version": VERSION,
        "frozen_date": date.today().isoformat(),
        "N2_TOL": N2_TOL,
        "thresholds": {
            "ence_max": ENCE_MAX,
            "rc1_note": (
                "hard constraint ⇒ 0 by construction in Phase 3; "
                "report cost in RMSE/sharpness instead"
            ),
        },
        "git_sha": _git_sha(),
        "changelog": [
            {
                "version": VERSION,
                "date": date.today().isoformat(),
                "note": "Initial frozen evalphys package (PLAN-v2-recovery Phase 0).",
            }
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
