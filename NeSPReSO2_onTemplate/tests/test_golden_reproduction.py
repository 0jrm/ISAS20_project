"""Golden-reproduction regression guard (anomaly 3).

The program "golden" number T=0.416 / S=0.072 was measured on a **random**
70/15/15 split (the compare-notebook / compare_runs recipe, ``split_mode:
random``). The cube-native program evaluates on the **chronological** split,
on which the *same* golden checkpoint and the *same* evaluator score ~0.514.

These tests pin both numbers so that:
  1. The historical 0.416 stays reproducible (random split) — it is a real
     number, just for a different split.
  2. Nobody again compares a random-split golden (0.416) against a
     chronological-split cube model (point_cube 0.577). The apples-to-apples
     golden reference on the chronological test split is ~0.514.

Run: ``pytest tests/test_golden_reproduction.py -m golden_repro -q``
"""

from __future__ import annotations

import copy
import uuid
from pathlib import Path

import pytest

from base.util import read_json
from parse_config import ConfigParser

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "config/argo/config_argo.json"
GOLDEN_CKPT = ROOT / "saved/models/NeSPReSO2_ARGO_GoM/argo16_scales/model_best.pth"
# Golden's original training cache; inputs identical to the current argo_v2
# cache for the test split (verified: same eval loss to full precision).
GOLDEN_CACHE = ROOT / "../data/cache/train_ready_651f62a4b596.pkl"

# Historical program target (random split, argo_v2), from
# notebooks/compare_outputs/results.json.
GOLDEN_RANDOM_T = 0.4158
GOLDEN_RANDOM_S = 0.0722
# Apples-to-apples golden on the chronological test split (same model + eval).
GOLDEN_CHRONO_T = 0.5143
GOLDEN_CHRONO_S = 0.0834
ATOL = 0.01


def _eval_golden(split_mode: str) -> dict:
    if not GOLDEN_CKPT.is_file():
        pytest.skip("golden checkpoint missing")
    if not GOLDEN_CACHE.is_file():
        pytest.skip("golden argo_v2 cache missing")

    from eval_run import main as eval_main

    cfg = copy.deepcopy(read_json(CFG))
    cfg["n_gpu"] = 0  # CPU: deterministic and avoids GPU contention
    # Unique run name so back-to-back ConfigParser instances don't collide on
    # the timestamped save_dir (mkdir exist_ok=False).
    cfg["name"] = f"golden_repro_{split_mode}_{uuid.uuid4().hex[:8]}"
    dl = cfg["data_loader"]["args"]
    dl["cache_path"] = str(GOLDEN_CACHE)
    dl["split_mode"] = split_mode
    dl["split_seed"] = 42
    config = ConfigParser(cfg)
    return eval_main(config, str(GOLDEN_CKPT), split="test")


@pytest.mark.golden_repro
def test_golden_random_split_reproduces_0416():
    """The historical 0.416/0.072 is reproducible — on the RANDOM split only."""
    rep = _eval_golden("random")
    t = rep["raw_profile_rmse"]["temperature"]
    s = rep["raw_profile_rmse"]["salinity"]
    assert abs(t - GOLDEN_RANDOM_T) < ATOL, f"random-split golden T={t:.4f} != {GOLDEN_RANDOM_T}"
    assert abs(s - GOLDEN_RANDOM_S) < ATOL, f"random-split golden S={s:.4f} != {GOLDEN_RANDOM_S}"


@pytest.mark.golden_repro
def test_golden_chronological_is_apples_to_apples_reference():
    """On the chronological split (cube-program split) golden is ~0.514, not 0.416.

    This is the number that must be compared against point_cube / residual.
    """
    rep = _eval_golden("chronological")
    t = rep["raw_profile_rmse"]["temperature"]
    s = rep["raw_profile_rmse"]["salinity"]
    assert abs(t - GOLDEN_CHRONO_T) < ATOL, f"chronological golden T={t:.4f} != {GOLDEN_CHRONO_T}"
    assert abs(s - GOLDEN_CHRONO_S) < ATOL, f"chronological golden S={s:.4f} != {GOLDEN_CHRONO_S}"


@pytest.mark.golden_repro
def test_split_explains_the_0416_vs_0514_gap():
    """Same model + same evaluator: the only difference is the split definition."""
    rnd = _eval_golden("random")["raw_profile_rmse"]["temperature"]
    chrono = _eval_golden("chronological")["raw_profile_rmse"]["temperature"]
    # Chronological (latest-in-time profiles) is strictly harder here.
    assert chrono > rnd + 0.05, (
        f"expected chronological ({chrono:.4f}) meaningfully worse than random ({rnd:.4f}); "
        "if not, the split is no longer the explanation for the 0.416/0.514 gap"
    )
