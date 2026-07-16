#!/usr/bin/env python3
"""Retrain residual_cube on the anchoring-fixed cache (S0 fix, 2026-07-05).

Uses the scratch-run manifest for the fresh golden (PCA source) and point_cube
(warmstart anchor) checkpoints, builds the fixed feature cache (new hash from the
point_block_norm marker), retrains, and updates the manifest so the notebook can
re-evaluate with USE_TRAINED_MODEL=1.
"""

from __future__ import annotations

import copy
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from parse_config import ConfigParser  # noqa: E402
from train import main as train_main  # noqa: E402
from preproc.features.export_feature_cache import build_feature_cache  # noqa: E402

MANIFEST = ROOT / "notebooks/scratch_outputs/scratch_manifest.json"


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    for dep in ("golden_point", "point_cube"):
        ck = (manifest.get(dep) or {}).get("checkpoint")
        assert ck and Path(ck).is_file(), f"manifest missing trained {dep} checkpoint"

    cfg = json.loads((ROOT / "config/argo/config_argo_residual_cube.json").read_text())
    cfg["data_loader"]["args"]["batch_size"] = 128  # shared-GPU VRAM budget
    cfg["io"]["pca_ckpt"] = manifest["golden_point"]["checkpoint"]
    cfg["arch"]["args"]["warmstart_ckpt"] = manifest["point_cube"]["checkpoint"]

    t0 = time.time()
    cache_path = build_feature_cache(cfg, force=False)
    print(f"fixed residual cache: {cache_path} ({time.time() - t0:.0f}s)", flush=True)
    cfg["data_loader"]["args"]["cache_path"] = str(cache_path)

    stamp = datetime.now().strftime("%m%d_%H%M%S")
    parser = ConfigParser(copy.deepcopy(cfg), run_id=f"scratch_{stamp}_residual_cube_fixed")
    print(f"training residual_cube (fixed anchor) -> {parser.save_dir}", flush=True)
    t0 = time.time()
    train_main(parser)
    ck = Path(parser.save_dir) / "model_best.pth"
    assert ck.is_file(), f"training finished but {ck} missing"

    sys.path.insert(0, str(ROOT / "notebooks"))
    from nb_checkpoints import checkpoint_epoch

    manifest["residual_cube"] = {
        "checkpoint": str(ck),
        "cache_path": str(cache_path),
        "trained_at": stamp,
        "wall_s": round(time.time() - t0),
        "best_epoch": checkpoint_epoch(ck),
        "note": "retrained after S0 anchoring fix (point block z-scored)",
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {ck}", flush=True)


if __name__ == "__main__":
    main()
