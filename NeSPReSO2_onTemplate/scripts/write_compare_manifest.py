#!/usr/bin/env python3
"""Write saved/compare_runs/manifest.json from nested model_best checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notebooks"))

from nb_checkpoints import discover_compare_checkpoint
from nb_configs import COMPARE_CONFIG_KEYS, make_compare_config_parser


def main() -> int:
    parser = argparse.ArgumentParser(description="Write compare training manifest JSON")
    parser.add_argument(
        "-o",
        "--out",
        default="saved/compare_runs/manifest.json",
        help="manifest output path (relative to NeSPReSO2_onTemplate)",
    )
    args = parser.parse_args()

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out

    runs = []
    for key in COMPARE_CONFIG_KEYS:
        cfg = make_compare_config_parser(key, template_root=ROOT)
        ckpt = discover_compare_checkpoint(key, cfg, template_root=ROOT)
        runs.append({
            "key": key,
            "checkpoint": str(ckpt) if ckpt is not None else None,
            "status": "ok" if ckpt is not None else "missing",
        })

    doc = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "phase_a": {
            "decoder_dir": "saved/decoders/isas20/Autoencoder_dim128",
            "target_key": "ae_targets_dim128",
        },
        "runs": runs,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2) + "\n")
    print(json.dumps(doc, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
