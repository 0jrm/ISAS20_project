#!/usr/bin/env python3
"""Frozen 9-d stoch_eof μ PCs for every cache row. Pair nets read these as the source."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> int:
    import model.model as module_arch
    from parse_config import ConfigParser, validate_config
    from train import ensure_cache, set_seed
    from base.util import prepare_device, read_json

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-c", "--config", default="config/argo/config_argo_stoch_eof.json")
    ap.add_argument(
        "-r",
        "--checkpoint",
        default=(
            "saved/stoch_eof/models/NeSPReSO2_ARGO_GoM_stoch_eof_"
            "stoch_eof_s42_s2/stoch_eof_s42_s2/model_best.pth"
        ),
    )
    ap.add_argument("-o", "--out", default="../data/cache/synth_pcs_stoch_eof_s42.npy")
    args = ap.parse_args()

    cfg = read_json(args.config)
    validate_config(cfg)
    config = ConfigParser(cfg, run_id="")
    set_seed(config.config.get("seed", 42))
    ensure_cache(config)
    device, _ = prepare_device(config["n_gpu"])
    model = config.init_obj("arch", module_arch).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)))
    model.eval()

    import data_loader.data_loaders as module_data

    dl = dict(config["data_loader"]["args"])
    dl["split"] = "train"
    dl["shuffle"] = False
    loader = getattr(module_data, config["data_loader"]["type"])(**dl)
    x = np.asarray(loader.cache["inputs"], dtype=np.float32)
    d = int(config["arch"]["args"]["output_dim"])
    chunks = []
    xt = torch.as_tensor(x, device=device)
    with torch.no_grad():
        for i in range(0, xt.shape[0], 512):
            chunks.append(model(xt[i : i + 512])[:, :d].cpu().numpy())
    pcs = np.concatenate(chunks, axis=0).astype(np.float32)
    out = Path(args.out)
    if not out.is_absolute():
        out = _ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, pcs)
    print(f"wrote {out} shape={pcs.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
