#!/usr/bin/env python3
"""Stage A2: encode cache profiles with frozen AEs; write ae_targets for decoder training."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.loss import ae_weights_numpy, load_profile_decoder
from parse_config import ConfigParser, validate_config
from train import ensure_cache


def profiles_sample_major(cache: dict, name: str) -> np.ndarray:
    prof = np.asarray(cache["profiles"][name], dtype=np.float32)
    n = cache["inputs"].shape[0]
    if prof.ndim != 2:
        raise ValueError(f"profiles[{name!r}] must be 2-D, got {prof.shape}")
    if prof.shape[0] == n:
        return prof
    if prof.shape[1] == n:
        return prof.T
    raise ValueError(f"profiles[{name!r}] shape {prof.shape} inconsistent with n={n}")


@torch.no_grad()
def encode_profiles(model: torch.nn.Module, profiles: np.ndarray, mask: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    latents = []
    n = profiles.shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        x = torch.tensor(profiles[start:end], dtype=torch.float32, device=device)
        m = torch.tensor(mask[start:end], dtype=torch.bool, device=device)
        latents.append(model.encode(x, m).cpu().numpy())
    return np.concatenate(latents, axis=0).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export AE latent targets into train cache (Phase 5 A2)")
    parser.add_argument("-c", "--config", required=True, type=str)
    parser.add_argument("--cache", default=None, type=str, help="override cache pickle path")
    parser.add_argument("--decoder-dir", default=None, type=str, help="saved/decoders/TAG/ARCH")
    parser.add_argument("--target-key", default="ae_targets", type=str)
    parser.add_argument("--weight-key", default="ae_weights", type=str)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--dry-run", action="store_true", help="print stats only; do not rewrite cache")
    parser.add_argument("-d", "--device", default=None, type=str)
    args = parser.parse_args()

    if args.device is not None:
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config_dict = json.loads(config_path.read_text())
    validate_config(config_dict)
    config = ConfigParser(config_dict, run_id="")

    cache_path = Path(args.cache or ensure_cache(config))
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)

    outputs = dict(cache["outputs"])
    tag = cache.get("dataset_tag", config_dict["io"].get("dataset_tag", "unknown"))
    decoder_dir = Path(args.decoder_dir or f"saved/decoders/{tag}/Autoencoder")
    if not decoder_dir.is_absolute():
        decoder_dir = ROOT / decoder_dir

    blocks = []
    out_dims = {}
    for name in outputs:
        ckpt = decoder_dir / name / "decoder_best.pth"
        if not ckpt.is_file():
            raise FileNotFoundError(f"missing decoder for {name!r}: {ckpt}")
        prof = profiles_sample_major(cache, name)
        mask = np.isnan(prof)
        prof = np.nan_to_num(prof, nan=0.0)
        model = load_profile_decoder(ckpt, device)
        k = int(model.encoding_dim)
        out_dims[name] = k
        latent = encode_profiles(model, prof, mask, device, args.batch_size)
        if latent.shape[1] != k:
            raise ValueError(f"{name}: encoder dim {latent.shape[1]} != {k}")
        blocks.append(latent)
        print(f"{name}: encoded {latent.shape[0]} profiles -> {latent.shape[1]}-d")

    ae_targets = np.hstack(blocks).astype(np.float32)
    ae_weights = ae_weights_numpy(ae_targets, out_dims)

    if args.dry_run:
        print(f"ae_targets shape {ae_targets.shape}; dry-run, cache unchanged")
        return 0

    if "pca_targets" not in cache:
        cache["pca_targets"] = cache["targets"]
        cache["pca_weights"] = cache["weights"]
    cache[args.target_key] = ae_targets
    cache[args.weight_key] = ae_weights

    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(cache_path)
    print(f"Wrote {args.target_key} / {args.weight_key} to {cache_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
