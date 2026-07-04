#!/usr/bin/env python3
"""Scaffold for self-supervised patch encoder pretraining (Phase 6)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.model import ResidualPatchEncoder


class PatchReconstructionHead(nn.Module):
  def __init__(self, d_model: int, patch_flat: int):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(d_model, d_model),
      nn.ReLU(inplace=True),
      nn.Linear(d_model, patch_flat),
    )

  def forward(self, x):
    return self.net(x)


def main():
  parser = argparse.ArgumentParser(description="Self-supervised patch encoder pretraining scaffold")
  parser.add_argument("-c", "--config", required=True, help="Residual config JSON (for patch_shape)")
  parser.add_argument("--epochs", type=int, default=5)
  parser.add_argument("--lr", type=float, default=1e-3)
  parser.add_argument("--out", default="saved/pretrain/patch_enc.pth")
  parser.add_argument("--max-batches", type=int, default=50)
  args = parser.parse_args()

  from base.util import read_json
  from parse_config import validate_config
  from preproc.export_argo_residual_cache import build_argo_residual_cache
  from preproc.l3_input import sync_arch_with_io

  cfg = read_json(args.config)
  validate_config(cfg)
  cache_path = build_argo_residual_cache(cfg, force=False)
  sync_arch_with_io(cfg)

  import pickle

  with open(cache_path, "rb") as f:
    cache = pickle.load(f)

  patch_offset = int(cache.get("patch_offset", cfg["arch"]["args"]["patch_offset"]))
  patch_shape = tuple(cache.get("sat_patch_shape", cfg["arch"]["args"]["patch_shape"]))
  c, t, h, w = patch_shape
  patch_flat = c * t * h * w

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  enc = ResidualPatchEncoder(patch_shape=patch_shape).to(device)

  with torch.no_grad():
    feat_dim = enc(torch.zeros(1, patch_flat, device=device)).shape[1]
  head = PatchReconstructionHead(feat_dim, patch_flat).to(device)

  opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()), lr=args.lr)
  inputs = torch.tensor(cache["inputs"][:, patch_offset:], dtype=torch.float32)

  enc.train()
  head.train()
  losses = []
  for epoch in range(args.epochs):
    perm = torch.randperm(inputs.shape[0])
    for bi, start in enumerate(range(0, inputs.shape[0], 256)):
      if bi >= args.max_batches:
        break
      idx = perm[start : start + 256]
      batch = inputs[idx].to(device)
      mask = (torch.rand_like(batch) > 0.15).float()
      masked = batch * mask
      feat = enc(masked)
      pred = head(feat)
      loss = nn.functional.mse_loss(pred, batch)
      opt.zero_grad()
      loss.backward()
      opt.step()
      losses.append(float(loss.item()))
    print(f"epoch {epoch + 1}: loss={np.mean(losses[-args.max_batches:]):.6f}")

  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  torch.save({"patch_enc": enc.state_dict(), "losses": losses, "patch_shape": list(patch_shape)}, out)
  meta = {"checkpoint": str(out), "final_loss": losses[-1] if losses else None}
  out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
  print(json.dumps(meta, indent=2))
  return 0


if __name__ == "__main__":
  import numpy as np

  raise SystemExit(main())
