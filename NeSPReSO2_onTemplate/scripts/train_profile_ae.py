#!/usr/bin/env python3
"""Stage A: train per-variable profile autoencoder from a train-ready cache."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import model.model as module_arch
from parse_config import ConfigParser, validate_config
from train import ensure_cache


class MaskedMSELoss(nn.Module):
    def forward(self, pred, target, mask):
        valid = (~mask).float()
        err = (pred - target) ** 2 * valid
        n = valid.sum()
        if n == 0:
            return pred.new_tensor(0.0)
        return err.sum() / n


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


def pca_baseline_rmse(profiles: np.ndarray, mask: np.ndarray, n_comp: int) -> float:
    from sklearn.decomposition import PCA

    fit_x = np.nan_to_num(profiles, nan=0.0)
    pca = PCA(n_components=n_comp).fit(fit_x)
    recon = pca.inverse_transform(pca.transform(fit_x))
    valid = ~mask
    if not np.any(valid):
        return float("nan")
    err = (recon - profiles) ** 2
    return float(np.sqrt(np.mean(err[valid])))


def build_ae(arch: str, encoding_dim: int, input_dim: int, *, layer_scale: int = 1) -> tuple[nn.Module, list[int], list[int]]:
    if arch in ("Autoencoder", "ResAutoencoder"):
        enc = [min(512, max(128, input_dim // 4)), 128, 64]
        dec = [64, 128, min(512, max(128, input_dim // 4))]
        if layer_scale != 1:
            enc = [h * layer_scale for h in enc]
            dec = [h * layer_scale for h in dec]
        use_residual = arch == "ResAutoencoder"
        cls = module_arch.ResAutoencoder if use_residual else module_arch.Autoencoder
        return (
            cls(
                encoding_dim,
                encoder_layers=enc,
                decoder_layers=dec,
                input_dim=input_dim,
                residual=use_residual,
            ),
            enc,
            dec,
        )
    if arch == "KAN_Autoencoder":
        return module_arch.KAN_Autoencoder(encoding_dim, input_dim=input_dim), [], []
    raise ValueError(f"unknown arch {arch!r}")


def train_variable(
    *,
    profiles: np.ndarray,
    mask: np.ndarray,
    arch: str,
    encoding_dim: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    val_frac: float,
    seed: int,
    layer_scale: int = 1,
) -> tuple[nn.Module, dict]:
    n, depth = profiles.shape
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = max(1, int(n * val_frac))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    def tensors(idxs):
        x = torch.tensor(profiles[idxs], dtype=torch.float32)
        m = torch.tensor(mask[idxs], dtype=torch.bool)
        return x, m

    x_train, m_train = tensors(train_idx)
    x_val, m_val = tensors(val_idx)

    train_loader = DataLoader(TensorDataset(x_train, m_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, m_val), batch_size=batch_size, shuffle=False)

    model, enc_layers, dec_layers = build_ae(arch, encoding_dim, depth, layer_scale=layer_scale)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = MaskedMSELoss()

    best_val = float("inf")
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, mb in train_loader:
            xb, mb = xb.to(device), mb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb, mb)
            loss = criterion(pred, xb, mb)
            loss.backward()
            opt.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, mb in val_loader:
                xb, mb = xb.to(device), mb.to(device)
                pred = model(xb, mb)
                val_losses.append(criterion(pred, xb, mb).item())
        val_loss = float(np.mean(val_losses))
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        xv, mv = x_val.to(device), m_val.to(device)
        recon = model(xv, mv)
        valid = ~mv
        rmse = torch.sqrt(((recon - xv) ** 2 * valid.float()).sum() / valid.float().sum()).item()

    stats = {
        "val_rmse": rmse,
        "best_val_loss": best_val,
        "n_train": len(train_idx),
        "n_val": n_val,
        "depth": depth,
        "encoder_layers": enc_layers,
        "decoder_layers": dec_layers,
        "layer_scale": layer_scale,
    }
    return model, stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Train profile autoencoder (Phase 5 Stage A)")
    parser.add_argument("-c", "--config", required=True, type=str)
    parser.add_argument("--cache", default=None, type=str, help="override cache pickle path")
    parser.add_argument("--arch", default="Autoencoder", choices=["Autoencoder", "ResAutoencoder", "KAN_Autoencoder"])
    parser.add_argument("--encoding-dim", type=int, default=16)
    parser.add_argument("--layer-scale", type=int, default=1, help="multiply AE hidden layer widths")
    parser.add_argument("--arch-tag", default=None, help="subdir under out-dir/TAG/ (default: --arch)")
    parser.add_argument("--variable", default="all", help="temperature | salinity | all")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--out-dir", default="saved/decoders", type=str)
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

    cache_path = args.cache or ensure_cache(config)
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)

    outputs = dict(cache["outputs"])
    vars_to_run = list(outputs.keys()) if args.variable == "all" else [args.variable]
    tag = cache.get("dataset_tag", config_dict["io"].get("dataset_tag", "unknown"))
    seed = int(config_dict.get("seed", 42))

    arch_tag = args.arch_tag or args.arch
    summary = {
        "config": str(config_path),
        "cache": cache_path,
        "dataset_tag": tag,
        "arch": args.arch,
        "arch_tag": arch_tag,
        "encoding_dim": args.encoding_dim,
        "layer_scale": args.layer_scale,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "variables": {},
    }

    for name in vars_to_run:
        if name not in outputs:
            raise KeyError(f"unknown variable {name!r}; outputs={list(outputs.keys())}")
        n_comp = outputs[name]
        prof = profiles_sample_major(cache, name)
        mask = np.isnan(prof)
        prof = np.nan_to_num(prof, nan=0.0)

        pca_rmse = pca_baseline_rmse(prof, mask, n_comp)
        print(f"\n=== {name} depth={prof.shape[1]} PCA-{n_comp} recon RMSE: {pca_rmse:.6f} ===")

        model, stats = train_variable(
            profiles=prof,
            mask=mask,
            arch=args.arch,
            encoding_dim=args.encoding_dim,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            val_frac=args.val_frac,
            seed=seed,
            layer_scale=args.layer_scale,
        )
        stats["pca_recon_rmse"] = pca_rmse
        print(f"AE val RMSE: {stats['val_rmse']:.6f}  (PCA baseline: {pca_rmse:.6f})")

        out_dir = Path(args.out_dir) / tag / arch_tag / name
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "arch": args.arch,
            "encoding_dim": args.encoding_dim,
            "input_dim": prof.shape[1],
            "variable": name,
            "dataset_tag": tag,
            "encoder_layers": stats.get("encoder_layers"),
            "decoder_layers": stats.get("decoder_layers"),
            "layer_scale": args.layer_scale,
            "residual": args.arch == "ResAutoencoder",
            "state_dict": model.state_dict(),
            "stats": stats,
        }
        torch.save(ckpt, out_dir / "decoder_best.pth")
        (out_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
        summary["variables"][name] = {"out_dir": str(out_dir), **stats}

    summary_path = Path(args.out_dir) / tag / arch_tag / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nWrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
