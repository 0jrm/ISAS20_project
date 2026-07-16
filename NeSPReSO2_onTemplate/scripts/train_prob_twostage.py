#!/usr/bin/env python3
"""Phase 4.3 — two-stage probabilistic training launcher.

Stage 1: freeze σ head, MSE on μ (loss_config.freeze_sigma / prob_mode=mse).
Stage 2: unfreeze σ, μ LR × 0.1, switch to prob_mode (crps|nll|quantile).
"""

from __future__ import annotations

import argparse
import collections
import json
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _write_json(path: Path, cfg: dict) -> None:
    path.write_text(json.dumps(cfg, indent=4) + "\n")


def _run_train(cfg_path: Path, run_id: str, resume: str | None = None) -> Path:
    from parse_config import ConfigParser
    import train as train_mod

    argv = ["-c", str(cfg_path), "-id", run_id]
    if resume:
        argv += ["-r", resume]
    # ConfigParser.from_args expects argparse namespace via its own parser
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default=None, type=str)
    ap.add_argument("-r", "--resume", default=None, type=str)
    ap.add_argument("-d", "--device", default=None, type=str)
    ap.add_argument("-id", "--run-id", default=None, type=str)
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["--lr", "--learning_rate"], type=float, target="optimizer;args;lr"),
        CustomArgs(["--bs", "--batch_size"], type=int, target="data_loader;args;batch_size"),
    ]
    ns = ap.parse_args(argv)
    # rebuild argv for from_args
    sys_argv_backup = sys.argv
    sys.argv = ["train_prob_twostage.py", "-c", str(cfg_path), "-id", run_id] + (
        ["-r", resume] if resume else []
    )
    try:
        config = ConfigParser.from_args(ap, options)
        train_mod.main(config)
        ckpt_dir = Path(config._save_dir)
    finally:
        sys.argv = sys_argv_backup
    return ckpt_dir


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-c", "--config", required=True, help="base density_spice prob config JSON")
    ap.add_argument("--prob-mode", default="crps", choices=("crps", "nll", "quantile"))
    ap.add_argument("--stage1-epochs", type=int, default=None)
    ap.add_argument("--stage2-epochs", type=int, default=None)
    ap.add_argument("--parent-tag", default=None, help="shared run parent id")
    ap.add_argument("--workdir", default=None, help="temp dir for stage configs")
    args = ap.parse_args()

    base = _load_json(Path(args.config))
    parent = args.parent_tag or f"prob_{args.prob_mode}"
    work = Path(args.workdir or (_ROOT / "saved" / f"_twostage_{parent}"))
    work.mkdir(parents=True, exist_ok=True)

    # --- Stage 1 ---
    s1 = json.loads(json.dumps(base))
    s1["name"] = base.get("name", "nespreso") + f"_{parent}_s1"
    arch = s1.setdefault("arch", {}).setdefault("args", {})
    arch["probabilistic"] = True
    if args.prob_mode == "quantile":
        arch["n_quantiles"] = 9
    else:
        arch["n_quantiles"] = 0
    lc = s1.setdefault("loss_config", {})
    lc["mode"] = "density_spice"
    lc["prob_mode"] = "mse"
    lc["freeze_sigma"] = True
    if args.stage1_epochs is not None:
        s1.setdefault("trainer", {})["epochs"] = int(args.stage1_epochs)
    s1.setdefault("trainer", {})["save_period"] = 1
    s1_path = work / "stage1.json"
    _write_json(s1_path, s1)

    print(f"=== Stage 1 (μ MSE, σ frozen) → {parent}_s1 ===", flush=True)
    # Use subprocess-style via train.main for cleaner ConfigParser
    import subprocess

    cmd1 = [
        sys.executable,
        str(_ROOT / "train.py"),
        "-c",
        str(s1_path),
        "-id",
        f"{parent}_s1",
    ]
    subprocess.check_call(cmd1, cwd=str(_ROOT))
    s1_ckpt = _find_latest_ckpt(_ROOT / "saved", f"{parent}_s1")
    print(f"Stage 1 checkpoint: {s1_ckpt}", flush=True)

    # --- Stage 2 ---
    s2 = json.loads(json.dumps(base))
    s2["name"] = base.get("name", "nespreso") + f"_{parent}_s2"
    arch2 = s2.setdefault("arch", {}).setdefault("args", {})
    arch2["probabilistic"] = True
    arch2["n_quantiles"] = 9 if args.prob_mode == "quantile" else 0
    lc2 = s2.setdefault("loss_config", {})
    lc2["mode"] = "density_spice"
    lc2["prob_mode"] = args.prob_mode
    lc2["freeze_sigma"] = False
    # μ LR × 0.1 relative to base
    base_lr = float(s2.get("optimizer", {}).get("args", {}).get("lr", 1e-3))
    s2.setdefault("optimizer", {}).setdefault("args", {})["lr"] = base_lr * 0.1
    # Resume bumps start_epoch to stage1_epoch+1; total epochs must cover stage2 steps.
    s1_ep = int(args.stage1_epochs or base.get("trainer", {}).get("epochs", 2))
    s2_ep = int(args.stage2_epochs or base.get("trainer", {}).get("epochs", 2))
    s2.setdefault("trainer", {})["epochs"] = s1_ep + s2_ep
    s2.setdefault("trainer", {})["save_period"] = 1
    s2_path = work / "stage2.json"
    _write_json(s2_path, s2)

    print(f"=== Stage 2 ({args.prob_mode}, σ unfrozen, lr×0.1) → {parent}_s2 ===", flush=True)
    cmd2 = [
        sys.executable,
        str(_ROOT / "train.py"),
        "-c",
        str(s2_path),
        "-id",
        f"{parent}_s2",
        "-r",
        str(s1_ckpt),
    ]
    subprocess.check_call(cmd2, cwd=str(_ROOT))
    s2_ckpt = _find_latest_ckpt(_ROOT / "saved", f"{parent}_s2")
    manifest = {
        "parent_tag": parent,
        "prob_mode": args.prob_mode,
        "stage1_ckpt": str(s1_ckpt),
        "stage2_ckpt": str(s2_ckpt),
        "stage1_config": str(s1_path),
        "stage2_config": str(s2_path),
    }
    man_path = work / "manifest.json"
    _write_json(man_path, manifest)
    print(json.dumps(manifest, indent=2))
    print(f"wrote {man_path}")
    return 0


def _find_latest_ckpt(saved_root: Path, run_id_substr: str) -> Path:
    """Find model_best.pth or latest checkpoint under saved/*/run_id*."""
    candidates = []
    for p in saved_root.rglob("*.pth"):
        if run_id_substr in str(p):
            candidates.append(p)
    if not candidates:
        raise FileNotFoundError(f"no checkpoint containing {run_id_substr!r} under {saved_root}")
    # prefer model_best
    best = [p for p in candidates if p.name == "model_best.pth"]
    if best:
        return max(best, key=lambda p: p.stat().st_mtime)
    return max(candidates, key=lambda p: p.stat().st_mtime)


if __name__ == "__main__":
    raise SystemExit(main())
