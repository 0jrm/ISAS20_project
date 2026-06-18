#!/usr/bin/env python3
"""Plot train/val loss curves from trainer info.log files."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt


_LINE = re.compile(
    r"epoch\s+(\d+)/\d+\s+loss=([\d.eE+-]+)\s+.*?\bval_loss=([\d.eE+-]+)"
)


def parse_info_log(path: Path) -> tuple[list[int], list[float], list[float]]:
    epochs, train_loss, val_loss = [], [], []
    for line in path.read_text().splitlines():
        m = _LINE.search(line)
        if not m:
            continue
        epochs.append(int(m.group(1)))
        train_loss.append(float(m.group(2)))
        val_loss.append(float(m.group(3)))
    return epochs, train_loss, val_loss


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot NeSPReSO train/val loss curves")
    parser.add_argument("--run", action="append", nargs=2, metavar=("LABEL", "INFO_LOG"), required=True)
    parser.add_argument("-o", "--out", required=True, type=str)
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(10, 6))
    for label, log_path in args.run:
        epochs, train, val = parse_info_log(Path(log_path))
        if not epochs:
            raise ValueError(f"no epochs parsed from {log_path}")
        ax.plot(epochs, train, label=f"{label} train", alpha=0.85)
        ax.plot(epochs, val, label=f"{label} val", linestyle="--", alpha=0.85)

    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Decoder training curves (train / val)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
