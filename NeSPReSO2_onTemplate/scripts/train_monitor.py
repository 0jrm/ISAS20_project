#!/usr/bin/env python3
"""Dual-run training status from status.json. --once only; stdlib."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ponytail: fixed thresholds; upgrade path = CLI flags if needed
WARN_STALL_SEC = 5 * 60
FAIL_STALL_SEC = 20 * 60
KILL_GRACE_SEC = 30


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_updated_at(value: str) -> datetime:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def save_dir_for(config_path: str | Path, run_id: str) -> Path:
    cfg = json.loads(Path(config_path).read_text())
    return Path(cfg["trainer"]["save_dir"]) / "models" / cfg["name"] / run_id


def read_status(save_dir: Path) -> dict | None:
    path = save_dir / "status.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def classify(status: dict | None) -> tuple[str, str]:
    if status is None:
        return "unknown", "no status.json"
    state = status.get("state", "unknown")
    if state == "done":
        return "done", status.get("reason") or "done"
    if state == "failed":
        return "failed", status.get("reason") or "failed"
    updated = status.get("updated_at")
    if not updated:
        return "warning", "missing updated_at"
    age = (_utc_now() - _parse_updated_at(updated)).total_seconds()
    if age > FAIL_STALL_SEC:
        return "stalled", f"no update for {int(age)}s"
    if age > WARN_STALL_SEC:
        return "warning", f"no update for {int(age)}s"
    return "running", "ok"


def kill_pids(pids: list[int], grace: int = KILL_GRACE_SEC) -> None:
    # ponytail: SIGTERM then SIGKILL; upgrade path = Slurm scancel by job id
    alive = [p for p in pids if p and _pid_alive(p)]
    for pid in alive:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.time() + grace
    while time.time() < deadline:
        alive = [p for p in alive if _pid_alive(p)]
        if not alive:
            return
        time.sleep(1)
    for pid in alive:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def summarize(manifest_path: Path, do_kill: bool) -> int:
    manifest = json.loads(manifest_path.read_text())
    run_root = manifest.get("run_id", "?")
    rows = []
    exit_code = 0
    kill_pids_list: list[int] = []

    for entry in manifest["runs"]:
        tag = entry["tag"]
        run_id = f"{manifest['run_id']}_{tag}"
        save_dir = save_dir_for(entry["config"], run_id)
        status = read_status(save_dir)
        level, detail = classify(status)
        epoch = status.get("epoch", "?") if status else "?"
        val_loss = status.get("val_loss", "?") if status else "?"
        rows.append((tag, level, epoch, val_loss, detail, save_dir))

        if level == "done":
            continue
        if level in ("running", "warning"):
            exit_code = max(exit_code, 1)
            continue
        exit_code = 2
        if entry.get("pid"):
            kill_pids_list.append(int(entry["pid"]))

    print(f"run_id={run_root}")
    for tag, level, epoch, val_loss, detail, save_dir in rows:
        print(f"  {tag:8s} {level:8s} epoch={epoch} val_loss={val_loss}  {detail}")
        print(f"           {save_dir}")

    if do_kill and kill_pids_list:
        kill_pids(kill_pids_list)
        print(f"killed pids: {kill_pids_list}")

    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description="NeSPReSO dual-run monitor")
    parser.add_argument("--once", action="store_true", help="print summary and exit")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--kill", action="store_true", help="SIGTERM then SIGKILL stalled/failed PIDs")
    args = parser.parse_args()
    if not args.once:
        parser.error("--once is required (ponytail: no --watch)")
    return summarize(args.manifest.resolve(), args.kill)


if __name__ == "__main__":
    sys.exit(main())
