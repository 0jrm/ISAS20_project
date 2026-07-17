#!/usr/bin/env python3
"""Phase 5 ablation matrix launcher (prereg: reports/ablation_preregistration.md).

Run order: C×CRPS×{42,43,44} first, then remaining core cells.
Prob cells: train_prob_twostage + val per-dim σ recalib; test scored once after freeze.
Error-channel axis: skipped until v3 HDF5 lands.
Env pin: conda-env.lock.yml + sha256 next to this manifest; asserted before launch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_REPO = _ROOT.parent
_MANIFEST_DIR = _ROOT / "saved" / "runs" / "phase5_matrix"
_ENV_LOCK = _MANIFEST_DIR / "conda-env.lock.yml"
_ENV_SHA = _MANIFEST_DIR / "conda-env.sha256"
# tracked mirror (git) — same bytes as _ENV_LOCK
_ENV_LOCK_TRACKED = _REPO / "reports" / "phase5_conda-env.lock.yml"
_ENV_SHA_TRACKED = _REPO / "reports" / "phase5_conda-env.sha256"
_SEEDS = (42, 43, 44)

_CFG = {
    "C": "config/argo/config_argo_densityspice_lowrank_crps.json",
    "A": "config/argo/config_argo.json",
    "B": "config/argo/config_argo_joint_eof.json",
}


def _read_pinned_sha() -> str:
    for p in (_ENV_SHA, _ENV_SHA_TRACKED):
        if p.is_file():
            return p.read_text().strip().split()[0]
    raise FileNotFoundError(
        f"missing env pin {_ENV_SHA} (and tracked {_ENV_SHA_TRACKED}); "
        "export conda env before matrix launch"
    )


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def assert_env_hash() -> str:
    """Fail closed if lockfile missing or sha mismatch (matrix reproducibility)."""
    lock = _ENV_LOCK if _ENV_LOCK.is_file() else _ENV_LOCK_TRACKED
    if not lock.is_file():
        raise FileNotFoundError(f"conda lockfile missing: tried {_ENV_LOCK} and {_ENV_LOCK_TRACKED}")
    pinned = _read_pinned_sha()
    got = _hash_file(lock)
    if got != pinned:
        raise RuntimeError(
            f"conda env lock hash mismatch:\n  pinned={pinned}\n  got   ={got}\n"
            f"  lock  ={lock}\nRefuse to launch; re-export or restore the pin."
        )
    return got


def _cells(*, only_rep: str | None, only_head: str | None):
    """Yield (rep, head, seed) in prereg run order."""
    ordered = []
    for rep in ("C", "A", "B"):
        for head in ("CRPS", "NLL", "det"):
            for seed in _SEEDS:
                ordered.append((rep, head, seed))
    for rep, head, seed in ordered:
        if only_rep and rep != only_rep:
            continue
        if only_head and head != only_head:
            continue
        yield rep, head, seed


def _cell_id(rep: str, head: str, seed: int) -> str:
    return f"p5_{rep}_{head}_s{seed}"


def _prepare_cfg(rep: str, head: str, seed: int, out: Path) -> Path | None:
    tmpl = _CFG.get(rep)
    if tmpl is None:
        return None
    cfg = json.loads((_ROOT / tmpl).read_text())
    cfg["seed"] = int(seed)
    cfg["name"] = f"NeSPReSO2_ARGO_GoM_p5_{rep}_{head}"
    cfg.setdefault("trainer", {})["save_dir"] = f"saved/phase5_matrix/{rep}_{head}/"
    arch = cfg.setdefault("arch", {}).setdefault("args", {})
    lc = cfg.setdefault("loss_config", {})
    if head == "det":
        arch["probabilistic"] = False
        if rep == "C":
            lc["mode"] = "density_spice"
            lc["prob_mode"] = "mse"
            lc["freeze_sigma"] = True
        else:
            lc.pop("prob_mode", None)
            lc.pop("freeze_sigma", None)
            if rep == "B":
                lc["mode"] = "combined"
    elif head == "CRPS":
        arch["probabilistic"] = True
        arch["n_quantiles"] = 0
        lc["mode"] = "density_spice" if rep == "C" else lc.get("mode", "combined")
        lc["prob_mode"] = "crps"
        lc["freeze_sigma"] = False
    elif head == "NLL":
        arch["probabilistic"] = True
        arch["n_quantiles"] = 0
        lc["mode"] = "density_spice" if rep == "C" else lc.get("mode", "combined")
        lc["prob_mode"] = "nll"
        lc["freeze_sigma"] = False
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cfg, indent=2) + "\n")
    return out


def _selfcheck() -> None:
    cells = list(_cells(only_rep="C", only_head="CRPS"))
    assert cells == [("C", "CRPS", s) for s in _SEEDS], cells
    assert _cell_id("C", "CRPS", 42) == "p5_C_CRPS_s42"
    assert _CFG["B"] is not None and (_ROOT / _CFG["B"]).is_file()
    sha = assert_env_hash()
    print(f"launch_matrix selfcheck OK (env sha={sha[:12]}…)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="write manifest only")
    ap.add_argument("--launch", action="store_true", help="submit srun jobs")
    ap.add_argument("--only", default=None, help="filter e.g. C,CRPS")
    ap.add_argument("--stage1-epochs", type=int, default=60)
    ap.add_argument("--stage2-epochs", type=int, default=190)
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--max-launch", type=int, default=3, help="cap concurrent/sequential launches")
    ap.add_argument("--skip-env-check", action="store_true", help="danger: skip lock hash assert")
    args = ap.parse_args()
    if args.selfcheck:
        _selfcheck()
        return 0

    env_sha = None
    if not args.skip_env_check:
        env_sha = assert_env_hash()
        print(f"env pin OK sha256={env_sha[:16]}…")

    only_rep = only_head = None
    if args.only:
        parts = [p.strip() for p in args.only.split(",")]
        if len(parts) >= 1 and parts[0]:
            only_rep = parts[0]
        if len(parts) >= 2 and parts[1]:
            only_head = parts[1]

    _MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    cfg_dir = _MANIFEST_DIR / "configs"
    rows = []
    for rep, head, seed in _cells(only_rep=only_rep, only_head=only_head):
        cid = _cell_id(rep, head, seed)
        cfg_path = cfg_dir / f"{cid}.json"
        prepared = _prepare_cfg(rep, head, seed, cfg_path)
        status = "ready" if prepared else "blocked_no_template"
        note = ""
        if rep == "B":
            note = "joint_eof io.representation=joint_eof; CombinedPCALoss + T/S decode"
        if head in ("CRPS", "NLL") and rep in ("A", "B"):
            note = (note + "; " if note else "") + (
                f"{rep} prob: hetero head on PC space (CombinedPCALoss; twostage is C-only)"
            )
        rows.append(
            {
                "id": cid,
                "rep": rep,
                "head": head,
                "seed": seed,
                "config": str(prepared) if prepared else None,
                "status": status,
                "note": note,
                "twostage": head in ("CRPS", "NLL") and rep == "C",
            }
        )

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "prereg": str(_REPO / "reports" / "ablation_preregistration.md"),
        "env_lock": str(_ENV_LOCK if _ENV_LOCK.is_file() else _ENV_LOCK_TRACKED),
        "env_sha256": env_sha,
        "seeds": list(_SEEDS),
        "stage1_epochs": args.stage1_epochs,
        "stage2_epochs": args.stage2_epochs,
        "error_channels": "deferred_until_v3_hdf5",
        "eval_rule": "val_only_selection_recalib; one_test_score_per_frozen_cell",
        "cells": rows,
    }
    man_path = _MANIFEST_DIR / "manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {man_path} ({len(rows)} cells)")

    if args.dry_run or not args.launch:
        for r in rows[:10]:
            print(f"  {r['id']}: {r['status']} {r.get('note','')}")
        if len(rows) > 10:
            print(f"  ... +{len(rows)-10} more")
        if not args.launch:
            print("pass --launch to submit (use --dry-run to only write manifest)")
        return 0

    # re-assert immediately before first GPU job
    if not args.skip_env_check:
        assert_env_hash()

    launched = 0
    for r in rows:
        if r["status"] != "ready":
            print(f"skip {r['id']}: {r['status']} — {r['note']}")
            continue
        if launched >= args.max_launch:
            print(f"max-launch={args.max_launch} reached; remaining stay pending in manifest")
            break
        if not args.skip_env_check:
            assert_env_hash()
        cid = r["id"]
        log = _MANIFEST_DIR / f"{cid}.log"
        in_job = bool(os.environ.get("SLURM_JOB_ID"))
        prefix = (
            []
            if in_job
            else ["srun", "--ntasks=1", "--cpus-per-task=8", "--gres=gpu:1"]
        )
        conda_prefix = [
            "conda",
            "run",
            "-n",
            "nespreso",
            "--no-capture-output",
            "env",
            "PYTHONUNBUFFERED=1",
            "python",
            "-u",
        ]
        if r["twostage"]:
            cmd = prefix + conda_prefix + [
                str(_ROOT / "scripts" / "train_prob_twostage.py"),
                "-c",
                r["config"],
                "--prob-mode",
                "crps" if r["head"] == "CRPS" else "nll",
                "--stage1-epochs",
                str(args.stage1_epochs),
                "--stage2-epochs",
                str(args.stage2_epochs),
                "--parent-tag",
                cid,
                "--workdir",
                str(_MANIFEST_DIR / "twostage" / cid),
            ]
        else:
            cmd = prefix + conda_prefix + [
                str(_ROOT / "train.py"),
                "-c",
                r["config"],
                "-id",
                cid,
            ]
        print(f"LAUNCH {cid}: {' '.join(cmd)}", flush=True)
        with log.open("w") as fh:
            fh.write(f"CMD {' '.join(cmd)}\n")
            fh.write(f"ENV_SHA {env_sha}\n")
            fh.flush()
            rc = subprocess.call(cmd, cwd=str(_ROOT), stdout=fh, stderr=subprocess.STDOUT)
        r["exit_code"] = rc
        r["status"] = "done" if rc == 0 else "failed"
        r["log"] = str(log)
        launched += 1
        man_path.write_text(json.dumps(manifest, indent=2) + "\n")
        if rc != 0:
            print(f"FAIL {cid} rc={rc}; stopping launch loop")
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
