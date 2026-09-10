#!/usr/bin/env python3
"""Permute-column importance on chrono test. Frozen checkpoint, no retrain.

Layout after ONI/RONI splice (geoff / z32-ops):
  [6 harmonics][oni, roni][sss, sst, ssh][19 operators]
  n_enc=11 sees harmonics+ENSO+sat; n_sat=19 is Linear(ops→128), not SST.
9-d stoch_eof: [6 harmonics][sss, sst, ssh].
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from preproc.export_heave_ablation_cache import OP_NAMES
from preproc.preproc_isas_sat import ENCODING_KEYS, SAT_KEYS

HARM = list(ENCODING_KEYS)
SAT = list(SAT_KEYS)


def column_names(n_enc: int, n_sat: int) -> list[str]:
    if n_enc == 6 and n_sat == 3:
        return HARM + SAT
    if n_enc == 11 and n_sat == 19:
        return HARM + ["oni", "roni"] + SAT + list(OP_NAMES)
    raise ValueError(f"no column map for n_enc={n_enc} n_sat={n_sat}")


def groups_for(n_enc: int, n_sat: int) -> dict[str, list[int]]:
    names = column_names(n_enc, n_sat)
    idx = {n: i for i, n in enumerate(names)}

    def cols(*keys):
        return [idx[k] for k in keys]

    g = {
        "time": cols("timecos", "timesin"),
        "lat": cols("latcos", "latsin"),
        "lon": cols("loncos", "lonsin"),
        "space": cols("latcos", "latsin", "loncos", "lonsin"),
        "sss": cols("sss"),
        "sst": cols("sst"),
        "ssh": cols("ssh"),
        "sat": cols("sss", "sst", "ssh"),
        "all": list(range(len(names))),
    }
    if "oni" in idx:
        g["oni"] = cols("oni")
        g["roni"] = cols("roni")
        g["enso"] = cols("oni", "roni")
        g["ops"] = [idx[n] for n in OP_NAMES]
        g["ops_sst_grad"] = [idx[n] for n in OP_NAMES if n.startswith("sst.grad")]
        g["ops_sss_grad"] = [idx[n] for n in OP_NAMES if n.startswith("sss.grad")]
        g["ops_ssh_grad"] = [idx[n] for n in OP_NAMES if n.startswith("ssh.grad")]
        g["ops_ssh_lap"] = [idx[n] for n in OP_NAMES if "laplacian" in n]
        g["ops_tendency"] = [idx[n] for n in OP_NAMES if "tendency" in n]
        g["ops_geo"] = [idx[n] for n in OP_NAMES if ".geo_" in n]
        for n in OP_NAMES:
            g[f"op:{n}"] = [idx[n]]
    return g


def permute_cols(x: np.ndarray, cols: list[int], rng: np.random.Generator) -> np.ndarray:
    out = np.array(x, copy=True)
    perm = rng.permutation(out.shape[0])
    out[:, cols] = out[perm][:, cols]
    return out


def _rmse(mu, y, z, sl):
    e = mu[:, sl] - y[:, sl]
    out = {"all": float(np.sqrt(np.nanmean(e**2)))}
    from evalphys.constants import DEPTH_BAND_LABELS, DEPTH_BANDS

    for label, (lo, hi) in zip(DEPTH_BAND_LABELS, DEPTH_BANDS):
        m = (z >= lo) & (z < hi) if np.isfinite(hi) else (z >= lo)
        out[label] = float(np.sqrt(np.nanmean(e[:, m] ** 2)))
    return out


def main() -> int:
    from scripts.eval_acrps_phys import _predict
    import model.model as module_arch
    from model.loss import make_loss, uses_native_z_profiles
    from parse_config import ConfigParser, validate_config
    from train import ensure_cache, set_seed
    from base.util import prepare_device, read_json

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("-r", "--checkpoint", required=True)
    ap.add_argument("--out", default="../reports/eval_input_ablation.json")
    ap.add_argument("--md", default="")
    ap.add_argument("--compare", default="")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg_dict = read_json(args.config)
    validate_config(cfg_dict)
    config = ConfigParser(cfg_dict, run_id="")
    set_seed(config.config.get("seed", 42))
    ensure_cache(config)
    device, _ = prepare_device(config["n_gpu"])
    model = config.init_obj("arch", module_arch).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)))
    model.eval()

    import data_loader.data_loaders as module_data
    from base.split_utils import build_split_indices
    from collections import OrderedDict

    dl0 = dict(config["data_loader"]["args"])
    dl0["split"] = "test"
    dl0["shuffle"] = False
    ip = config.config.get("input_params")
    if ip:
        dl0["input_params"] = ip
    loader = getattr(module_data, config["data_loader"]["type"])(**dl0)
    cache_obj = loader.cache
    outputs = OrderedDict(config["outputs"])
    train_idx = build_split_indices(
        cache_obj["inputs"].shape[0],
        cache_obj.get("JULD"),
        dl0,
        dataset_tag=cache_obj.get("dataset_tag", "unknown"),
        v2_src=dl0.get("v2_src"),
    )["train"]
    loss_cfg = config.config.get("loss_config") or {}
    loss_fn = make_loss(
        pca_models=cache_obj.get("pca_models"),
        outputs=outputs,
        weights=cache_obj.get("weights"),
        device=device,
        loss_config=loss_cfg,
        loss_scales=config.config.get("loss_scales"),
        pres_levels=cache_obj.get("PRES"),
        true_profiles=(
            cache_obj.get("profiles")
            if uses_native_z_profiles(loss_cfg) or str(loss_cfg.get("crps_space", "pca")) == "stoch_eof"
            else None
        ),
        targets=cache_obj.get("targets"),
        train_idx=train_idx,
        lat=cache_obj.get("LAT"),
        lon=cache_obj.get("LON"),
        clim_profiles=cache_obj.get("clim_profiles"),
    )

    n_enc = int(config["arch"]["args"]["n_enc"])
    n_sat = int(config["arch"]["args"]["n_sat"])
    names = column_names(n_enc, n_sat)
    groups = groups_for(n_enc, n_sat)

    xs = []
    with torch.no_grad():
        for data, _target, _indices in loader:
            xs.append(data.cpu().numpy())
    x = np.concatenate(xs, axis=0)
    if x.shape[1] != len(names):
        raise ValueError(f"inputs {x.shape[1]} != names {len(names)}")

    mu0, _sg, y0, z, n_test = _predict(config, model, loss_fn, "test", device)
    nt = mu0.shape[1] // 2
    base_t = _rmse(mu0, y0, z, slice(0, nt))
    base_s = _rmse(mu0, y0, z, slice(nt, None))

    def forward_x(arr):
        mus = []
        i0 = 0
        with torch.no_grad():
            for data, _target, indices in loader:
                b = data.shape[0]
                xb = torch.as_tensor(arr[i0 : i0 + b], device=device)
                out = model(xb)
                if hasattr(loss_fn, "phys_mu_sigma") and (
                    getattr(loss_fn, "identity", False) or getattr(loss_fn, "raw_targets", False)
                ):
                    mu, _ = loss_fn.phys_mu_sigma(out, indices.to(device))
                else:
                    mu = loss_fn._mu_from_raw(out[:, : loss_fn.d])
                mus.append(mu.cpu().numpy())
                i0 += b
        return np.concatenate(mus, axis=0)

    rng = np.random.default_rng(args.seed)
    rows = {}
    for name, cols in groups.items():
        mu = forward_x(permute_cols(x, cols, rng))
        t = _rmse(mu, y0, z, slice(0, nt))
        rows[name] = {
            "n_col": len(cols),
            "cols": cols,
            "dT": t["all"] - base_t["all"],
            "dT_0-50": t["0-50"] - base_t["0-50"],
            "T": t["all"],
            "T_0-50": t["0-50"],
        }

    ranked = sorted(rows.items(), key=lambda kv: -kv[1]["dT"])
    payload = {
        "checkpoint": str(args.checkpoint),
        "n_test": n_test,
        "n_enc": n_enc,
        "n_sat": n_sat,
        "names": names,
        "baseline_T": base_t,
        "baseline_S": base_s,
        "note": (
            "Joint row-shuffle of the named columns on test. "
            "dT>0 means the trained net used that group. Not a retrain LOO."
        ),
        "ranked_dT": [{"name": k, **v} for k, v in ranked],
        "groups": rows,
    }
    out = Path(args.out)
    if not out.is_absolute():
        out = _ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    md = Path(args.md) if args.md else out.with_suffix(".md")
    if not md.is_absolute():
        md = _ROOT / md
    _write_ablation_md(md, payload, compare_path=args.compare)
    print(f"baseline T RMSE {base_t['all']:.4f}  0-50 {base_t['0-50']:.4f}  n={n_test}")
    for k, v in ranked[:12]:
        print(f"  {v['dT']:+.4f}  {k}")
    print(f"wrote {out}")
    print(f"wrote {md}")
    return 0


def _write_ablation_md(path: Path, payload: dict, compare_path: str = "") -> None:
    ranked = payload["ranked_dT"]
    bt = payload["baseline_T"]
    lines = [
        "# Input permutation importance",
        "",
        f"Checkpoint `{payload['checkpoint']}`. Chronological test n={payload['n_test']}.",
        "Joint row-shuffle of the named columns. ΔT > 0 means the trained net used that group.",
        "Not a retrain leave-one-out.",
        "",
        f"Baseline T RMSE = {bt['all']:.3f} (0–50 m {bt['0-50']:.3f}). "
        f"S RMSE = {payload['baseline_S']['all']:.3f}.",
        "",
        "| group | n | ΔT | T | ΔT 0–50 m |",
        "|-------|--:|---:|--:|----------:|",
    ]
    skip = {"all"}
    for row in ranked:
        if row["name"] in skip:
            continue
        if row["name"].startswith("op:") and abs(row["dT"]) < 0.01:
            continue
        lines.append(
            f"| {row['name']} | {row['n_col']} | {row['dT']:+.3f} | {row['T']:.3f} | {row['dT_0-50']:+.3f} |"
        )
    cmp_path = Path(compare_path) if compare_path else None
    if cmp_path and not cmp_path.is_absolute():
        cmp_path = _ROOT / cmp_path
    if cmp_path and cmp_path.is_file():
        other = json.loads(cmp_path.read_text())
        by = {r["name"]: r for r in other.get("ranked_dT", [])}
        keys = [
            "sat", "ssh", "sst", "sss", "ops", "ops_geo", "roni", "oni", "enso",
            "time", "space", "lon", "lat",
        ]
        lines += [
            "",
            f"## vs `{cmp_path.name}`",
            "",
            "| group | ΔT this | ΔT other | this − other |",
            "|-------|--------:|---------:|-------------:|",
        ]
        this = {r["name"]: r for r in ranked}
        for k in keys:
            if k not in this or k not in by:
                continue
            d0, d1 = this[k]["dT"], by[k]["dT"]
            lines.append(f"| {k} | {d0:+.3f} | {d1:+.3f} | {d0 - d1:+.3f} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
