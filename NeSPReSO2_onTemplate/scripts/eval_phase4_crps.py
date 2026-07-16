#!/usr/bin/env python3
"""Phase 4.8 — CRPS / ENCE / PIT / spread-skill / Spearman × depth band × season."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.eval_density_spice import decode_density_spice_to_ts, _load_depth  # noqa: E402

ENCE_MAX = 0.20
SPEARMAN_BASELINE = 0.12


def _band_masks(depth: np.ndarray) -> dict[str, np.ndarray]:
    from evalphys.constants import DEPTH_BAND_LABELS, DEPTH_BANDS

    z = np.asarray(depth, dtype=np.float64).reshape(-1)
    out = {}
    for label, (lo, hi) in zip(DEPTH_BAND_LABELS, DEPTH_BANDS):
        if np.isfinite(hi):
            out[label] = (z >= lo) & (z < hi)
        else:
            out[label] = z >= lo
    return out


def _cal_bundle(mu, sigma, y, depth=None):
    from evalphys.calibration import ence, gaussian_crps, pit_histogram, spread_skill

    crps = gaussian_crps(mu, sigma, y)
    ss = spread_skill(mu, sigma, y)
    en = ence(mu, sigma, y)
    pit = pit_histogram(mu, sigma, y)
    return {
        "crps_mean": float(np.nanmean(crps)),
        "ence": en.get("ence"),
        "pit_sup_dev": pit.get("sup_bin_deviation"),
        "spread_skill_slope": ss.get("slope_rmse_vs_sigma"),
        "spearman_sigma_abs_err": ss.get("spearman_sigma_abs_error"),
        "n": int(np.isfinite(mu).sum()),
    }


def run_phase4_eval(cfg: dict, checkpoint: Path, split: str = "test") -> dict:
    from base.split_utils import build_split_indices
    from evalphys.calibration import season_from_juld
    from evalphys.constants import DEPTH_BAND_LABELS
    from evalphys.gsw_backend import set_headline_frozen
    from evalphys.metrics import summarize_physical
    from model.density_spice import decode_sigma0_ctrl
    from model.model import PatchConvMLP
    from model.prob_head import split_mu_sigma
    from preproc.export_v2_cache import build_argo_cache

    set_headline_frozen(True)
    cache_path = build_argo_cache(cfg)
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)

    n = int(cache["inputs"].shape[0])
    dl = cfg["data_loader"]["args"]
    indices = build_split_indices(
        n,
        cache.get("JULD"),
        {
            "split_mode": dl.get("split_mode", "chronological"),
            "train_frac": float(dl.get("train_frac", 0.7)),
            "val_frac": float(dl.get("val_frac", 0.15)),
            "test_frac": float(dl.get("test_frac", 0.15)),
            "split_seed": int(dl.get("split_seed", 42)),
            "unassigned": dl.get("unassigned", "exclude"),
        },
        dataset_tag=cache.get("dataset_tag", "argo_v2"),
        v2_src=cfg.get("io", {}).get("v2_src"),
    )
    idx = np.asarray(indices[split], dtype=int)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arch = dict(cfg["arch"]["args"])
    arch["probabilistic"] = True
    arch.setdefault("n_quantiles", 0)
    model = PatchConvMLP(**arch).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state, strict=False)
    model.eval()

    x = torch.tensor(cache["inputs"][idx], dtype=torch.float32, device=device)
    d = int(sum(cfg["outputs"].values()))
    meta = cache["density_spice_meta"]
    dz = torch.tensor(meta["dz_tilde"], dtype=torch.float32, device=device)
    mu_s = torch.tensor(meta["sigma0_ctrl_mean"], dtype=torch.float32, device=device)
    sd_s = torch.tensor(np.maximum(meta["sigma0_ctrl_std"], 1e-6), dtype=torch.float32, device=device)
    k = int(meta["K"])

    with torch.no_grad():
        out = model(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        mu_raw, sigma_lat = split_mu_sigma(out, d)
        n_scores = int(cfg["outputs"]["density_ctrl"])
        n_spice = int(cfg["outputs"]["spice"])
        basis = meta.get("delta_sigma0_basis")
        # model forward already softplus+floor; do not softplus again
        if basis is not None:
            from model.density_spice import decode_sigma0_from_scores

            clim = meta.get("sigma0_clim", meta["sigma0_ctrl_mean"])
            clim_t = torch.tensor(clim, dtype=torch.float32, device=device)
            basis_t = torch.tensor(basis, dtype=torch.float32, device=device)
            sig, z_tau = decode_sigma0_from_scores(
                mu_raw, clim_t, n_scores, n_spice, basis_t
            )
            sig_z = (sig - mu_s) / sd_s
            mu = torch.cat([sig_z, z_tau], dim=-1)
            # Σ_ρ = V diag(σ_z²) Vᵀ → per-level std, then standardize
            sz = sigma_lat[:, :n_scores]
            st = sigma_lat[:, n_scores:]
            var = (basis_t.unsqueeze(0) ** 2) * (sz.unsqueeze(-1) ** 2)
            std_phys = torch.sqrt(var.sum(dim=1).clamp_min(1e-12))
            sigma = torch.cat([std_phys / sd_s, st], dim=-1)
        else:
            # μ in standardized target space; a = a_clim + δa
            from model.density_spice import encode_a_from_sigma0_ctrl

            a_clim = meta.get("a_clim")
            if a_clim is None:
                a_clim = encode_a_from_sigma0_ctrl(
                    np.asarray(meta["sigma0_ctrl_mean"], dtype=np.float64),
                    np.asarray(meta["dz_tilde"], dtype=np.float64),
                    np.asarray(meta["z_ctrl"], dtype=np.float64),
                )
            a_clim_t = torch.tensor(a_clim, dtype=torch.float32, device=device)
            a = mu_raw[:, :k] + a_clim_t
            z_tau = mu_raw[:, k:]
            sig = decode_sigma0_ctrl(a, dz)
            sig_z = (sig - mu_s) / sd_s
            mu = torch.cat([sig_z, z_tau], dim=-1)
            sigma = sigma_lat
        mu_np = mu.cpu().numpy()
        sig_np = sigma.cpu().numpy()
        mu_raw_np = mu_raw.cpu().numpy()

    y = np.asarray(cache["targets"][idx], dtype=np.float64)
    # Physical T/S for stability report
    T_hat, S_hat, inv = decode_density_spice_to_ts(mu_raw_np, cache, indices=idx)
    depth = _load_depth(cache)
    T_true = np.asarray(cache["profiles"]["temperature"], dtype=np.float64).T[idx]
    S_true = np.asarray(cache["profiles"]["salinity"], dtype=np.float64).T[idx]
    lat = np.asarray(cache["LAT"], dtype=np.float64)[idx]
    lon = np.asarray(cache["LON"], dtype=np.float64)[idx]
    phys = summarize_physical(T_hat, S_hat, T_true, S_true, depth, lat, lon)

    seasons = season_from_juld(
        np.asarray(cache["JULD"])[idx], dataset_tag=cache.get("dataset_tag", "argo_v2")
    )
    # Calibration on standardized (σ₀_ctrl || spice) — native depth bands N/A for ctrl.
    # Report: overall + by season; depth-band strata use spice-PC dims only as proxy via
    # physical-level CRPS if we expand — ponytail: calibrate in latent/standardized space
    # and stratify by season; depth bands applied to physical |err| vs σ mapped via ctrl.
    overall = _cal_bundle(mu_np, sig_np, y)

    by_season = {}
    for s in ("DJF", "MAM", "JJA", "SON"):
        m = seasons == s
        if not np.any(m):
            continue
        by_season[s] = _cal_bundle(mu_np[m], sig_np[m], y[m])

    # Depth-band × season: use density-ctrl columns only, map ctrl depths to bands
    z_ctrl = np.asarray(meta["z_ctrl"], dtype=np.float64)
    bands = _band_masks(z_ctrl)
    by_band = {}
    by_band_season = {}
    for blabel, bmask in bands.items():
        # density block only (first K)
        mu_b = mu_np[:, :k][:, bmask]
        sg_b = sig_np[:, :k][:, bmask]
        y_b = y[:, :k][:, bmask]
        by_band[blabel] = _cal_bundle(mu_b, sg_b, y_b)
        by_band_season[blabel] = {}
        for s in ("DJF", "MAM", "JJA", "SON"):
            m = seasons == s
            if not np.any(m):
                continue
            by_band_season[blabel][s] = _cal_bundle(mu_b[m], sg_b[m], y_b[m])

    ence_ok = overall["ence"] is not None and overall["ence"] < ENCE_MAX
    sp = overall["spearman_sigma_abs_err"]
    spearman_ok = sp is not None and sp > SPEARMAN_BASELINE

    return {
        "checkpoint": str(checkpoint),
        "cache": cache_path,
        "split": split,
        "n": int(idx.size),
        "inversion": inv,
        "physical_summary": {
            "sigma0_profile_rate": phys["sigma0_monotonicity_pred"]["violation_rate_profile"],
            "n2_profile_rate": phys["static_stability_pred"]["1e-08"]["violation_rate_profile"],
            "ts_rmse": phys["ts_rmse"],
            "mld_rmse": phys["mld"]["pred_vs_true"]["rmse"],
            "drhodz_rmse": phys["drhodz_rmse"]["rmse_overall"],
        },
        "calibration_space": "standardized_sigma0_ctrl_plus_spice_pcs",
        "overall": overall,
        "by_season": by_season,
        "by_depth_band_density_ctrl": by_band,
        "by_depth_band_x_season": by_band_season,
        "anchors": {
            "ence_max": ENCE_MAX,
            "ence_pass": ence_ok,
            "spearman_baseline": SPEARMAN_BASELINE,
            "spearman_pass": spearman_ok,
            "pass": ence_ok and spearman_ok,
        },
        "caveat": (
            "No inputs_err / input-error tercile stratum (Phase 2.2 full HDF5 blocker). "
            "T2 stale gate OPEN. Formal product errors are relative indicators only."
        ),
        "season_counts": {s: int((seasons == s).sum()) for s in ("DJF", "MAM", "JJA", "SON")},
    }


def _md(data: dict) -> str:
    o = data["overall"]
    a = data["anchors"]
    lines = [
        "# Phase 4 — full CRPS two-stage eval (4.8)",
        "",
        f"**Checkpoint:** `{data['checkpoint']}`  ",
        f"**Cache:** `{data['cache']}`  ",
        f"**Split:** {data['split']} n={data['n']}",
        "",
        f"**Anchors:** {'PASS' if a['pass'] else 'MISS'} "
        f"(ENCE < {a['ence_max']}: {'yes' if a['ence_pass'] else 'NO'}; "
        f"Spearman ≫ {a['spearman_baseline']}: {'yes' if a['spearman_pass'] else 'NO'})",
        "",
        "## Overall (standardized σ₀_ctrl + spice PCs)",
        "",
        f"| CRPS | ENCE | PIT sup-dev | spread-skill slope | σ–|err| Spearman |",
        f"|------|------|-------------|--------------------|-----------------|",
        f"| {o['crps_mean']:.4f} | {o['ence']} | {o['pit_sup_dev']} | "
        f"{o['spread_skill_slope']} | {o['spearman_sigma_abs_err']} |",
        "",
        "## By season",
        "",
        "| season | CRPS | ENCE | PIT | slope | Spearman | n |",
        "|--------|------|------|-----|-------|----------|---|",
    ]
    for s, v in data["by_season"].items():
        lines.append(
            f"| {s} | {v['crps_mean']:.4f} | {v['ence']} | {v['pit_sup_dev']} | "
            f"{v['spread_skill_slope']} | {v['spearman_sigma_abs_err']} | {v['n']} |"
        )
    lines += [
        "",
        "## By depth band (density ctrl only) × season",
        "",
    ]
    for band, seasons in data["by_depth_band_x_season"].items():
        lines.append(f"### {band} m")
        lines.append("")
        lines.append("| season | CRPS | ENCE | Spearman |")
        lines.append("|--------|------|------|----------|")
        for s, v in seasons.items():
            lines.append(
                f"| {s} | {v['crps_mean']:.4f} | {v['ence']} | {v['spearman_sigma_abs_err']} |"
            )
        lines.append("")
    phys = data["physical_summary"]
    lines += [
        "## Physical (point μ after inversion)",
        "",
        f"- σ₀ profile rate: {phys['sigma0_profile_rate']:.4f}",
        f"- N² profile rate: {phys['n2_profile_rate']:.4f}",
        f"- MLD RMSE: {phys['mld_rmse']}",
        f"- dρ/dz RMSE: {phys['drhodz_rmse']}",
        "",
        f"**Caveat:** {data['caveat']}",
        "",
    ]
    if not a["pass"]:
        lines += [
            "## Anchor miss",
            "",
        ]
        if not a["ence_pass"]:
            lines.append(f"- ENCE={o['ence']} (need < {a['ence_max']})")
        if not a["spearman_pass"]:
            lines.append(f"- Spearman={o['spearman_sigma_abs_err']} (need ≫ {a['spearman_baseline']})")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("-r", "--checkpoint", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out-json", default="../reports/phase4_full_eval.json")
    ap.add_argument("--out-md", default="../reports/phase4_full_eval.md")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    data = run_phase4_eval(cfg, Path(args.checkpoint), split=args.split)
    out_j, out_m = Path(args.out_json), Path(args.out_md)
    out_j.parent.mkdir(parents=True, exist_ok=True)
    out_j.write_text(json.dumps(data, indent=2, default=str) + "\n")
    out_m.write_text(_md(data))
    print(f"wrote {out_j} and {out_m}")
    print(f"ANCHORS={'PASS' if data['anchors']['pass'] else 'MISS'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
