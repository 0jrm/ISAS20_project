#!/usr/bin/env python3
"""Eval density_spice checkpoint → T/S (softplus+PCHIP+spice PCA+Newton) + frozen evalphys."""

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

# T1-A separate-PCA reconstruction RMSE on test (reports/t1_basis_stability.md).
T1_A_T_RMSE = {"0-50": 0.2025, "50-200": 0.2227, "200-800": 0.1065, ">800": 0.0162}
# Trained separate-PCA overall raw T RMSE (saved/eval_argo16_test.json).
ARGO16_T_RMSE_OVERALL = 0.4158


def _load_depth(cache: dict) -> np.ndarray:
    return np.asarray(cache["PRES"], dtype=np.float64).reshape(-1)


def decode_density_spice_to_ts(
    mu_raw: np.ndarray,
    cache: dict,
    *,
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    from evalphys.gsw_backend import get_gsw
    from evalphys.inversion import ts_from_sigma0_spice
    from model.density_spice import (
        decode_a_from_output,
        decode_sigma0_ctrl,
        decode_sigma0_from_scores,
        encode_a_from_sigma0_ctrl,
        project_monotone_sigma0_ctrl,
        upsample_pchip,
    )

    meta = cache["density_spice_meta"]
    pca = cache["pca_models"]["spice"]
    z_ctrl = np.asarray(meta["z_ctrl"], dtype=np.float64)
    dz = np.asarray(meta["dz_tilde"], dtype=np.float64)
    k = int(meta["K"])
    n_spice = int(meta["n_spice"])
    n_scores = int(meta.get("n_scores", k))
    basis = meta.get("delta_sigma0_basis")
    if basis is not None:
        # σ₀-space low-rank: clim + scores@V is NOT monotone by construction.
        # Stability = §3.6 opt-2 / T1-D: isotonic projection at inference (required).
        clim = np.asarray(meta.get("sigma0_clim", meta["sigma0_ctrl_mean"]), dtype=np.float64)
        sig_raw, z_tau = decode_sigma0_from_scores(
            mu_raw, clim, n_scores, n_spice, np.asarray(basis, dtype=np.float64)
        )
        neg_iface = np.diff(sig_raw, axis=1) < -1e-12
        sig_ctrl = project_monotone_sigma0_ctrl(sig_raw, z_ctrl)
        low_rank = True
        stability_guarantee = "inference_isotonic"  # not head_softplus
        proj_cost_sigma0_rmse = float(np.sqrt(np.mean((sig_ctrl - sig_raw) ** 2)))
        pre_iso = {
            "neg_iface_rate": float(neg_iface.mean()),
            "neg_profile_rate": float(neg_iface.any(axis=1).mean()),
            "n_neg_iface": int(neg_iface.sum()),
            "n_neg_profile": int(neg_iface.any(axis=1).sum()),
        }
    else:
        a_clim = meta.get("a_clim")
        if a_clim is None:
            a_clim = encode_a_from_sigma0_ctrl(
                np.asarray(meta["sigma0_ctrl_mean"], dtype=np.float64), dz, z_ctrl
            )
        a, z_tau = decode_a_from_output(
            mu_raw, np.asarray(a_clim, dtype=np.float64), n_scores, n_spice, basis=None
        )
        with torch.no_grad():
            sig_ctrl = decode_sigma0_ctrl(
                torch.from_numpy(a.astype(np.float32)),
                torch.from_numpy(dz.astype(np.float32)),
            ).numpy()
        low_rank = False
        stability_guarantee = "head_softplus_cumsum"
        proj_cost_sigma0_rmse = 0.0
        pre_iso = None

    depth = _load_depth(cache)
    sig_hat = upsample_pchip(sig_ctrl, z_ctrl, depth)

    tau_z = pca.inverse_transform(z_tau)
    tm = np.asarray(meta["spice_mean"], dtype=np.float64)
    ts = np.asarray(meta["spice_std"], dtype=np.float64)
    tau_hat = tau_z * ts + tm

    lat = np.asarray(cache["LAT"], dtype=np.float64)[indices]
    lon = np.asarray(cache["LON"], dtype=np.float64)[indices]
    n_prof, n_lev = sig_hat.shape[0], depth.size
    gsw = get_gsw()
    p = gsw.p_from_z(-np.broadcast_to(depth, (n_prof, n_lev)), lat[:, None])
    T_hat, S_hat, ok = ts_from_sigma0_spice(sig_hat, tau_hat, p, lon[:, None], lat[:, None])
    return T_hat, S_hat, {
        "inversion_fail_frac": float(1.0 - np.mean(ok)),
        "pre_inv_neg_dsigma0": int((np.diff(sig_hat, axis=1) < -1e-12).sum()),
        "n_scores": n_scores,
        "K": k,
        "low_rank": low_rank,
        "lowrank_target": meta.get("lowrank_target"),
        "stability_guarantee": stability_guarantee,
        "isotonic_projection_sigma0_rmse": proj_cost_sigma0_rmse,
        "pre_isotonic": pre_iso,
    }


def run_eval(cfg: dict, checkpoint: Path, split: str = "test") -> dict:
    from base.split_utils import build_split_indices
    from evalphys.calibration import season_from_juld
    from evalphys.gsw_backend import set_headline_frozen
    from evalphys.metrics import summarize_physical, ts_rmse_by_band
    from model.model import PatchConvMLP
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
    # deterministic head
    arch.setdefault("probabilistic", False)
    model = PatchConvMLP(**arch).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state, strict=False)
    model.eval()

    x = torch.tensor(cache["inputs"][idx], dtype=torch.float32, device=device)
    with torch.no_grad():
        out = model(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        out = out.detach().cpu().numpy()
    d = int(sum(cfg["outputs"].values()))
    mu_raw = out[:, :d] if out.shape[-1] >= d else out

    T_hat, S_hat, inv_info = decode_density_spice_to_ts(mu_raw, cache, indices=idx)
    depth = _load_depth(cache)
    T_true = np.asarray(cache["profiles"]["temperature"], dtype=np.float64).T[idx]
    S_true = np.asarray(cache["profiles"]["salinity"], dtype=np.float64).T[idx]
    lat = np.asarray(cache["LAT"], dtype=np.float64)[idx]
    lon = np.asarray(cache["LON"], dtype=np.float64)[idx]

    phys = summarize_physical(T_hat, S_hat, T_true, S_true, depth, lat, lon)
    s0 = phys["sigma0_monotonicity_pred"]
    t_rmse = phys["ts_rmse"]["T"]
    # overall T RMSE (all levels)
    overall_t = float(np.sqrt(np.nanmean((T_hat - T_true) ** 2)))

    upper = ("0-50", "50-200", "200-800")
    band_vs_A = {}
    gate_vs_A = True
    for b in upper:
        base = T1_A_T_RMSE[b]
        pred = t_rmse.get(b)
        ratio = (pred / base) if (pred is not None and base > 0) else None
        ok = ratio is not None and ratio <= 1.10
        band_vs_A[b] = {"pred": pred, "A_recon": base, "ratio": ratio, "pass_le_1_10": ok}
        if not ok:
            gate_vs_A = False
    # Low-rank δσ₀: hard mono is isotonic at eval (pre-inv); post-inv rate is
    # Newton fidelity, not control-grid failure (PLAN §3.2 σ₀-vs-N² note).
    pre_inv_neg = int(inv_info.get("pre_inv_neg_dsigma0", -1))
    s0_rate = float(s0.get("violation_rate_profile"))
    if inv_info.get("low_rank"):
        s0_ok = pre_inv_neg == 0
    else:
        s0_ok = s0_rate < 0.01
    # Corrected chrono floor (HANDOFF 2026-07-16): clean argo16 × 1.10.
    # Published-random 0.4158 kept for side-by-side only.
    CLEAN_CHRONO_T = 0.5367
    vs_argo16 = overall_t / ARGO16_T_RMSE_OVERALL
    vs_clean = overall_t / CLEAN_CHRONO_T
    skill_ok = vs_clean <= 1.10
    gate_pass = s0_ok and skill_ok

    seasons = season_from_juld(
        np.asarray(cache["JULD"])[idx], dataset_tag=cache.get("dataset_tag", "argo_v2")
    )
    return {
        "checkpoint": str(checkpoint),
        "cache": cache_path,
        "split": split,
        "n": int(idx.size),
        "inversion": inv_info,
        "physical": phys,
        "overall_T_rmse": overall_t,
        "gate": {
            "sigma0_profile_rate": s0_rate,
            "sigma0_pre_inv_neg": pre_inv_neg,
            "sigma0_near_zero": s0_ok,
            "overall_T_vs_argo16": {
                "pred": overall_t,
                "argo16_published_random": ARGO16_T_RMSE_OVERALL,
                "ratio": vs_argo16,
                "pass_le_1_10": vs_argo16 <= 1.10,
            },
            "overall_T_vs_clean_chrono": {
                "pred": overall_t,
                "argo16_clean_chrono": CLEAN_CHRONO_T,
                "floor": CLEAN_CHRONO_T * 1.10,
                "ratio": vs_clean,
                "pass_le_1_10": skill_ok,
            },
            "upper_ocean_T_vs_T1A_recon": band_vs_A,
            "T1A_recon_gate_pass": gate_vs_A,
            "pass": gate_pass,
            "note": (
                "STOP uses pre-inv σ₀ (low-rank) or post-inv near-zero (full-rank) + "
                "overall T ≤ clean-chrono argo16×1.10 (floor 0.5903). "
                "Published-random 0.4158 reported side-by-side only."
            ),
        },
        "season_counts": {s: int((seasons == s).sum()) for s in ("DJF", "MAM", "JJA", "SON")},
        "mu_raw_shape": list(mu_raw.shape),
        "_arrays": {  # for phase4 reuse when called in-process
            "T_hat": T_hat,
            "S_hat": S_hat,
            "mu_raw": mu_raw,
            "idx": idx,
            "cache": cache,
            "depth": depth,
            "lat": lat,
            "lon": lon,
            "T_true": T_true,
            "S_true": S_true,
        },
    }


def _md(data: dict) -> str:
    g = data["gate"]
    phys = data["physical"]
    t, s = phys["ts_rmse"]["T"], phys["ts_rmse"]["S"]
    s0 = phys["sigma0_monotonicity_pred"]
    n2 = phys["static_stability_pred"]["1e-08"]
    clean = g.get("overall_T_vs_clean_chrono") or {}
    lines = [
        "# Phase 3 — density_spice eval",
        "",
        f"**Checkpoint:** `{data['checkpoint']}`  ",
        f"**Cache:** `{data['cache']}`  ",
        f"**Split:** {data['split']} n={data['n']}",
        "",
        f"**Gate:** {'PASS' if g['pass'] else 'FAIL'}",
        "",
        f"- σ₀ profile rate (post-inv): {g['sigma0_profile_rate']:.4f}; "
        f"pre-inv neg: {g.get('sigma0_pre_inv_neg')}; near-zero OK: {g['sigma0_near_zero']}",
        f"- N² profile / level @ 1e-8: {n2['violation_rate_profile']:.4f} / {n2['violation_rate_level']:.6f}",
        f"- overall T RMSE: {data['overall_T_rmse']:.4f}",
    ]
    if clean:
        lines.append(
            f"- vs clean chrono argo16 {clean.get('argo16_clean_chrono')}: "
            f"ratio {clean.get('ratio'):.3f} (floor {clean.get('floor'):.4f}, "
            f"pass={clean.get('pass_le_1_10')})"
        )
    lines += [
        f"- vs published-random argo16: ratio {g['overall_T_vs_argo16']['ratio']:.3f} (side-by-side only)",
        f"- MLD RMSE: {phys['mld']['pred_vs_true']['rmse']}",
        f"- dρ/dz RMSE: {phys['drhodz_rmse']['rmse_overall']}",
        f"- inversion fail frac: {data['inversion']['inversion_fail_frac']:.4f}",
        "",
        "## T/S RMSE by depth band",
        "",
        "| band | T RMSE | S RMSE | vs T1-A recon |",
        "|------|--------|--------|---------------|",
    ]
    for b in ("0-50", "50-200", "200-800", ">800"):
        vs = g["upper_ocean_T_vs_T1A_recon"].get(b, {})
        ratio = vs.get("ratio")
        rtxt = f"{ratio:.3f}" if ratio is not None else "—"
        lines.append(f"| {b} | {t.get(b):.4f} | {s.get(b):.4f} | {rtxt} |")
    lines += [
        "",
        f"_Gate note:_ {g['note']}",
        "",
        "## Phase 2 caveat",
        "",
        "T2 stale gate OPEN (0% SSS/SST/SSH on val/test). Full HDF5 lacks v3 error fields "
        "(`err_sla` / `analysis_error` / `sos_error` only in `*_err_smoke.h5`). "
        "density_spice cache has no `inputs_err` — headline metrics may be SSS-confounded "
        "only if stale returns; currently not. Formal product-error channels not in model inputs.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("-r", "--checkpoint", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out-json", default="../reports/phase3_full_train_eval.json")
    ap.add_argument("--out-md", default="../reports/phase3_full_train_eval.md")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    data = run_eval(cfg, Path(args.checkpoint), split=args.split)
    # strip non-JSON arrays
    arrays = data.pop("_arrays", None)
    out_j, out_m = Path(args.out_json), Path(args.out_md)
    out_j.parent.mkdir(parents=True, exist_ok=True)
    out_j.write_text(json.dumps(data, indent=2, default=str) + "\n")
    out_m.write_text(_md(data))
    print(f"wrote {out_j} and {out_m}")
    print(f"GATE={'PASS' if data['gate']['pass'] else 'FAIL'}")
    del arrays
    return 0 if data["gate"]["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
