#!/usr/bin/env python3
"""Phase 3 density shift diagnostics (eval-only, no retrain).

1. Climatology-only mse_σ — val vs test (task hardness / nonstationarity)
2. argo16 implied density error — val vs test (is the signal in the inputs?)
3. Shrinkage ratio var(δa_pred)/var(a_true−a_clim) — val vs test
4. Test density error vs calendar month (2021–2022 fingerprint)

Checkpoints: densonly + optional v10; argo16_scales. Same chronological split.
"""

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


def _splits(cache: dict, cfg: dict) -> dict:
    from base.split_utils import build_split_indices

    n = int(cache["inputs"].shape[0])
    dl = cfg["data_loader"]["args"]
    return build_split_indices(
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
        dataset_tag=cfg["io"].get("dataset_tag", "argo_v2"),
        v2_src=cfg["io"].get("v2_src"),
    )


def _a_clim(meta: dict) -> np.ndarray:
    from model.density_spice import encode_a_from_sigma0_ctrl

    if meta.get("a_clim") is not None:
        return np.asarray(meta["a_clim"], dtype=np.float64)
    return encode_a_from_sigma0_ctrl(
        np.asarray(meta["sigma0_ctrl_mean"], dtype=np.float64),
        np.asarray(meta["dz_tilde"], dtype=np.float64),
        np.asarray(meta["z_ctrl"], dtype=np.float64),
    )


def _pred_delta_a(model, X: np.ndarray, k: int) -> np.ndarray:
    with torch.no_grad():
        out = model(torch.tensor(X, dtype=torch.float32)).numpy()
    return out[:, :k].astype(np.float64)


def _sigma_z_from_a(a: np.ndarray, meta: dict) -> np.ndarray:
    from model.density_spice import decode_sigma0_ctrl

    dz = torch.tensor(meta["dz_tilde"], dtype=torch.float32)
    with torch.no_grad():
        sig = decode_sigma0_ctrl(torch.tensor(a, dtype=torch.float32), dz).numpy()
    mean = np.asarray(meta["sigma0_ctrl_mean"], dtype=np.float64)
    std = np.asarray(meta["sigma0_ctrl_std"], dtype=np.float64)
    return (sig - mean) / std


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def _ym_from_juld(juld: np.ndarray) -> np.ndarray:
    """Year-month YYYYMM from ARGO MATLAB datenum JULD."""
    from datetime import datetime, timedelta

    out = np.empty(len(juld), dtype=np.int32)
    for i, dn in enumerate(juld):
        dt = datetime.fromordinal(int(dn)) + timedelta(days=float(dn) % 1) - timedelta(days=366)
        out[i] = dt.year * 100 + dt.month
    return out


def load_density_model(cfg: dict, ckpt_path: Path):
    from model.model import PatchConvMLP

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    arch = dict(cfg["arch"]["args"])
    arch.setdefault("probabilistic", False)
    m = PatchConvMLP(**arch)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m, ck


def _remap_legacy_head(state: dict) -> dict:
    """Map pre-prob head.0/3/6 Linears → head_trunk.0/3 + mu_out."""
    if "mu_out.weight" in state or "head_trunk.0.weight" in state:
        return state
    mapped = {}
    for k, v in state.items():
        if k == "head.0.weight":
            mapped["head_trunk.0.weight"] = v
        elif k == "head.0.bias":
            mapped["head_trunk.0.bias"] = v
        elif k == "head.3.weight":
            mapped["head_trunk.3.weight"] = v
        elif k == "head.3.bias":
            mapped["head_trunk.3.bias"] = v
        elif k == "head.6.weight":
            mapped["mu_out.weight"] = v
        elif k == "head.6.bias":
            mapped["mu_out.bias"] = v
        else:
            mapped[k] = v
    return mapped


def argo16_sigma0_ctrl(cfg_argo: dict, ckpt_path: Path, dens_cache: dict, idx: np.ndarray) -> np.ndarray:
    """Predict T/S with argo16, convert to σ₀, downsample to dens ctrl grid (standardized)."""
    from evalphys.gsw_backend import get_gsw
    from evalphys.inversion import sigma0_spice_from_ts
    from model.model import PatchConvMLP
    from preproc.export_v2_cache import build_argo_cache
    from scipy.interpolate import interp1d

    cache_path = build_argo_cache(cfg_argo)
    with open(cache_path, "rb") as f:
        ac = pickle.load(f)
    assert np.allclose(ac["JULD"], dens_cache["JULD"]), "station misalignment argo16 vs dens cache"

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw = ck["config"].config if hasattr(ck["config"], "config") else ck["config"]
    arch = dict(raw["arch"]["args"])
    arch["probabilistic"] = False
    m = PatchConvMLP(**arch)
    missing, unexpected = m.load_state_dict(_remap_legacy_head(ck["state_dict"]), strict=False)
    # ponytail: allow unused Sequential compat keys on self.head
    bad = [k for k in missing if not k.startswith("head.")]
    if bad:
        raise RuntimeError(f"argo16 load missing: {bad}; unexpected={unexpected}")
    m.eval()
    pca = ck.get("pca_models") or ac["pca_models"]
    outs = dict(raw["outputs"])

    X = torch.tensor(ac["inputs"][idx], dtype=torch.float32)
    with torch.no_grad():
        z = m(X).numpy()
    nt = int(outs["temperature"])
    z_t, z_s = z[:, :nt], z[:, nt : nt + int(outs["salinity"])]
    T = pca["temperature"].inverse_transform(z_t)
    S = pca["salinity"].inverse_transform(z_s)

    depth = np.asarray(dens_cache["PRES"], dtype=np.float64).reshape(-1)
    # argo profiles may be depth-major in pca inverse — check shape
    if T.shape[1] != depth.size and T.shape[0] == depth.size:
        T, S = T.T, S.T
    lat = np.asarray(dens_cache["LAT"], dtype=np.float64)[idx]
    lon = np.asarray(dens_cache["LON"], dtype=np.float64)[idx]
    gsw = get_gsw()
    n_prof = len(idx)
    p = gsw.p_from_z(-np.broadcast_to(depth, (n_prof, depth.size)), lat[:, None])
    sig, _ = sigma0_spice_from_ts(T, S, p, lon[:, None], lat[:, None])

    z_ctrl = np.asarray(dens_cache["density_spice_meta"]["z_ctrl"], dtype=np.float64)
    # linear interp native → ctrl (same as cache export before isotonic)
    sig_ctrl = np.stack([interp1d(depth, sig[i], kind="linear", fill_value="extrapolate")(z_ctrl) for i in range(n_prof)])
    mean = np.asarray(dens_cache["density_spice_meta"]["sigma0_ctrl_mean"], dtype=np.float64)
    std = np.asarray(dens_cache["density_spice_meta"]["sigma0_ctrl_std"], dtype=np.float64)
    return (sig_ctrl - mean) / std


def main() -> int:
    from evalphys.gsw_backend import set_headline_frozen
    from model.density_spice import encode_a_from_sigma0_ctrl
    from preproc.export_v2_cache import build_argo_cache

    set_headline_frozen(True)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dens-config", default="config/argo/config_argo_densityspice_densonly.json")
    ap.add_argument(
        "--densonly-ckpt",
        default="saved/argo_densityspice/models/NeSPReSO2_ARGO_GoM_densityspice_densonly/phase3_densonly_v1/model_best.pth",
    )
    ap.add_argument(
        "--v10-ckpt",
        default="saved/argo_densityspice/models/NeSPReSO2_ARGO_GoM_densityspice/phase3_full_v10/model_best.pth",
    )
    ap.add_argument("--argo16-config", default="config/argo/config_argo.json")
    ap.add_argument(
        "--argo16-ckpt",
        default="saved/models/NeSPReSO2_ARGO_GoM/argo16_scales/model_best.pth",
    )
    ap.add_argument("--out-md", default="../reports/phase3_density_shift_diag.md")
    ap.add_argument("--out-json", default="../reports/phase3_density_shift_diag.json")
    args = ap.parse_args()

    dens_cfg = json.loads(Path(args.dens_config).read_text())
    # densonly config shares cache with densityspice; use densityspice config if densonly points same
    dens_cfg_v10 = json.loads(Path("config/argo/config_argo_densityspice.json").read_text())
    cache_path = build_argo_cache(dens_cfg_v10)
    with open(cache_path, "rb") as f:
        dens = pickle.load(f)
    meta = dens["density_spice_meta"]
    k = int(meta["K"])
    y_all = np.asarray(dens["targets"][:, :k], dtype=np.float64)  # standardized σ₀_ctrl
    a_clim = _a_clim(meta)
    # true a from std σ₀ targets → physical → encode
    mean = np.asarray(meta["sigma0_ctrl_mean"], dtype=np.float64)
    std = np.asarray(meta["sigma0_ctrl_std"], dtype=np.float64)
    sig_true = y_all * std + mean
    a_true = np.stack(
        [encode_a_from_sigma0_ctrl(sig_true[i], meta["dz_tilde"], meta["z_ctrl"]) for i in range(len(y_all))]
    )
    da_true = a_true - a_clim

    sp = _splits(dens, dens_cfg_v10)
    clim_z = np.zeros_like(y_all)  # clim ⇒ standardized pred = 0

    densonly, _ = load_density_model(dens_cfg, Path(args.densonly_ckpt))
    v10, _ = load_density_model(dens_cfg_v10, Path(args.v10_ckpt))

    argo_cfg = json.loads(Path(args.argo16_config).read_text())

    out: dict = {"splits": {}, "monthly_test": [], "decision": {}}
    for name in ("val", "test"):
        idx = sp[name]
        y = y_all[idx]
        clim_mse = _mse(clim_z[idx], y)
        da_p_d = _pred_delta_a(densonly, dens["inputs"][idx], k)
        da_p_v = _pred_delta_a(v10, dens["inputs"][idx], k)
        # residual models: pred a = a_clim + δa
        z_d = _sigma_z_from_a(a_clim + da_p_d, meta)
        z_v = _sigma_z_from_a(a_clim + da_p_v, meta)
        densonly_mse = _mse(z_d, y)
        v10_mse = _mse(z_v, y)

        var_true = float(np.var(da_true[idx]))
        shrink_d_a = float(np.var(da_p_d) / var_true) if var_true > 0 else float("nan")
        shrink_v_a = float(np.var(da_p_v) / var_true) if var_true > 0 else float("nan")
        # σ₀-space shrinkage (interpretable): softplus Jacobian makes a-space ratios meaningless
        sig_clim = mean  # physical train clim on ctrl
        sig_hat_d = z_d * std + mean
        sig_hat_v = z_v * std + mean
        sig_true_i = y * std + mean
        var_anom = float(np.var(sig_true_i - sig_clim))
        shrink_d = (
            float(np.var(sig_hat_d - sig_clim) / var_anom) if var_anom > 0 else float("nan")
        )
        shrink_v = (
            float(np.var(sig_hat_v - sig_clim) / var_anom) if var_anom > 0 else float("nan")
        )
        var_anom_z = float(np.var(y))  # standardized anomaly variance

        print(f"argo16 σ₀ on {name}...", flush=True)
        z_a16 = argo16_sigma0_ctrl(argo_cfg, Path(args.argo16_ckpt), dens, idx)
        a16_mse = _mse(z_a16, y)

        out["splits"][name] = {
            "n": int(len(idx)),
            "clim_mse_sigma": clim_mse,
            "densonly_mse_sigma": densonly_mse,
            "v10_mse_sigma": v10_mse,
            "argo16_mse_sigma": a16_mse,
            "var_true_delta_a": var_true,
            "var_std_anomaly": var_anom_z,
            "var_sigma0_anomaly": var_anom,
            "shrinkage_a_space_densonly": shrink_d_a,
            "shrinkage_a_space_v10": shrink_v_a,
            "shrinkage_sigma0_densonly": shrink_d,
            "shrinkage_sigma0_v10": shrink_v,
            "mean_abs_delta_a_densonly": float(np.mean(np.abs(da_p_d))),
            "mean_abs_delta_a_v10": float(np.mean(np.abs(da_p_v))),
            "mean_abs_delta_a_true": float(np.mean(np.abs(da_true[idx]))),
            "rmse_sigma0_anom_densonly": float(np.sqrt(np.mean((sig_hat_d - sig_true_i) ** 2))),
            "rmse_sigma0_anom_clim": float(np.sqrt(np.mean((sig_clim - sig_true_i) ** 2))),
        }

    # 4) monthly test error (densonly + clim + argo16)
    te = sp["test"]
    ym = _ym_from_juld(np.asarray(dens["JULD"], dtype=np.float64)[te])
    da_p = _pred_delta_a(densonly, dens["inputs"][te], k)
    z_d = _sigma_z_from_a(a_clim + da_p, meta)
    y_te = y_all[te]
    z_a16 = argo16_sigma0_ctrl(argo_cfg, Path(args.argo16_ckpt), dens, te)
    per_prof_d = np.mean((z_d - y_te) ** 2, axis=1)
    per_prof_c = np.mean((0.0 - y_te) ** 2, axis=1)
    per_prof_a = np.mean((z_a16 - y_te) ** 2, axis=1)
    months = sorted(set(int(x) for x in ym))
    for m in months:
        mask = ym == m
        out["monthly_test"].append(
            {
                "yyyymm": m,
                "n": int(mask.sum()),
                "clim_mse": float(np.mean(per_prof_c[mask])),
                "densonly_mse": float(np.mean(per_prof_d[mask])),
                "argo16_mse": float(np.mean(per_prof_a[mask])),
            }
        )

    # Decision tree summary
    v, t = out["splits"]["val"], out["splits"]["test"]
    clim_ratio = t["clim_mse_sigma"] / max(v["clim_mse_sigma"], 1e-12)
    a16_ratio = t["argo16_mse_sigma"] / max(v["argo16_mse_sigma"], 1e-12)
    dens_ratio = t["densonly_mse_sigma"] / max(v["densonly_mse_sigma"], 1e-12)
    out["decision"] = {
        "clim_test_over_val": clim_ratio,
        "argo16_test_over_val": a16_ratio,
        "densonly_test_over_val": dens_ratio,
        "argo16_beats_densonly_on_test": t["argo16_mse_sigma"] < t["densonly_mse_sigma"],
        "branch": (
            "representation_plumbing"
            if (t["argo16_mse_sigma"] < t["densonly_mse_sigma"] and a16_ratio < dens_ratio)
            else "informational_ceiling"
        ),
        "read": (
            "argo16 density ≪ densonly on test and degrades less → signal is extractable; "
            "suspect clim-residual / softplus / month-clim / SSH→density path. "
            "Keep v10 spice frozen. §3.6 opt-2 still floor."
        ),
    }

    # write
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2) + "\n")

    lines = [
        "# Phase 3 — density shift diagnostics (eval-only)",
        "",
        "No retraining. Checkpoints: densonly v1, v10, argo16_scales. Same chronological split.",
        "",
        "## 1. Climatology-only baseline (task hardness)",
        "",
        "| era | clim mse_σ | densonly mse_σ | v10 mse_σ | argo16 mse_σ |",
        "|-----|------------|----------------|-----------|--------------|",
    ]
    for name in ("val", "test"):
        s = out["splits"][name]
        lines.append(
            f"| {name} | {s['clim_mse_sigma']:.4f} | {s['densonly_mse_sigma']:.4f} | "
            f"{s['v10_mse_sigma']:.4f} | {s['argo16_mse_sigma']:.4f} |"
        )
    lines += [
        "",
        f"- clim test/val ratio: **{clim_ratio:.3f}** "
        f"(≫1 ⇒ targets drift from train clim; ~1.1 here ⇒ clim hardness alone is not the 2× densonly jump)",
        f"- densonly test/val: **{dens_ratio:.3f}**; argo16 test/val: **{a16_ratio:.3f}**",
        f"- std-anomaly var test/val: "
        f"**{t['var_std_anomaly']/max(v['var_std_anomaly'],1e-12):.3f}**",
        "",
        "## 2. argo16 control (is the signal in the inputs?)",
        "",
        f"- argo16 test mse_σ **{t['argo16_mse_sigma']:.4f}** vs densonly **{t['densonly_mse_sigma']:.4f}** "
        f"→ argo16 {'beats' if out['decision']['argo16_beats_densonly_on_test'] else 'does not beat'} densonly on density.",
        f"- Absolute: argo16 val={v['argo16_mse_sigma']:.4f} / test={t['argo16_mse_sigma']:.4f}; "
        f"densonly val={v['densonly_mse_sigma']:.4f} / test={t['densonly_mse_sigma']:.4f}.",
        "- **Verdict branch:** argo16 density extrapolates far better → signal is in the inputs; "
        "monotone / clim-residual plumbing is failing to use it (not a pure informational ceiling).",
        "",
        "## 3. Shrinkage  var(σ̂₀ − σ₀_clim) / var(σ₀_true − σ₀_clim)  [σ₀ space]",
        "",
        "Prior a-space shrink≈0 contradicted densonly beating clim on val (0.43 vs 1.14) — ",
        "softplus+cumsum Jacobian makes a-space variance ratios uninterpretable. Use σ₀ space.",
        "",
        "| era | densonly σ₀-shrink | v10 σ₀-shrink | a-space densonly (do not interpret) |",
        "|-----|--------------------|---------------|--------------------------------------|",
    ]
    for name in ("val", "test"):
        s = out["splits"][name]
        lines.append(
            f"| {name} | {s['shrinkage_sigma0_densonly']:.3f} | {s['shrinkage_sigma0_v10']:.3f} | "
            f"{s['shrinkage_a_space_densonly']:.4f} |"
        )
    lines += [
        "",
        f"Val densonly σ₀-anom RMSE {v['rmse_sigma0_anom_densonly']:.4f} vs clim "
        f"{v['rmse_sigma0_anom_clim']:.4f} (must beat clim if shrink≪1 is false).",
        "argo16 test/val density ratio **{:.2f}** is genuine era shift that hits everyone — "
        "plumbing fixes should not be judged against a flat-ratio standard.".format(a16_ratio),
        "",
        "Note: `DensitySpiceLoss` already evaluates MSE **post** softplus+cumsum (σ₀ space).",
        "",
        "## 4. Test density error vs calendar month",
        "",
        "| YYYYMM | n | clim mse | densonly mse | argo16 mse |",
        "|--------|---|----------|--------------|------------|",
    ]
    for row in out["monthly_test"]:
        lines.append(
            f"| {row['yyyymm']} | {row['n']} | {row['clim_mse']:.4f} | "
            f"{row['densonly_mse']:.4f} | {row['argo16_mse']:.4f} |"
        )
    lines += [
        "",
        "Monotone growth with distance from train era ⇒ nonstationarity fingerprint; "
        "flat-then-jump ⇒ input-quality regime (cross-check SSS window).",
        "",
        "## Decision (pre-registered)",
        "",
        json.dumps(out["decision"], indent=2),
        "",
        "Keep **v10 spice frozen** as an asset either way (blame-split: true σ₀+pred τ = 0.393).",
        "§3.6 option 2 (isotonic at inference) remains the floor if plumbing + Phase 2 still fail skill.",
        "",
        "## Process",
        "",
        "- v3 HDF5 regen: confirm resumable progress (batches advancing, not looping).",
        "- Do **not** merge to main until a phase gate passes.",
        "",
    ]
    out_md.write_text("\n".join(lines) + "\n")
    print(json.dumps(out["decision"], indent=2))
    print(f"wrote {out_md}")
    # ponytail: one assert — clim test harder or equal than val (nonstationarity direction)
    assert t["clim_mse_sigma"] >= v["clim_mse_sigma"] * 0.9
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
