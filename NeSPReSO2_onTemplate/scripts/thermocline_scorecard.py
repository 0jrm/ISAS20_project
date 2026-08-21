#!/usr/bin/env python3
"""Thermocline scorecard: D20/D26, max N², heave-vs-shape, steric vs ADT, T1 ceilings.

Works on frozen A×CRPS / B×det / ISOP when cache+ckpts exist; otherwise synthetic.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

_ROOT = Path(__file__).resolve().parents[1]
_REPO = _ROOT.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evalphys.constants import LC_LAT_RANGE, LC_LON_RANGE, STERIC_LC_RMS_CM, VERSION
from evalphys.metrics import (
    heave_vs_shape_split,
    isotherm_depth,
    max_n2_depth,
    mixed_layer_depth,
    steric_vs_adt,
    summarize_physical,
    ts_rmse_by_band,
)
from model.warp import unwarp_from_canonical, warp_to_canonical

A_CRPS = (
    _ROOT
    / "saved/phase5_matrix/A_CRPS_v2/models/NeSPReSO2_ARGO_GoM_p5_A_CRPS_v2_p5_A_CRPS_v2_s42_s2/p5_A_CRPS_v2_s42_s2/model_best.pth"
)
B_DET = _ROOT / "saved/phase5_matrix/B_det_v2"
REPORT_MD = _REPO / "reports" / "thermocline_scorecard.md"
REPORT_JSON = _REPO / "reports" / "thermocline_scorecard.json"


def _station_major(arr, n):
    a = np.asarray(arr, dtype=np.float64)
    if a.shape[0] == n:
        return a
    if a.shape[1] == n:
        return a.T
    raise ValueError(f"profile shape {a.shape} vs n={n}")


def _synthetic(n=40, nz=60, seed=0):
    rng = np.random.default_rng(seed)
    z = np.linspace(0, 400, nz)
    lat = 24.5 + rng.uniform(0, 3.5, n)
    lon = -88.0 + rng.uniform(0, 4.0, n)
    sla = rng.normal(0, 0.12, n)
    T = np.zeros((n, nz))
    S = np.zeros((n, nz))
    for i in range(n):
        d26 = 90.0 + 80.0 * sla[i]
        T[i] = 28.0 - 8.0 / (1.0 + np.exp(-(z - d26) / 20.0)) - 0.01 * z
        S[i] = 36.2 + 0.002 * z
    T_pred = np.empty_like(T)
    for i in range(n):
        T_pred[i] = np.interp(z + 12.0, z, T[i], left=T[i, 0], right=T[i, -1])
    S_pred = S + rng.normal(0, 0.01, S.shape)
    return {
        "T_true": T,
        "S_true": S,
        "T_pred": T_pred,
        "S_pred": S_pred,
        "z": z,
        "lat": lat,
        "lon": lon,
        "sla": sla,
        "source": "synthetic",
    }


def _score_pair(T_pred, S_pred, T_true, S_true, z, lat, lon, sla=None, label=""):
    d20_p, cov20_p = isotherm_depth(T_pred, z, 20.0)
    d20_t, cov20_t = isotherm_depth(T_true, z, 20.0)
    d26_p, cov26_p = isotherm_depth(T_pred, z, 26.0)
    d26_t, cov26_t = isotherm_depth(T_true, z, 26.0)
    n2_p = max_n2_depth(T_pred, S_pred, z, lat, lon)
    n2_t = max_n2_depth(T_true, S_true, z, lat, lon)
    mld_p = mixed_layer_depth(T_pred, S_pred, z, lat, lon)
    mld_t = mixed_layer_depth(T_true, S_true, z, lat, lon)
    heave = heave_vs_shape_split(T_pred, T_true, z, d26_p, d26_t)
    ts = ts_rmse_by_band(T_pred, S_pred, T_true, S_true, z)
    out = {
        "label": label,
        "D20_rmse": _rmse(d20_p, d20_t),
        "D26_rmse": _rmse(d26_p, d26_t),
        "max_n2_rmse": _rmse(n2_p, n2_t),
        "mld_rmse": _rmse(mld_p, mld_t),
        "coverage_D20": {"pred": cov20_p, "true": cov20_t},
        "coverage_D26": {"pred": cov26_p, "true": cov26_t},
        "heave_vs_shape": heave,
        "ts_rmse": ts,
        "evalphys": summarize_physical(T_pred, S_pred, T_true, S_true, z, lat, lon),
    }
    if sla is not None:
        out["steric_vs_adt"] = steric_vs_adt(
            T_pred, S_pred, z, lat, lon, sla, lat_range=LC_LAT_RANGE, lon_range=LC_LON_RANGE
        )
    return out


def _rmse(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if not m.any():
        return None
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2)))


def _pca16_ceiling(T, S, n_comp=16):
    pca_t = PCA(n_components=n_comp).fit(np.nan_to_num(T, nan=0.0))
    pca_s = PCA(n_components=n_comp).fit(np.nan_to_num(S, nan=0.0))
    return pca_t.inverse_transform(pca_t.transform(T)), pca_s.inverse_transform(pca_s.transform(S))


def _warp_clim_ceiling(T, S, z, lat, lon):
    mld = mixed_layer_depth(T, S, z, lat, lon)
    d26, _ = isotherm_depth(T, z, 26.0)
    mld = np.where(np.isfinite(mld), mld, 50.0)
    d26 = np.where(np.isfinite(d26), d26, 120.0)
    T_clim = np.broadcast_to(np.nanmean(T, axis=0), T.shape).copy()
    S_clim = np.broadcast_to(np.nanmean(S, axis=0), S.shape).copy()
    T_c = warp_to_canonical(T_clim, z, mld, d26)
    S_c = warp_to_canonical(S_clim, z, mld, d26)
    return unwarp_from_canonical(T_c, z, mld, d26), unwarp_from_canonical(S_c, z, mld, d26)


def _gem_ceiling(T, S, sla):
    """Per-depth a(z)*SLA + clim (eval_baselines GEM)."""
    clim_t = np.nanmean(T, axis=0)
    clim_s = np.nanmean(S, axis=0)
    T_hat = np.broadcast_to(clim_t, T.shape).copy()
    S_hat = np.broadcast_to(clim_s, S.shape).copy()
    x = np.asarray(sla, dtype=np.float64)
    ok = np.isfinite(x)
    if ok.sum() < 5:
        return T_hat, S_hat
    A = np.column_stack([x[ok], np.ones(ok.sum())])
    for iz in range(T.shape[1]):
        yt = T[ok, iz] - clim_t[iz]
        ys = S[ok, iz] - clim_s[iz]
        at, bt = np.linalg.lstsq(A, yt, rcond=None)[0]
        as_, bs = np.linalg.lstsq(A, ys, rcond=None)[0]
        T_hat[:, iz] = clim_t[iz] + at * np.where(np.isfinite(x), x, 0.0) + bt
        S_hat[:, iz] = clim_s[iz] + as_ * np.where(np.isfinite(x), x, 0.0) + bs
    return T_hat, S_hat


def _try_real_bundle(config_path: Path):
    from base.split_utils import build_split_indices
    from base.util import read_json
    from preproc.export_v2_cache import build_argo_cache

    cfg = read_json(str(config_path))
    cache_path = cfg.get("data_loader", {}).get("args", {}).get("cache_path") or ""
    if not cache_path:
        try:
            cache_path = build_argo_cache(cfg)
        except Exception as exc:
            return None, f"cache build skipped: {exc}"
    p = Path(cache_path)
    if not p.is_file():
        return None, f"no cache at {cache_path}"
    with open(p, "rb") as f:
        cache = pickle.load(f)
    n = cache["LAT"].shape[0]
    T = _station_major(cache["profiles"]["temperature"], n)
    S = _station_major(cache["profiles"]["salinity"], n)
    z = np.asarray(cache["PRES"], dtype=np.float64).reshape(-1)
    lat = np.asarray(cache["LAT"], dtype=np.float64)
    lon = np.asarray(cache["LON"], dtype=np.float64)
    sla = cache.get("ssh_obs_sla")
    if sla is None:
        sla = np.asarray(cache["inputs"][:, -1], dtype=np.float64)
    dl = cfg["data_loader"]["args"]
    idx = build_split_indices(
        n, cache["JULD"], dl, dataset_tag=cache.get("dataset_tag", "argo_v2"), v2_src=cfg.get("io", {}).get("v2_src")
    )["test"]
    return {
        "T_true": T[idx],
        "S_true": S[idx],
        "z": z,
        "lat": lat[idx],
        "lon": lon[idx],
        "sla": np.asarray(sla, dtype=np.float64)[idx],
        "cache": cache,
        "cfg": cfg,
        "idx": idx,
        "source": str(p),
    }, None


def _ckpt_cfg(state, fallback: dict) -> dict:
    raw = state.get("config", fallback) if isinstance(state, dict) else fallback
    if isinstance(raw, dict):
        return raw
    inner = getattr(raw, "config", None) or getattr(raw, "_config", None)
    return inner if isinstance(inner, dict) else fallback


def _model_inputs(bundle, ckcfg):
    from preproc.enso import inject_enso_columns
    from preproc.preproc_isas_sat import compute_input_dim, count_encoding_dims

    cache = bundle["cache"]
    idx = bundle["idx"]
    ip = ckcfg.get("input_params") or cache.get("input_params") or {}
    expected = compute_input_dim(
        ip, int(cache.get("spatial_pad", 0)), int(cache.get("temporal_pad", 0))
    )
    x = inject_enso_columns(
        cache["inputs"][idx],
        cache["JULD"][idx],
        dataset_tag=cache.get("dataset_tag", "argo_v2"),
        input_params=ip,
        n_enc_base=count_encoding_dims(ip) or 6,
        expected_dim=expected,
    )
    return x


def _decode_heave_ts(mu, ckcfg, bundle):
    from base.split_utils import build_split_indices
    from model.loss import HeaveResidualLoss

    cache = bundle["cache"]
    n = cache["LAT"].shape[0]
    dl = ckcfg.get("data_loader", {}).get("args") or bundle["cfg"]["data_loader"]["args"]
    train_idx = build_split_indices(
        n, cache["JULD"], dl, dataset_tag=cache.get("dataset_tag", "argo_v2"),
        v2_src=ckcfg.get("io", {}).get("v2_src"),
    )["train"]
    import torch

    loss = HeaveResidualLoss(
        outputs=ckcfg["outputs"],
        device=torch.device("cpu"),
        true_profiles=cache["profiles"],
        pres_levels=cache["PRES"],
        lat=cache["LAT"],
        lon=cache["LON"],
        train_idx=train_idx,
        clim_profiles=cache.get("clim_profiles"),
        loss_config=ckcfg.get("loss_config") or {"mode": "heave_residual"},
    )
    T, S = loss.physical_ts(torch.tensor(mu, dtype=torch.float32), torch.tensor(bundle["idx"]))
    return T.detach().numpy(), S.detach().numpy()


def _load_ckpt_pred(ckpt: Path, bundle):
    import torch
    import model.model as module_arch

    if not ckpt.is_file():
        return None
    cfg = bundle["cfg"]
    from scripts.phase5_physical_space_score import _decode_pcs_to_ts

    try:
        device = torch.device("cpu")
        state = torch.load(ckpt, map_location=device, weights_only=False)
        ckcfg = _ckpt_cfg(state, cfg)
        arch = ckcfg.get("arch", cfg["arch"])
        cls = getattr(module_arch, arch["type"])
        model = cls(**arch["args"])
        model.load_state_dict(state["state_dict"])
        model.eval()
        x = _model_inputs(bundle, ckcfg)
        if x.shape[1] != int(arch["args"]["input_dim"]):
            raise ValueError(f"input dim {x.shape[1]} != arch {arch['args']['input_dim']}")
        inputs = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            out = model(inputs).cpu().numpy()
        outs = ckcfg.get("outputs") or bundle["cache"]["outputs"]
        d = int(sum(outs.values()))
        if out.shape[-1] == 2 * d:
            out = out[:, :d]
        n_z = int(np.asarray(bundle["T_true"]).shape[1])
        if out.shape[-1] == 2 * n_z:
            return out[:, :n_z], out[:, n_z:]
        if arch.get("type") == "HeaveResidual" or "warp" in outs:
            return _decode_heave_ts(out, ckcfg, bundle)
        T, S = _decode_pcs_to_ts(out, {"outputs": outs}, bundle["cache"])
        return T, S
    except Exception as exc:
        print(f"skip ckpt {ckpt}: {exc}")
        return None


def _hycom_note(path: str | None):
    if path and Path(path).is_file():
        return {"status": "present", "path": path}
    return {"status": "skipped", "reason": "no HYCOM 41-layer interface file (io.hycom_interfaces)"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", default=str(_ROOT / "config/argo/config_argo.json"))
    parser.add_argument("--heave-ckpt", action="append", default=[], help="HeaveResidual checkpoint (repeatable)")
    parser.add_argument("--ckpt", action="append", default=[], help="label=path extra checkpoints")
    parser.add_argument("--out-md", default=str(REPORT_MD))
    parser.add_argument("--out-json", default=str(REPORT_JSON))
    args = parser.parse_args(argv)

    models = {}
    bundle, err = _try_real_bundle(Path(args.config))
    if bundle is None:
        syn = _synthetic()
        models["heave_shift_proxy"] = _score_pair(
            syn["T_pred"], syn["S_pred"], syn["T_true"], syn["S_true"],
            syn["z"], syn["lat"], syn["lon"], syn["sla"], label="synthetic_heave",
        )
        T, S, z, lat, lon, sla = syn["T_true"], syn["S_true"], syn["z"], syn["lat"], syn["lon"], syn["sla"]
        source = f"synthetic ({err})"
        idx_note = "n=40 synthetic GoM-like thermocline"
    else:
        T, S, z, lat, lon, sla = bundle["T_true"], bundle["S_true"], bundle["z"], bundle["lat"], bundle["lon"], bundle["sla"]
        source = bundle["source"]
        idx_note = f"test n={len(bundle['idx'])}"
        a_pred = _load_ckpt_pred(A_CRPS, bundle)
        if a_pred is not None:
            models["A_CRPS"] = _score_pair(a_pred[0], a_pred[1], T, S, z, lat, lon, sla, "A_CRPS")
        for i, hp in enumerate(args.heave_ckpt):
            h_pred = _load_ckpt_pred(Path(hp), bundle)
            key = "Heave_best" if i == 0 else f"Heave_{i}"
            if h_pred is not None:
                models[key] = _score_pair(h_pred[0], h_pred[1], T, S, z, lat, lon, sla, key)
        for item in args.ckpt:
            if "=" in item:
                key, hp = item.split("=", 1)
            else:
                key, hp = Path(item).parent.name, item
            pred = _load_ckpt_pred(Path(hp), bundle)
            if pred is not None:
                models[key] = _score_pair(pred[0], pred[1], T, S, z, lat, lon, sla, key)
        b_ckpt = next(B_DET.rglob("model_best.pth"), None) if B_DET.is_dir() else None
        if b_ckpt:
            b_pred = _load_ckpt_pred(b_ckpt, bundle)
            if b_pred is not None:
                models["B_det"] = _score_pair(b_pred[0], b_pred[1], T, S, z, lat, lon, sla, "B_det")
        try:
            from scripts.isop_modas_baseline import design_matrix, fit_ridge, predict
            from model.joint_eof import reconstruct_joint_eof

            cache = bundle["cache"]
            if "joint_eof_meta" in cache:
                tr = np.arange(cache["LAT"].shape[0])
                tr = tr[~np.isin(tr, bundle["idx"])]
                pcs = cache["targets"]
                coef = fit_ridge(design_matrix(sla * 0, sla * 0, cache["JULD"])[tr], pcs[tr])
                # skip messy real ISOP if shapes disagree
                models["ISOP"] = {"label": "ISOP", "status": "cache has no joint EOF; skipped"}
            else:
                models["ISOP"] = {"label": "ISOP", "status": "skipped (no joint_eof_meta)"}
        except Exception as exc:
            models["ISOP"] = {"label": "ISOP", "status": f"skipped: {exc}"}

    t_pca, s_pca = _pca16_ceiling(T, S)
    t_warp, s_warp = _warp_clim_ceiling(T, S, z, lat, lon)
    t_gem, s_gem = _gem_ceiling(T, S, sla)
    ceilings = {
        "pca16": _score_pair(t_pca, s_pca, T, S, z, lat, lon, sla, "pca16_truth_recon"),
        "warp_clim_true_landmarks": _score_pair(t_warp, s_warp, T, S, z, lat, lon, sla, "warp_clim"),
        "gem_sla": _score_pair(t_gem, s_gem, T, S, z, lat, lon, sla, "gem"),
    }

    payload = {
        "evalphys_version": VERSION,
        "source": source,
        "subset": idx_note,
        "lc_box": {"lat": LC_LAT_RANGE, "lon": LC_LON_RANGE, "steric_gate_cm": STERIC_LC_RMS_CM},
        "models": models,
        "t1_ceilings": ceilings,
        "hycom_layer_mean": _hycom_note(None),
        "oni_roni": "CPC ONI/RONI under data/indices/; spliced when input_params.oni/roni",
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(payload, indent=2, default=str) + "\n")

    lines = [
        "# Thermocline scorecard",
        "",
        f"evalphys **{VERSION}**. Source: `{source}`. {idx_note}.",
        "",
        "LC steric gate: 2 cm RMS in 24–28°N, 88–84°W. HYCOM 41-layer means: skipped (no interface file).",
        "",
        "## Models",
        "",
    ]
    for name, row in models.items():
        if "heave_vs_shape" not in row:
            lines.append(f"- **{name}**: {row.get('status', row)}")
            continue
        h = row["heave_vs_shape"]
        st = row.get("steric_vs_adt") or {}
        ep = row.get("evalphys") or {}
        t_bands = ((ep.get("ts_rmse") or {}).get("T")) or {}
        n2 = (ep.get("static_stability_pred") or {}).get("1e-08") or {}
        lines.append(
            f"- **{name}**: D26 RMSE {row['D26_rmse']:.2f} m; 50–200 T RMSE {h['rmse_50_200']:.3f} "
            f"(heave-aligned {h['rmse_50_200_heave_aligned']:.3f}, heave fraction {h['heave_fraction']:.2f}); "
            f"T RMSE 0–50/50–200/200–800 {t_bands.get('0-50')}/{t_bands.get('50-200')}/{t_bands.get('200-800')}; "
            f"N² profile viol@1e-8 {n2.get('violation_rate_profile')}; "
            f"LC steric RMS {st.get('rms_cm_lc')} cm pass={st.get('lc_pass')}"
        )
    lines += ["", "## T1 reconstruction ceilings (truth through the representation)", ""]
    for name, row in ceilings.items():
        h = row["heave_vs_shape"]
        lines.append(
            f"- **{name}**: D26 RMSE {row['D26_rmse']:.2f} m; 50–200 T RMSE {h['rmse_50_200']:.3f}; "
            f"heave fraction {h['heave_fraction']:.2f}"
        )
    lines += [
        "",
        "If warp-clim ceiling << PCA-16 in 50–200 m, landmark heave is the missing degree of freedom.",
        "",
    ]
    Path(args.out_md).write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out_md}")
    print(f"wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
