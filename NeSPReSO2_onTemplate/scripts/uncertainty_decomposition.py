#!/usr/bin/env python3
"""Phase 4.6 — MC input-noise uncertainty decomposition (aleatoric vs input-driven).

total_var = mean_m(σ_m²) + var_m(μ_m)
Requires cache with physical-scale error fields for meaningful α>0; otherwise
reports model σ only (input-driven term ≈ 0).
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-r", "--checkpoint", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--m", type=int, default=50)
    ap.add_argument("--max-samples", type=int, default=64)
    ap.add_argument("--out", type=Path, default=_ROOT.parent / "reports" / "uncertainty_decomposition.json")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    with open(args.cache, "rb") as f:
        cache = pickle.load(f)

    from model.model import PatchConvMLP
    from model.prob_head import split_mu_sigma

    state = ckpt.get("state_dict", ckpt)
    raw_cfg = ckpt.get("config")
    if raw_cfg is None:
        cfg = {}
    elif hasattr(raw_cfg, "config"):
        cfg = raw_cfg.config  # ConfigParser
    elif isinstance(raw_cfg, dict):
        cfg = raw_cfg
    else:
        cfg = {}
    arch = (cfg.get("arch") or {}).get("args") or {
        "input_dim": int(cache["inputs"].shape[1]),
        "output_dim": int(cache["targets"].shape[1]),
        "probabilistic": True,
        "n_enc": 6,
        "n_sat": 3,
        "d_model": 64,
        "head_layers": [256, 256],
    }

    model = PatchConvMLP(**{k: arch[k] for k in arch if k in (
        "input_dim", "output_dim", "dropout_prob", "d_model", "head_layers",
        "conv_channels", "patch_shape", "n_enc", "n_sat", "residual",
        "probabilistic", "sigma_min", "n_quantiles",
    )})
    model.load_state_dict(state, strict=False)
    model.eval()

    x = np.asarray(cache["inputs"][: args.max_samples], dtype=np.float32)
    d = int(cache["targets"].shape[1])
    err = cache.get("inputs_err_physical")  # optional
    rng = np.random.default_rng(42)

    mus, sigs = [], []
    with torch.no_grad():
        for _ in range(args.m):
            xb = x.copy()
            if err is not None and args.alpha > 0:
                e = np.asarray(err[: args.max_samples], dtype=np.float32)
                # align columns if err has fewer channels — apply to trailing sat cols
                n_sat = min(e.shape[1], xb.shape[1])
                noise = rng.normal(size=(xb.shape[0], n_sat)).astype(np.float32)
                xb[:, -n_sat:] = xb[:, -n_sat:] + float(args.alpha) * noise * e[:, :n_sat]
            out = model(torch.from_numpy(xb))
            if out.shape[-1] == 2 * d:
                mu_raw, sig = split_mu_sigma(out, d)
            else:
                mu_raw, sig = out, torch.ones_like(out)
            mus.append(mu_raw.numpy())
            sigs.append(sig.numpy())
    mus = np.stack(mus, axis=0)
    sigs = np.stack(sigs, axis=0)
    mean_sig2 = np.mean(sigs ** 2, axis=0)
    var_mu = np.var(mus, axis=0)
    total = mean_sig2 + var_mu
    frac_input = float(np.mean(var_mu / np.maximum(total, 1e-12)))
    payload = {
        "m": args.m,
        "alpha": args.alpha,
        "n_profiles": int(x.shape[0]),
        "fraction_input_driven": frac_input,
        "mean_total_var": float(np.mean(total)),
        "mean_model_var": float(np.mean(mean_sig2)),
        "mean_input_var": float(np.mean(var_mu)),
        "note": "Formal product errors are relative indicators; α scales them (PLAN caveat).",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
