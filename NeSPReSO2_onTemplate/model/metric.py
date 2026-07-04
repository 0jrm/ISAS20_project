"""Metrics: PCA inverse transform and per-variable profile RMSE."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch

from model.loss import output_slices, reconstruct_physical_profiles, sklearn_inverse_transform_pcs, torch_reconstruct_profile


def inverse_transform_profiles(pcs, pca_models, outputs):
    """Sklearn inverse per output; returns dict name -> (n_depth, n_batch)."""
    if torch.is_tensor(pcs):
        pcs = pcs.detach().cpu().numpy()
    return sklearn_inverse_transform_pcs(pcs, pca_models, outputs)


def profile_rmse(output, target, indices, data_loader):
    """Mean RMSE across outputs in physical profile space."""
    pca_models = data_loader.pca_models
    outputs = data_loader.outputs
    out_np = output.detach().cpu().numpy()
    tgt_np = target.detach().cpu().numpy()

    pred = inverse_transform_profiles(out_np, pca_models, outputs)
    true = inverse_transform_profiles(tgt_np, pca_models, outputs)

    clim = getattr(data_loader, "cache", {}).get("clim_profiles") if hasattr(data_loader, "cache") else None
    if clim is not None:
        idx = indices.detach().cpu().numpy() if torch.is_tensor(indices) else np.asarray(indices)
        pred = reconstruct_physical_profiles(out_np, pca_models, outputs, clim_profiles=clim, indices=idx)
        true = reconstruct_physical_profiles(tgt_np, pca_models, outputs, clim_profiles=clim, indices=idx)

    rmses = []
    for name in outputs:
        diff = pred[name] - true[name]
        rmses.append(float(np.sqrt(np.mean(diff ** 2))))
    return sum(rmses) / len(rmses)


def per_variable_rmse(output, target, pca_models, outputs):
    """Return ordered dict of RMSE per output variable."""
    out_np = output.detach().cpu().numpy() if torch.is_tensor(output) else output
    tgt_np = target.detach().cpu().numpy() if torch.is_tensor(target) else target
    pred = inverse_transform_profiles(out_np, pca_models, outputs)
    true = inverse_transform_profiles(tgt_np, pca_models, outputs)
    return {name: float(np.sqrt(np.mean((pred[name] - true[name]) ** 2))) for name in outputs}
