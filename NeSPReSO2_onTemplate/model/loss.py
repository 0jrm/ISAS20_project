"""Training objectives for PCA-space profile prediction (N configurable outputs)."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

from model.density import DensityConstraint

# ponytail: GoM temp/sal-specific scales; re-derive for new outputs or regions
DEFAULT_PROFILE_SCALES = {"temperature": 37.86, "salinity": 0.28}
DEFAULT_COMBINED_PCA_SCALE = 2.8294
DEFAULT_COMBINED_MSE_SCALE = 0.0255

OUTPUT_H5_VARS = {"temperature": "TEMP", "salinity": "PSAL"}


def output_slices(outputs: Mapping[str, int]) -> list[tuple[str, int, int]]:
    """Ordered (name, start, end) for concatenated PCA targets."""
    slices = []
    start = 0
    for name, k in outputs.items():
        slices.append((name, start, start + k))
        start += k
    return slices


def torch_reconstruct_profile(pcs, components, mean):
    return pcs @ components + mean


def torch_reconstruct_profiles(pcs_dict, components_dict, means_dict, output_order):
    return {
        name: torch_reconstruct_profile(pcs_dict[name], components_dict[name], means_dict[name])
        for name in output_order
    }


def sklearn_inverse_transform_pcs(pcs, pca_models, outputs):
    """Inverse PCA per output; returns profiles (n_depth, n_samples) per variable."""
    profiles = {}
    for name, start, end in output_slices(outputs):
        profiles[name] = pca_models[name].inverse_transform(pcs[:, start:end]).T
    return profiles


def get_pca_weights(pca_models, pcs_by_name, output_order):
    """Concatenated PC weights (v2 ``get_pca_weights`` generalized)."""
    parts = []
    for name in output_order:
        pca = pca_models[name]
        pcs = pcs_by_name[name]
        parts.append(pca.explained_variance_ratio_ / pcs.var(axis=1))
    return np.concatenate(parts)


class WeightedMSELoss(nn.Module):
    def __init__(self, weights, device):
        super().__init__()
        self.weights = torch.tensor(weights, dtype=torch.float32, device=device)

    def forward(self, input, target):
        squared_diff = (input - target) ** 2
        return (self.weights * squared_diff).mean()


def genWeightedMSELoss(weights, device):
    normalized = weights / np.sum(weights)
    return WeightedMSELoss(normalized, device)


class PCALoss(nn.Module):
    def __init__(self, pca_models, outputs, profile_scales=None, device=None):
        super().__init__()
        self.outputs = OrderedDict(outputs)
        self.output_order = list(outputs.keys())
        self.slices = output_slices(outputs)
        self.profile_scales = dict(DEFAULT_PROFILE_SCALES)
        if profile_scales:
            self.profile_scales.update(profile_scales)

        for name in self.output_order:
            pca = pca_models[name]
            dev = device or torch.device("cpu")
            self.register_buffer(
                f"{name}_components",
                torch.tensor(pca.components_, dtype=torch.float32, device=dev),
            )
            self.register_buffer(
                f"{name}_mean",
                torch.tensor(pca.mean_, dtype=torch.float32, device=dev).unsqueeze(0),
            )

    def _components(self, name):
        return getattr(self, f"{name}_components")

    def _mean(self, name):
        return getattr(self, f"{name}_mean")

    def inverse_transform(self, pcs, components, mean):
        return torch_reconstruct_profile(pcs, components, mean)

    def forward(self, pcs, targets):
        total = pcs.new_tensor(0.0)
        for name, start, end in self.slices:
            pred = pcs[:, start:end]
            true = targets[:, start:end]
            pred_profiles = self.inverse_transform(pred, self._components(name), self._mean(name))
            true_profiles = self.inverse_transform(true, self._components(name), self._mean(name))
            mse = nn.functional.mse_loss(pred_profiles, true_profiles)
            scale = self.profile_scales.get(name, 1.0)
            total = total + mse / scale
        return total


class CombinedPCALoss(nn.Module):
    def __init__(
        self,
        pca_models,
        outputs,
        weights,
        device,
        profile_scales=None,
        combined_pca_scale=None,
        combined_mse_scale=None,
        density_config=None,
        density_meta=None,
    ):
        super().__init__()
        self.outputs = OrderedDict(outputs)
        self.output_order = list(outputs.keys())
        self.slices = output_slices(outputs)
        self.combined_pca_scale = combined_pca_scale or DEFAULT_COMBINED_PCA_SCALE
        self.combined_mse_scale = combined_mse_scale or DEFAULT_COMBINED_MSE_SCALE

        self.pca_loss = PCALoss(pca_models, outputs, profile_scales, device)
        self.weighted_mse_loss = genWeightedMSELoss(weights, device)

        for name in self.output_order:
            pca = pca_models[name]
            self.register_buffer(
                f"{name}_components",
                torch.tensor(pca.components_, dtype=torch.float32, device=device),
            )
            self.register_buffer(
                f"{name}_mean",
                torch.tensor(pca.mean_, dtype=torch.float32, device=device).unsqueeze(0),
            )

        self.density_helper = None
        if density_config and density_config.get("enabled", False):
            self.density_helper = DensityConstraint(dataset=density_meta, device=device, config=density_config)

    def _reconstruct_profiles(self, pcs):
        temp_name = self.output_order[0]
        sal_name = self.output_order[1] if len(self.output_order) > 1 else None
        recon = {}
        for name, start, end in self.slices:
            recon[name] = torch_reconstruct_profile(
                pcs[:, start:end], getattr(self, f"{name}_components"), getattr(self, f"{name}_mean")
            )
        if temp_name in recon and sal_name in recon:
            return recon[temp_name], recon[sal_name]
        return recon

    def forward(self, pcs, targets, indices=None):
        pca_loss = self.pca_loss(pcs, targets)
        weighted_mse_loss = self.weighted_mse_loss(pcs, targets)
        combined_loss = (pca_loss / self.combined_pca_scale + weighted_mse_loss / self.combined_mse_scale) / 2

        if self.density_helper is not None and indices is not None and len(self.output_order) >= 2:
            temp_name, sal_name = self.output_order[0], self.output_order[1]
            recon = self._reconstruct_profiles(pcs)
            if isinstance(recon, tuple):
                temp_profiles, sal_profiles = recon
            else:
                temp_profiles, sal_profiles = recon[temp_name], recon[sal_name]
            combined_loss = combined_loss + self.density_helper(temp_profiles, sal_profiles, indices)

        return combined_loss


def make_loss(
    *,
    pca_models,
    outputs,
    weights,
    device,
    density_config: dict[str, Any] | None = None,
    density_meta=None,
    loss_scales: dict[str, Any] | None = None,
    **kwargs,
) -> CombinedPCALoss:
    scales = loss_scales or {}
    return CombinedPCALoss(
        pca_models=pca_models,
        outputs=outputs,
        weights=weights,
        device=device,
        density_config=density_config,
        density_meta=density_meta,
        profile_scales=scales.get("profile_scales"),
        combined_pca_scale=scales.get("combined_pca_scale"),
        combined_mse_scale=scales.get("combined_mse_scale"),
        **kwargs,
    )
