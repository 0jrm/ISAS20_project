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

VALID_LOSS_MODES = ("combined", "pred_profile_cached", "pc_mse_only")


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


def compute_true_profiles(
    targets,
    pca_models,
    outputs,
    device,
    *,
    cached: Mapping[str, np.ndarray] | None = None,
) -> dict[str, torch.Tensor]:
    """Precompute sample-major true profiles (N, n_depth) for cached loss mode."""
    if cached is not None:
        order = list(outputs.keys())
        out = {}
        for name in order:
            arr = np.asarray(cached[name], dtype=np.float32)
            if arr.ndim != 2:
                raise ValueError(f"true_profiles[{name}] must be 2-D, got {arr.shape}")
            out[name] = torch.tensor(np.ascontiguousarray(arr), dtype=torch.float32, device=device)
        return out
    if targets is None:
        raise ValueError("targets required when cached true_profiles are absent")
    targets_np = np.asarray(targets, dtype=np.float64)
    prof = sklearn_inverse_transform_pcs(targets_np, pca_models, outputs)
    return {
        name: torch.tensor(prof[name].T.astype(np.float32), dtype=torch.float32, device=device)
        for name in outputs
    }


def true_profiles_numpy(targets, pca_models, outputs) -> dict[str, np.ndarray]:
    """Sample-major true profiles for cache pickle (N, n_depth)."""
    targets_np = np.asarray(targets, dtype=np.float64)
    prof = sklearn_inverse_transform_pcs(targets_np, pca_models, outputs)
    return {name: prof[name].T.astype(np.float32) for name in outputs}


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
    def __init__(
        self,
        pca_models,
        outputs,
        profile_scales=None,
        device=None,
        *,
        mode: str = "combined",
        true_profiles: Mapping[str, torch.Tensor] | None = None,
    ):
        super().__init__()
        if mode not in VALID_LOSS_MODES:
            raise ValueError(f"unknown PCALoss mode {mode!r}; expected one of {VALID_LOSS_MODES}")
        self.mode = mode
        self.outputs = OrderedDict(outputs)
        self.output_order = list(outputs.keys())
        self.slices = output_slices(outputs)
        self.profile_scales = dict(DEFAULT_PROFILE_SCALES)
        if profile_scales:
            self.profile_scales.update(profile_scales)

        dev = device or torch.device("cpu")
        for name in self.output_order:
            pca = pca_models[name]
            self.register_buffer(
                f"{name}_components",
                torch.tensor(pca.components_, dtype=torch.float32, device=dev),
            )
            self.register_buffer(
                f"{name}_mean",
                torch.tensor(pca.mean_, dtype=torch.float32, device=dev).unsqueeze(0),
            )
            if mode == "pred_profile_cached":
                if true_profiles is None or name not in true_profiles:
                    raise ValueError(f"pred_profile_cached requires true_profiles[{name!r}]")
                self.register_buffer(f"{name}_true_profiles", true_profiles[name])

    def _components(self, name):
        return getattr(self, f"{name}_components")

    def _mean(self, name):
        return getattr(self, f"{name}_mean")

    def _true_profiles(self, name):
        return getattr(self, f"{name}_true_profiles")

    def inverse_transform(self, pcs, components, mean):
        return torch_reconstruct_profile(pcs, components, mean)

    def forward(self, pcs, targets, indices=None):
        if self.mode == "pred_profile_cached":
            if indices is None:
                raise ValueError("pred_profile_cached PCALoss requires batch indices")
            total = pcs.new_tensor(0.0)
            for name, start, end in self.slices:
                pred = pcs[:, start:end]
                pred_profiles = self.inverse_transform(pred, self._components(name), self._mean(name))
                true_profiles = self._true_profiles(name)[indices]
                mse = nn.functional.mse_loss(pred_profiles, true_profiles)
                scale = self.profile_scales.get(name, 1.0)
                total = total + mse / scale
            return total

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
        *,
        mode: str = "combined",
        true_profiles: Mapping[str, torch.Tensor] | None = None,
    ):
        super().__init__()
        if mode not in VALID_LOSS_MODES:
            raise ValueError(f"unknown loss mode {mode!r}; expected one of {VALID_LOSS_MODES}")
        self.mode = mode
        self.outputs = OrderedDict(outputs)
        self.output_order = list(outputs.keys())
        self.slices = output_slices(outputs)
        self.combined_pca_scale = combined_pca_scale or DEFAULT_COMBINED_PCA_SCALE
        self.combined_mse_scale = combined_mse_scale or DEFAULT_COMBINED_MSE_SCALE

        pca_mode = mode if mode == "pred_profile_cached" else "combined"
        self.pca_loss = PCALoss(
            pca_models,
            outputs,
            profile_scales,
            device,
            mode=pca_mode,
            true_profiles=true_profiles,
        )
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
        weighted_mse_loss = self.weighted_mse_loss(pcs, targets)
        if self.mode == "pc_mse_only":
            combined_loss = weighted_mse_loss / self.combined_mse_scale
        else:
            pca_loss = self.pca_loss(pcs, targets, indices)
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
    loss_config: dict[str, Any] | None = None,
    targets=None,
    true_profiles=None,
    **kwargs,
) -> CombinedPCALoss:
    scales = loss_scales or {}
    cfg = loss_config or {}
    mode = cfg.get("mode", "combined")
    if mode not in VALID_LOSS_MODES:
        raise ValueError(f"unknown loss_config.mode {mode!r}; expected one of {VALID_LOSS_MODES}")

    cached_profiles = None
    if mode == "pred_profile_cached":
        cached_profiles = compute_true_profiles(
            targets,
            pca_models,
            outputs,
            device,
            cached=true_profiles,
        )

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
        mode=mode,
        true_profiles=cached_profiles,
        **kwargs,
    )
