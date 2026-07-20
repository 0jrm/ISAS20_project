"""Training objectives for PCA-space profile prediction (N configurable outputs)."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

import model.model as module_arch
from model.density import DensityConstraint
from model.steric import StericConstraint

# ponytail: GoM temp/sal-specific scales; re-derive for new outputs or regions
DEFAULT_PROFILE_SCALES = {"temperature": 37.86, "salinity": 0.28}
DEFAULT_COMBINED_PCA_SCALE = 2.8294
DEFAULT_COMBINED_MSE_SCALE = 0.0255

OUTPUT_H5_VARS = {"temperature": "TEMP", "salinity": "PSAL"}

VALID_LOSS_MODES = ("combined", "pred_profile_cached", "pc_mse_only", "decoder", "density_spice")
VALID_PROB_MODES = ("mse", "crps", "nll", "quantile")


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


def reconstruct_physical_profiles(
    pcs,
    pca_models,
    outputs,
    *,
    clim_profiles: Mapping[str, np.ndarray] | None = None,
    indices: np.ndarray | list[int] | None = None,
):
    """
    Inverse PCA to physical profiles; add climatology when ``anomaly_targets``.

    Returns dict var -> (n_depth, n_batch). Accepts torch or numpy ``pcs``.
    """
    if torch.is_tensor(pcs):
        pcs_np = pcs.detach().cpu().numpy()
    else:
        pcs_np = np.asarray(pcs, dtype=np.float64)
    prof = sklearn_inverse_transform_pcs(pcs_np, pca_models, outputs)
    if clim_profiles is None:
        return prof
    idx = np.arange(pcs_np.shape[0]) if indices is None else np.asarray(indices, dtype=int)
    for name in outputs:
        clim = np.asarray(clim_profiles[name], dtype=np.float32)
        if clim.ndim == 2 and clim.shape[1] > idx.max():
            prof[name] = prof[name] + clim[:, idx]
    return prof


def _add_clim_torch(
    recon: dict[str, torch.Tensor],
    clim_profiles: Mapping[str, np.ndarray],
    indices: torch.Tensor,
    output_order: list[str],
) -> dict[str, torch.Tensor]:
    out = {}
    idx = indices.detach().cpu().numpy()
    for name in output_order:
        clim = torch.as_tensor(clim_profiles[name][:, idx], dtype=recon[name].dtype, device=recon[name].device)
        if recon[name].shape == clim.T.shape:
            out[name] = recon[name] + clim.T
        else:
            out[name] = recon[name] + clim
    return out


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


def ae_weights_numpy(ae_targets: np.ndarray, outputs: Mapping[str, int]) -> np.ndarray:
    """Per-dim 1/var weights for AE latent targets (sample-major)."""
    parts = []
    start = 0
    for k in outputs.values():
        block = np.asarray(ae_targets[:, start : start + k], dtype=np.float64)
        var = block.var(axis=0)
        var = np.maximum(var, 1e-12)
        parts.append(1.0 / var)
        start += k
    return np.concatenate(parts)


def _build_profile_ae(
    arch: str,
    encoding_dim: int,
    input_dim: int,
    *,
    encoder_layers: list[int] | None = None,
    decoder_layers: list[int] | None = None,
    layer_scale: int = 1,
    residual: bool = False,
    variable: str = "temperature",
    surface_residual: bool | None = None,
) -> nn.Module:
    """Match ``scripts/train_profile_ae.py`` layer layout."""
    if arch in ("Autoencoder", "ResAutoencoder"):
        use_residual = residual or arch == "ResAutoencoder"
        if encoder_layers is None or decoder_layers is None:
            enc = [min(512, max(128, input_dim // 4)), 128, 64]
            dec = [64, 128, min(512, max(128, input_dim // 4))]
            if layer_scale != 1:
                enc = [h * layer_scale for h in enc]
                dec = [h * layer_scale for h in dec]
        else:
            enc, dec = encoder_layers, decoder_layers
        cls = module_arch.ResAutoencoder if use_residual else module_arch.Autoencoder
        kwargs = dict(
            encoding_dim=encoding_dim,
            encoder_layers=enc,
            decoder_layers=dec,
            input_dim=input_dim,
            residual=use_residual,
        )
        if use_residual:
            kwargs["variable"] = variable
            kwargs["surface_residual"] = True if surface_residual is None else surface_residual
        return cls(**kwargs)
    if arch == "KAN_Autoencoder":
        return module_arch.KAN_Autoencoder(encoding_dim, input_dim=input_dim)
    raise ValueError(f"unknown profile AE arch {arch!r}")


def resolve_decoder_dir(decoder_dir: str | Path) -> Path:
    path = Path(decoder_dir)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    return path


def load_profile_decoder(ckpt_path: str | Path, device: torch.device) -> nn.Module:
    """Load frozen profile AE (decode path used in training loss)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = _build_profile_ae(
        ckpt["arch"],
        ckpt["encoding_dim"],
        ckpt["input_dim"],
        encoder_layers=ckpt.get("encoder_layers"),
        decoder_layers=ckpt.get("decoder_layers"),
        layer_scale=int(ckpt.get("layer_scale", 1)),
        residual=bool(ckpt.get("residual", ckpt["arch"] == "ResAutoencoder")),
        variable=ckpt.get("variable", "temperature"),
        surface_residual=ckpt.get("surface_residual"),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model.to(device)


def load_decoders_from_dir(decoder_dir: str | Path, outputs: Mapping[str, int], device: torch.device) -> dict[str, nn.Module]:
    root = resolve_decoder_dir(decoder_dir)
    decoders = {}
    for name in outputs:
        ckpt = root / name / "decoder_best.pth"
        if not ckpt.is_file():
            raise FileNotFoundError(f"missing decoder checkpoint for {name!r}: {ckpt}")
        decoders[name] = load_profile_decoder(ckpt, device)
    return decoders


def decode_latent_profiles(
    pcs: torch.Tensor,
    decoders: Mapping[str, nn.Module],
    outputs: Mapping[str, int],
    *,
    inputs: torch.Tensor | None = None,
    surface_residual_layout: Mapping[str, Any] | None = None,
) -> dict[str, torch.Tensor]:
    """Batch-major profiles from concatenated latent slices."""
    from preproc.preproc_isas_sat import count_scalar_dims, surface_residual_from_features

    recon = {}
    start = 0
    for name, k in outputs.items():
        z = pcs[:, start : start + k]
        decoder = decoders[name]
        surface_residual = None
        if getattr(decoder, "surface_residual", False):
            if inputs is None or surface_residual_layout is None:
                raise ValueError("surface-residual decoder requires inputs and surface_residual_layout")
            layout = dict(surface_residual_layout)
            layout.setdefault("n_enc", count_scalar_dims(layout.get("input_params", {})))
            surface_residual = surface_residual_from_features(inputs, name, **layout)
            recon[name] = decoder.decode(z, surface_residual=surface_residual)
        else:
            recon[name] = decoder.decode(z)
        start += k
    return recon


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


def masked_profile_mse(pred_profiles, true_profiles, depth_mask):
    """MSE over profile levels where ``depth_mask`` is True."""
    mask = depth_mask.to(dtype=pred_profiles.dtype)
    diff2 = (pred_profiles - true_profiles) ** 2 * mask
    denom = mask.sum().clamp_min(1.0)
    return diff2.sum() / denom


def bathy_depth_mask(indices, bottom_depth, pres_levels, device):
    """``(batch, n_depth)`` mask: levels at or above bottom (``PRES <= bottom_depth``)."""
    if bottom_depth is None or pres_levels is None:
        return None
    idx = torch.as_tensor(indices, dtype=torch.long, device=device)
    bd = torch.as_tensor(bottom_depth, dtype=torch.float32, device=device)[idx]
    pres = torch.as_tensor(pres_levels, dtype=torch.float32, device=device).unsqueeze(0)
    return pres <= bd.unsqueeze(1)


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
        bottom_depth: np.ndarray | None = None,
        pres_levels: np.ndarray | None = None,
        joint_eof_meta: Mapping[str, Any] | None = None,
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
        self.joint_eof = joint_eof_meta is not None
        if self.joint_eof and list(outputs.keys()) != ["joint"]:
            raise ValueError("joint_eof_meta requires outputs {'joint': R}")

        dev = device or torch.device("cpu")
        self.use_bathy = bottom_depth is not None and pres_levels is not None
        if self.use_bathy:
            self.register_buffer(
                "bottom_depth",
                torch.tensor(np.asarray(bottom_depth, dtype=np.float32), device=dev),
            )
            self.register_buffer(
                "pres_levels",
                torch.tensor(np.asarray(pres_levels, dtype=np.float32), device=dev),
            )

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
            if mode == "pred_profile_cached" and not self.joint_eof:
                if true_profiles is None or name not in true_profiles:
                    raise ValueError(f"pred_profile_cached requires true_profiles[{name!r}]")
                self.register_buffer(f"{name}_true_profiles", true_profiles[name])

        if self.joint_eof:
            meta = joint_eof_meta
            self.n_lev = int(meta["n_lev"])
            for key in ("T_mean", "T_std", "S_mean", "S_std"):
                self.register_buffer(
                    f"joint_{key}",
                    torch.tensor(np.asarray(meta[key], dtype=np.float32), device=dev),
                )
            if true_profiles is None or "temperature" not in true_profiles or "salinity" not in true_profiles:
                raise ValueError("joint_eof PCALoss requires true_profiles temperature+salinity")
            for name in ("temperature", "salinity"):
                arr = true_profiles[name]
                if not torch.is_tensor(arr):
                    arr = torch.tensor(np.asarray(arr, dtype=np.float32), device=dev)
                else:
                    arr = arr.to(device=dev)
                self.register_buffer(f"{name}_true_profiles", arr)

    def _components(self, name):
        return getattr(self, f"{name}_components")

    def _mean(self, name):
        return getattr(self, f"{name}_mean")

    def _true_profiles(self, name):
        return getattr(self, f"{name}_true_profiles")

    def inverse_transform(self, pcs, components, mean):
        return torch_reconstruct_profile(pcs, components, mean)

    def _decode_joint_ts(self, pcs_joint: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        from model.joint_eof import torch_reconstruct_joint_eof

        return torch_reconstruct_joint_eof(
            pcs_joint,
            self._components("joint"),
            self._mean("joint"),
            self.joint_T_mean,
            self.joint_T_std,
            self.joint_S_mean,
            self.joint_S_std,
            self.n_lev,
        )

    def forward(self, pcs, targets, indices=None):
        depth_mask = None
        if self.use_bathy and indices is not None:
            depth_mask = bathy_depth_mask(indices, self.bottom_depth, self.pres_levels, pcs.device)

        if self.joint_eof:
            if indices is None:
                raise ValueError("joint_eof PCALoss requires batch indices")
            pred_t, pred_s = self._decode_joint_ts(pcs)
            true_t = self.temperature_true_profiles[indices]
            true_s = self.salinity_true_profiles[indices]
            total = pcs.new_tensor(0.0)
            for name, pred, true in (
                ("temperature", pred_t, true_t),
                ("salinity", pred_s, true_s),
            ):
                if depth_mask is not None:
                    mse = masked_profile_mse(pred, true, depth_mask)
                else:
                    mse = nn.functional.mse_loss(pred, true)
                total = total + mse / self.profile_scales.get(name, 1.0)
            return total

        if self.mode == "pred_profile_cached":
            if indices is None:
                raise ValueError("pred_profile_cached PCALoss requires batch indices")
            total = pcs.new_tensor(0.0)
            for name, start, end in self.slices:
                pred = pcs[:, start:end]
                pred_profiles = self.inverse_transform(pred, self._components(name), self._mean(name))
                true_profiles = self._true_profiles(name)[indices]
                if depth_mask is not None:
                    mse = masked_profile_mse(pred_profiles, true_profiles, depth_mask)
                else:
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
            if depth_mask is not None:
                mse = masked_profile_mse(pred_profiles, true_profiles, depth_mask)
            else:
                mse = nn.functional.mse_loss(pred_profiles, true_profiles)
            scale = self.profile_scales.get(name, 1.0)
            total = total + mse / scale
        return total


class DecoderProfileLoss(nn.Module):
    """Profile MSE via frozen learned decoders (Phase 5 Stage B)."""

    def __init__(
        self,
        decoders: Mapping[str, nn.Module],
        outputs,
        profile_scales=None,
        device=None,
        *,
        true_profiles: Mapping[str, torch.Tensor] | None = None,
        surface_residual_layout: Mapping[str, Any] | None = None,
    ):
        super().__init__()
        self.outputs = OrderedDict(outputs)
        self.output_order = list(outputs.keys())
        self.slices = output_slices(outputs)
        self.profile_scales = dict(DEFAULT_PROFILE_SCALES)
        if profile_scales:
            self.profile_scales.update(profile_scales)
        self.decoders = nn.ModuleDict({name: decoders[name] for name in self.output_order})
        self.surface_residual_layout = surface_residual_layout
        self.needs_inputs = any(getattr(d, "surface_residual", False) for d in self.decoders.values())
        if true_profiles is None:
            raise ValueError("DecoderProfileLoss requires true_profiles")
        for name in self.output_order:
            if name not in true_profiles:
                raise ValueError(f"DecoderProfileLoss missing true_profiles[{name!r}]")
            self.register_buffer(f"{name}_true_profiles", true_profiles[name])

    def _true_profiles(self, name):
        return getattr(self, f"{name}_true_profiles")

    def forward(self, pcs, targets, indices=None, inputs=None):
        if indices is None:
            raise ValueError("DecoderProfileLoss requires batch indices")
        if self.needs_inputs and inputs is None:
            raise ValueError("DecoderProfileLoss requires inputs for surface-residual decoders")
        total = pcs.new_tensor(0.0)
        for name, start, end in self.slices:
            z = pcs[:, start:end]
            surface_residual = None
            if getattr(self.decoders[name], "surface_residual", False):
                from preproc.preproc_isas_sat import count_scalar_dims, surface_residual_from_features

                layout = dict(self.surface_residual_layout or {})
                layout.setdefault("n_enc", count_scalar_dims(layout.get("input_params", {})))
                surface_residual = surface_residual_from_features(inputs, name, **layout)
                pred_profiles = self.decoders[name].decode(z, surface_residual=surface_residual)
            else:
                pred_profiles = self.decoders[name].decode(z)
            true_profiles = self._true_profiles(name)[indices]
            diff = pred_profiles - true_profiles
            mse = torch.nanmean(diff ** 2)
            scale = self.profile_scales.get(name, 1.0)
            total = total + mse / scale
        return total


class PCAHeteroLoss(nn.Module):
    """PC-space heteroscedastic CRPS/NLL (matrix A/B). Targets are PCA coeffs."""

    def __init__(
        self,
        *,
        prob_mode: str = "crps",
        nll_beta: float = 0.5,
        sigma_min: float = 1e-3,
        freeze_sigma: bool = False,
    ):
        super().__init__()
        if prob_mode not in VALID_PROB_MODES:
            raise ValueError(f"prob_mode must be one of {VALID_PROB_MODES}, got {prob_mode!r}")
        self.prob_mode = str(prob_mode)
        self.nll_beta = float(nll_beta)
        self.sigma_min = float(sigma_min)
        self.freeze_sigma = bool(freeze_sigma)

    def forward(self, output, target, indices=None, inputs=None):
        from evalphys.calibration import gaussian_crps_torch
        from model.prob_head import beta_nll, softplus_sigma, split_mu_sigma

        d = int(target.shape[-1])
        mu, raw = split_mu_sigma(output, d)
        if self.prob_mode == "mse" or self.freeze_sigma:
            return torch.mean((mu - target) ** 2)
        sigma = softplus_sigma(raw, sigma_min=self.sigma_min)
        if self.prob_mode == "crps":
            return torch.mean(gaussian_crps_torch(mu, sigma, target, sigma_min=self.sigma_min))
        if self.prob_mode == "nll":
            return beta_nll(mu, sigma, target, beta=self.nll_beta, sigma_min=self.sigma_min)
        raise ValueError(f"PCAHeteroLoss unsupported prob_mode={self.prob_mode!r}")


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
        steric_config=None,
        steric_meta=None,
        clim_profiles: Mapping[str, np.ndarray] | None = None,
        *,
        mode: str = "combined",
        true_profiles: Mapping[str, torch.Tensor] | None = None,
        decoders: Mapping[str, nn.Module] | None = None,
        surface_residual_layout: Mapping[str, Any] | None = None,
        bottom_depth: np.ndarray | None = None,
        pres_levels: np.ndarray | None = None,
        joint_eof_meta: Mapping[str, Any] | None = None,
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
        self.joint_eof_meta = joint_eof_meta

        if mode == "decoder":
            if decoders is None:
                raise ValueError("decoder mode requires decoders")
            cached_profiles = true_profiles
            if cached_profiles is None:
                raise ValueError("decoder mode requires true_profiles")
            self.pca_loss = None
            self.decoder_loss = DecoderProfileLoss(
                decoders,
                outputs,
                profile_scales,
                device,
                true_profiles=cached_profiles,
                surface_residual_layout=surface_residual_layout,
            )
            pca_mode = "combined"
        else:
            self.decoder_loss = None
            pca_mode = mode if mode == "pred_profile_cached" else "combined"
            self.pca_loss = PCALoss(
                pca_models,
                outputs,
                profile_scales,
                device,
                mode=pca_mode,
                true_profiles=true_profiles,
                bottom_depth=bottom_depth,
                pres_levels=pres_levels,
                joint_eof_meta=joint_eof_meta,
            )

        self.weighted_mse_loss = genWeightedMSELoss(weights, device)

        if mode != "decoder":
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

        self.needs_inputs = bool(self.decoder_loss and self.decoder_loss.needs_inputs)
        self.clim_profiles = clim_profiles
        self._use_clim = clim_profiles is not None
        if self._use_clim:
            for name in self.output_order:
                arr = np.asarray(clim_profiles[name], dtype=np.float32)
                self.register_buffer(f"{name}_clim_profiles", torch.tensor(arr, dtype=torch.float32, device=device))

        self.density_helper = None
        if density_config and density_config.get("enabled", False):
            self.density_helper = DensityConstraint(dataset=density_meta, device=device, config=density_config)
        self.steric_helper = None
        if steric_config and steric_config.get("enabled", False):
            self.steric_helper = StericConstraint(dataset=steric_meta, device=device, config=steric_config)

    def _physical_profiles(self, pcs, inputs=None, indices=None):
        recon = self._reconstruct_profiles(pcs, inputs=inputs)
        if not self._use_clim or indices is None:
            return recon
        if isinstance(recon, tuple):
            temp_name, sal_name = self.output_order[0], self.output_order[1]
            temp_p, sal_p = recon
            idx = torch.as_tensor(indices, dtype=torch.long, device=pcs.device)
            clim_t = getattr(self, f"{temp_name}_clim_profiles")[:, idx].T
            clim_s = getattr(self, f"{sal_name}_clim_profiles")[:, idx].T
            return temp_p + clim_t, sal_p + clim_s
        idx = torch.as_tensor(indices, dtype=torch.long, device=pcs.device)
        out = {}
        for name in self.output_order:
            out[name] = recon[name] + getattr(self, f"{name}_clim_profiles")[:, idx].T
        return out

    def _reconstruct_profiles(self, pcs, inputs=None):
        if self.mode == "decoder":
            recon = decode_latent_profiles(
                pcs,
                self.decoder_loss.decoders,
                self.outputs,
                inputs=inputs,
                surface_residual_layout=self.decoder_loss.surface_residual_layout,
            )
            temp_name = self.output_order[0]
            sal_name = self.output_order[1] if len(self.output_order) > 1 else None
            if sal_name is not None:
                return recon[temp_name], recon[sal_name]
            return recon

        if self.joint_eof_meta is not None and self.pca_loss is not None:
            return self.pca_loss._decode_joint_ts(pcs)

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

    def forward(self, pcs, targets, indices=None, inputs=None):
        weighted_mse_loss = self.weighted_mse_loss(pcs, targets)
        if self.mode == "pc_mse_only":
            combined_loss = weighted_mse_loss / self.combined_mse_scale
        elif self.mode == "decoder":
            profile_loss = self.decoder_loss(pcs, targets, indices, inputs=inputs)
            combined_loss = (profile_loss / self.combined_pca_scale + weighted_mse_loss / self.combined_mse_scale) / 2
        else:
            pca_loss = self.pca_loss(pcs, targets, indices)
            combined_loss = (pca_loss / self.combined_pca_scale + weighted_mse_loss / self.combined_mse_scale) / 2

        needs_physical = self.density_helper is not None or self.steric_helper is not None
        if needs_physical and indices is not None and len(self.output_order) >= 2:
            recon_phys = self._physical_profiles(pcs, inputs=inputs, indices=indices)
            if isinstance(recon_phys, tuple):
                temp_profiles, sal_profiles = recon_phys
            else:
                temp_name, sal_name = self.output_order[0], self.output_order[1]
                temp_profiles, sal_profiles = recon_phys[temp_name], recon_phys[sal_name]
            if self.density_helper is not None:
                combined_loss = combined_loss + self.density_helper(temp_profiles, sal_profiles, indices)
            if self.steric_helper is not None:
                combined_loss = combined_loss + self.steric_helper(temp_profiles, sal_profiles, indices)

        return combined_loss


class DensitySpiceLoss(nn.Module):
    """Phase 3 deterministic loss: MSE(σ̂₀_ctrl, σ₀_ctrl) + MSE(ẑ_τ, z_τ); λ_f=0 for v1.

    Full-rank (R=K): residual δa → softplus+cumsum.
    Low-rank (R<K): ``σ̂₀ = σ₀_clim + scores @ delta_sigma0_basis`` (σ₀-space PCA).
    Targets remain K-wide standardized σ₀_ctrl + spice PCs.
    """

    def __init__(
        self,
        outputs: Mapping[str, int],
        dz_tilde: np.ndarray | torch.Tensor,
        device,
        *,
        lambda_rho: float = 1.0,
        lambda_tau: float = 1.0,
        sigma0_mean: np.ndarray | None = None,
        sigma0_std: np.ndarray | None = None,
        a_clim: np.ndarray | torch.Tensor | None = None,
        delta_sigma0_basis: np.ndarray | torch.Tensor | None = None,
        sigma0_clim: np.ndarray | torch.Tensor | None = None,
        n_ctrl: int | None = None,
        # legacy alias ignored if delta_sigma0_basis set
        delta_a_basis: np.ndarray | torch.Tensor | None = None,
    ):
        super().__init__()
        self.outputs = OrderedDict(outputs)
        if "density_ctrl" not in self.outputs or "spice" not in self.outputs:
            raise ValueError("density_spice loss requires outputs density_ctrl + spice")
        self.n_scores = int(self.outputs["density_ctrl"])
        self.n_spice = int(self.outputs["spice"])
        self.k = int(n_ctrl) if n_ctrl is not None else self.n_scores
        if delta_a_basis is not None and delta_sigma0_basis is None:
            raise ValueError(
                "a-space delta_a_basis is retired (σ₀ recon ceiling); rebuild cache for delta_sigma0_basis"
            )
        if delta_sigma0_basis is not None:
            basis = np.asarray(delta_sigma0_basis)
            if basis.shape != (self.n_scores, self.k):
                raise ValueError(
                    f"delta_sigma0_basis shape {basis.shape} != (R={self.n_scores}, K={self.k})"
                )
        elif self.n_scores != self.k:
            raise ValueError(
                f"low-rank density_ctrl={self.n_scores} < K={self.k} requires delta_sigma0_basis"
            )
        self.lambda_rho = float(lambda_rho)
        self.lambda_tau = float(lambda_tau)
        dz = torch.as_tensor(dz_tilde, dtype=torch.float32, device=device)
        if dz.numel() != self.k:
            raise ValueError(f"dz_tilde length {dz.numel()} != K={self.k}")
        self.register_buffer("dz_tilde", dz)
        if a_clim is not None:
            ac = torch.as_tensor(a_clim, dtype=torch.float32, device=device).reshape(-1)
            if ac.numel() != self.k:
                raise ValueError(f"a_clim length {ac.numel()} != K={self.k}")
            self.register_buffer("a_clim", ac)
        else:
            self.a_clim = None
        if delta_sigma0_basis is not None:
            self.register_buffer(
                "delta_sigma0_basis",
                torch.as_tensor(delta_sigma0_basis, dtype=torch.float32, device=device),
            )
            clim = sigma0_clim if sigma0_clim is not None else sigma0_mean
            if clim is None:
                raise ValueError("low-rank density_spice requires sigma0_clim or sigma0_mean")
            self.register_buffer(
                "sigma0_clim", torch.as_tensor(clim, dtype=torch.float32, device=device).reshape(-1)
            )
        else:
            self.delta_sigma0_basis = None
            self.sigma0_clim = None
        if sigma0_mean is not None:
            self.register_buffer(
                "sigma0_mean", torch.as_tensor(sigma0_mean, dtype=torch.float32, device=device)
            )
            self.register_buffer(
                "sigma0_std",
                torch.as_tensor(np.maximum(sigma0_std, 1e-6), dtype=torch.float32, device=device),
            )
        else:
            self.sigma0_mean = None
            self.sigma0_std = None

    def _sigma0_from_raw(self, mu_raw: torch.Tensor) -> torch.Tensor:
        from model.density_spice import decode_a_from_output, decode_sigma0_ctrl, decode_sigma0_from_scores

        if self.delta_sigma0_basis is not None:
            sig, _ = decode_sigma0_from_scores(
                mu_raw,
                self.sigma0_clim,
                self.n_scores,
                self.n_spice,
                self.delta_sigma0_basis,
            )
            return sig
        if self.a_clim is None:
            raise ValueError("DensitySpiceLoss requires a_clim for full-rank residual-δa")
        a, _ = decode_a_from_output(
            mu_raw, self.a_clim, self.n_scores, self.n_spice, basis=None
        )
        return decode_sigma0_ctrl(a, self.dz_tilde)

    def forward(self, output, target, indices=None, inputs=None):
        z_tau_hat = output[:, self.n_scores : self.n_scores + self.n_spice]
        sig_tgt = target[:, : self.k]
        z_tau = target[:, self.k : self.k + self.n_spice]
        sig_hat = self._sigma0_from_raw(output)
        if self.sigma0_mean is not None:
            sig_hat = (sig_hat - self.sigma0_mean) / self.sigma0_std
        loss_rho = torch.mean((sig_hat - sig_tgt) ** 2)
        loss_tau = torch.mean((z_tau_hat - z_tau) ** 2)
        return self.lambda_rho * loss_rho + self.lambda_tau * loss_tau


class DensitySpiceProbLoss(DensitySpiceLoss):
    """Phase 4: CRPS / β-NLL / quantile / MSE on standardized (σ₀_ctrl, spice PCs)."""

    def __init__(
        self,
        outputs,
        dz_tilde,
        device,
        *,
        lambda_rho: float = 1.0,
        lambda_tau: float = 1.0,
        sigma0_mean: np.ndarray | None = None,
        sigma0_std: np.ndarray | None = None,
        a_clim: np.ndarray | torch.Tensor | None = None,
        delta_sigma0_basis: np.ndarray | torch.Tensor | None = None,
        sigma0_clim: np.ndarray | torch.Tensor | None = None,
        n_ctrl: int | None = None,
        delta_a_basis: np.ndarray | torch.Tensor | None = None,
        prob_mode: str = "crps",
        nll_beta: float = 0.5,
        sigma_min: float = 1e-3,
        freeze_sigma: bool = False,
        target_err: torch.Tensor | None = None,
    ):
        super().__init__(
            outputs,
            dz_tilde,
            device,
            lambda_rho=lambda_rho,
            lambda_tau=lambda_tau,
            sigma0_mean=sigma0_mean,
            sigma0_std=sigma0_std,
            a_clim=a_clim,
            delta_sigma0_basis=delta_sigma0_basis,
            sigma0_clim=sigma0_clim,
            n_ctrl=n_ctrl,
            delta_a_basis=delta_a_basis,
        )
        if prob_mode not in VALID_PROB_MODES:
            raise ValueError(f"prob_mode must be one of {VALID_PROB_MODES}, got {prob_mode!r}")
        if prob_mode == "quantile" and self.delta_sigma0_basis is not None:
            raise NotImplementedError(
                "quantile + low-rank δσ₀ deferred; use crps/nll (score-σ → induced σ₀)"
            )
        self.prob_mode = prob_mode
        self.nll_beta = float(nll_beta)
        self.sigma_min = float(sigma_min)
        self.freeze_sigma = bool(freeze_sigma)
        self.d = self.n_scores + self.n_spice
        if target_err is not None:
            self.register_buffer("target_err", torch.as_tensor(target_err, dtype=torch.float32, device=device))
        else:
            self.target_err = None

    def _mu_from_raw(self, mu_raw: torch.Tensor) -> torch.Tensor:
        z_tau = mu_raw[:, self.n_scores : self.n_scores + self.n_spice]
        sig_hat = self._sigma0_from_raw(mu_raw)
        if self.sigma0_mean is not None:
            sig_hat = (sig_hat - self.sigma0_mean) / self.sigma0_std
        return torch.cat([sig_hat, z_tau], dim=-1)

    def _sigma_target_space(self, sigma_lat: torch.Tensor) -> torch.Tensor:
        """Map latent σ to standardized (σ₀_ctrl || spice) for CRPS/NLL.

        Full-rank: σ already per ctrl level (+ spice PCs).
        Low-rank: density σ is on R scores; induce
        ``σ_i = sqrt(Σ_r V[r,i]² σ_z[r]²) / σ0_std[i]`` (diag of Σ_ρ = V diag(σ_z²) Vᵀ).
        """
        if self.delta_sigma0_basis is None:
            if sigma_lat.shape[-1] != self.k + self.n_spice:
                raise ValueError(
                    f"full-rank prob σ width {sigma_lat.shape[-1]} != K+spice "
                    f"{self.k + self.n_spice}"
                )
            return sigma_lat
        if sigma_lat.shape[-1] != self.d:
            raise ValueError(
                f"low-rank prob σ width {sigma_lat.shape[-1]} != R+spice {self.d}"
            )
        sz = sigma_lat[:, : self.n_scores]
        st = sigma_lat[:, self.n_scores :]
        # basis (R, K): var_i = Σ_r V[r,i]² σ_r²
        var = (self.delta_sigma0_basis.unsqueeze(0) ** 2) * (sz.unsqueeze(-1) ** 2)
        std_phys = torch.sqrt(var.sum(dim=1).clamp_min(1e-12))
        if self.sigma0_std is not None:
            std_phys = std_phys / self.sigma0_std
        return torch.cat([std_phys, st], dim=-1)

    def _sigma_tot(self, sigma: torch.Tensor, indices=None) -> torch.Tensor:
        """Phase 4.7: σ_tot² = σ_pred² + σ_target² when target_err buffer present."""
        if self.target_err is None or indices is None:
            return sigma
        te = self.target_err[indices]
        return torch.sqrt(sigma * sigma + te * te)

    def forward(self, output, target, indices=None, inputs=None):
        from evalphys.calibration import gaussian_crps_torch
        from model.prob_head import QUANTILE_TAUS, beta_nll, pinball_loss, split_mu_sigma

        if self.prob_mode == "mse" or self.freeze_sigma:
            # Stage 1 / deterministic: first D cols are μ raw (scores || z_τ)
            mu_raw = output[:, : self.d] if output.shape[-1] >= self.d else output
            mu = self._mu_from_raw(mu_raw)
            loss_rho = torch.mean((mu[:, : self.k] - target[:, : self.k]) ** 2)
            loss_tau = torch.mean((mu[:, self.k :] - target[:, self.k :]) ** 2)
            return self.lambda_rho * loss_rho + self.lambda_tau * loss_tau

        if self.prob_mode == "quantile":
            q = output.view(output.size(0), self.d, -1)
            # density: replace first K of each quantile with decoded-from-a? ponytail:
            # quantiles are predicted directly in target space (standardized σ₀ + spice).
            # For density block the raw head is not softplus-a; quantile mode drops hard
            # constraint during training (eval still uses a CRPS/NLL winner). Documented.
            return pinball_loss(q, target, QUANTILE_TAUS)

        mu_raw, sigma_lat = split_mu_sigma(output, self.d)
        mu = self._mu_from_raw(mu_raw)
        sigma = self._sigma_tot(self._sigma_target_space(sigma_lat), indices)
        if sigma.shape[-1] != target.shape[-1]:
            raise ValueError(
                f"prob σ width {sigma.shape[-1]} != target {target.shape[-1]}"
            )
        if self.prob_mode == "crps":
            crps = gaussian_crps_torch(mu, sigma, target, sigma_min=self.sigma_min)
            w = torch.ones_like(crps)
            w[:, : self.k] = self.lambda_rho
            w[:, self.k :] = self.lambda_tau
            return torch.mean(w * crps)
        nll = beta_nll(mu, sigma, target, beta=self.nll_beta, sigma_min=self.sigma_min)
        return nll


def make_loss(
    *,
    pca_models,
    outputs,
    weights,
    device,
    density_config: dict[str, Any] | None = None,
    density_meta=None,
    steric_config: dict[str, Any] | None = None,
    steric_meta=None,
    clim_profiles=None,
    loss_scales: dict[str, Any] | None = None,
    loss_config: dict[str, Any] | None = None,
    targets=None,
    true_profiles=None,
    ae_targets=None,
    ae_weights=None,
    surface_residual_layout: Mapping[str, Any] | None = None,
    bottom_depth=None,
    pres_levels=None,
    density_spice_meta: Mapping[str, Any] | None = None,
    joint_eof_meta: Mapping[str, Any] | None = None,
    **kwargs,
) -> nn.Module:
    scales = loss_scales or {}
    cfg = loss_config or {}
    mode = cfg.get("mode", "combined")
    if mode not in VALID_LOSS_MODES:
        raise ValueError(f"unknown loss_config.mode {mode!r}; expected one of {VALID_LOSS_MODES}")

    if mode == "density_spice":
        meta = density_spice_meta or {}
        if "dz_tilde" not in meta:
            raise ValueError("density_spice mode requires density_spice_meta['dz_tilde'] from cache")
        a_clim = meta.get("a_clim")
        if a_clim is None and meta.get("sigma0_ctrl_mean") is not None:
            from model.density_spice import encode_a_from_sigma0_ctrl

            a_clim = encode_a_from_sigma0_ctrl(
                np.asarray(meta["sigma0_ctrl_mean"], dtype=np.float64),
                np.asarray(meta["dz_tilde"], dtype=np.float64),
                np.asarray(meta["z_ctrl"], dtype=np.float64),
            )
        n_ctrl = int(meta.get("K", outputs["density_ctrl"]))
        basis = meta.get("delta_sigma0_basis")
        sigma0_clim = meta.get("sigma0_clim", meta.get("sigma0_ctrl_mean"))
        prob_mode = cfg.get("prob_mode")
        if prob_mode:
            return DensitySpiceProbLoss(
                outputs=outputs,
                dz_tilde=meta["dz_tilde"],
                device=device,
                lambda_rho=float(scales.get("lambda_rho", 1.0)),
                lambda_tau=float(scales.get("lambda_tau", 1.0)),
                sigma0_mean=meta.get("sigma0_ctrl_mean"),
                sigma0_std=meta.get("sigma0_ctrl_std"),
                a_clim=a_clim,
                delta_sigma0_basis=basis,
                sigma0_clim=sigma0_clim,
                n_ctrl=n_ctrl,
                prob_mode=str(prob_mode),
                nll_beta=float(cfg.get("nll_beta", 0.5)),
                sigma_min=float(cfg.get("sigma_min", 1e-3)),
                freeze_sigma=bool(cfg.get("freeze_sigma", False)),
                target_err=kwargs.get("target_err"),
            )
        return DensitySpiceLoss(
            outputs=outputs,
            dz_tilde=meta["dz_tilde"],
            device=device,
            lambda_rho=float(scales.get("lambda_rho", 1.0)),
            lambda_tau=float(scales.get("lambda_tau", 1.0)),
            sigma0_mean=meta.get("sigma0_ctrl_mean"),
            sigma0_std=meta.get("sigma0_ctrl_std"),
            a_clim=a_clim,
            delta_sigma0_basis=basis,
            sigma0_clim=sigma0_clim,
            n_ctrl=n_ctrl,
        )

    # A/B matrix cells: PC-space hetero when probabilistic
    if mode in ("combined", "pc_mse_only") and cfg.get("prob_mode"):
        return PCAHeteroLoss(
            prob_mode=str(cfg["prob_mode"]),
            nll_beta=float(cfg.get("nll_beta", 0.5)),
            sigma_min=float(cfg.get("sigma_min", 1e-3)),
            freeze_sigma=bool(cfg.get("freeze_sigma", False)),
        )

    cached_profiles = None
    if mode in ("pred_profile_cached", "decoder"):
        cached_profiles = compute_true_profiles(
            targets,
            pca_models,
            outputs,
            device,
            cached=true_profiles,
        )
    elif joint_eof_meta is not None:
        if true_profiles is None:
            raise ValueError("joint_eof requires cache true_profiles temperature+salinity")
        cached_profiles = true_profiles

    decoders = None
    latent_weights = weights
    if mode == "decoder":
        decoder_dir = cfg.get("decoder_dir")
        if not decoder_dir:
            raise ValueError("loss_config.decoder_dir required when mode='decoder'")
        decoder_path = resolve_decoder_dir(decoder_dir)
        if ae_targets is None:
            raise ValueError(
                "decoder mode requires ae_targets in cache; run scripts/export_ae_latents.py first"
            )
        decoders = load_decoders_from_dir(decoder_path, outputs, device)
        if surface_residual_layout is None and any(getattr(d, "surface_residual", False) for d in decoders.values()):
            raise ValueError("surface-residual decoders require surface_residual_layout (from cache metadata)")
        latent_weights = ae_weights if ae_weights is not None else ae_weights_numpy(np.asarray(ae_targets), outputs)

    return CombinedPCALoss(
        pca_models=pca_models,
        outputs=outputs,
        weights=latent_weights,
        device=device,
        density_config=density_config,
        density_meta=density_meta,
        steric_config=steric_config,
        steric_meta=steric_meta,
        clim_profiles=clim_profiles,
        profile_scales=scales.get("profile_scales"),
        combined_pca_scale=scales.get("combined_pca_scale"),
        combined_mse_scale=scales.get("combined_mse_scale"),
        mode=mode,
        true_profiles=cached_profiles,
        decoders=decoders,
        surface_residual_layout=surface_residual_layout,
        bottom_depth=bottom_depth,
        pres_levels=pres_levels,
        joint_eof_meta=joint_eof_meta,
    )
