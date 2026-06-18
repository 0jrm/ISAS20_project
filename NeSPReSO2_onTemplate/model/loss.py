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

# ponytail: GoM temp/sal-specific scales; re-derive for new outputs or regions
DEFAULT_PROFILE_SCALES = {"temperature": 37.86, "salinity": 0.28}
DEFAULT_COMBINED_PCA_SCALE = 2.8294
DEFAULT_COMBINED_MSE_SCALE = 0.0255

OUTPUT_H5_VARS = {"temperature": "TEMP", "salinity": "PSAL"}

VALID_LOSS_MODES = ("combined", "pred_profile_cached", "pc_mse_only", "decoder")


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
    from preproc.preproc_isas_sat import count_encoding_dims, surface_residual_from_features

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
            layout.setdefault("n_enc", count_encoding_dims(layout.get("input_params", {})))
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
                from preproc.preproc_isas_sat import count_encoding_dims, surface_residual_from_features

                layout = dict(self.surface_residual_layout or {})
                layout.setdefault("n_enc", count_encoding_dims(layout.get("input_params", {})))
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
        decoders: Mapping[str, nn.Module] | None = None,
        surface_residual_layout: Mapping[str, Any] | None = None,
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
        self.density_helper = None
        if density_config and density_config.get("enabled", False):
            self.density_helper = DensityConstraint(dataset=density_meta, device=device, config=density_config)

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

        if self.density_helper is not None and indices is not None and len(self.output_order) >= 2:
            temp_name, sal_name = self.output_order[0], self.output_order[1]
            recon = self._reconstruct_profiles(pcs, inputs=inputs)
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
    ae_targets=None,
    ae_weights=None,
    surface_residual_layout: Mapping[str, Any] | None = None,
    **kwargs,
) -> CombinedPCALoss:
    scales = loss_scales or {}
    cfg = loss_config or {}
    mode = cfg.get("mode", "combined")
    if mode not in VALID_LOSS_MODES:
        raise ValueError(f"unknown loss_config.mode {mode!r}; expected one of {VALID_LOSS_MODES}")

    cached_profiles = None
    if mode in ("pred_profile_cached", "decoder"):
        cached_profiles = compute_true_profiles(
            targets,
            pca_models,
            outputs,
            device,
            cached=true_profiles,
        )

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
        profile_scales=scales.get("profile_scales"),
        combined_pca_scale=scales.get("combined_pca_scale"),
        combined_mse_scale=scales.get("combined_mse_scale"),
        mode=mode,
        true_profiles=cached_profiles,
        decoders=decoders,
        surface_residual_layout=surface_residual_layout,
        **kwargs,
    )
