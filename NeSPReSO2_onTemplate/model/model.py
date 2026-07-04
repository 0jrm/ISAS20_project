import torch
import torch.nn as nn
import torch.nn.functional as F
from base.base_model import BaseModel
import math


class ResidualLinearBlock(nn.Module):
    """Linear block with a projection shortcut when widths differ."""

    def __init__(self, in_dim, out_dim, dropout_prob=0.0):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout_prob) if dropout_prob > 0 else nn.Identity()
        self.skip = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x):
        return F.relu(self.fc(x)) + self.skip(x)


class ResidualConvBlock(nn.Module):
    """Two-layer conv block with a 1x1 shortcut when channels differ."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        )
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x) + self.skip(x))

class FFNN(BaseModel):
    def __init__(self, input_dim=1, layers_config=[512, 256], output_dim=30, dropout_prob = 0.5, activation = nn.ReLU()):
        super(FFNN, self).__init__()
        
        # Construct layers based on the given configuration
        layers = []
        prev_dim = input_dim
        for neurons in layers_config:
            layers.append(nn.Linear(prev_dim, neurons))
            layers.append(activation) # can be changed to an array, just as layers_config
            if dropout_prob > 0:
                layers.append(nn.Dropout(dropout_prob)) # added dropout
            prev_dim = neurons
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class PredictionModel(BaseModel):
    """MLP for PCA-component prediction from surface inputs (v2 ``PredictionModel``)."""

    def __init__(self, input_dim=1, layers_config=None, output_dim=30, dropout_prob=0.5):
        super(PredictionModel, self).__init__()
        if layers_config is None:
            layers_config = [512, 256]
        layers = []
        prev_dim = input_dim
        self.layers_config = layers_config
        for neurons in layers_config:
            layers.append(nn.Linear(prev_dim, neurons))
            layers.append(nn.ReLU())
            if dropout_prob > 0:
                layers.append(nn.Dropout(dropout_prob))
            prev_dim = neurons
        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class PatchConvMLP(BaseModel):
    """
    Patch-aware surface encoder + MLP head for PCA latent prediction.

    Point mode (``patch_shape=None``): linear embed of 3 satellite scalars.
    Patch mode: reshape flattened sat block to ``(B, C, T, H, W)`` and run a small Conv2d trunk.
    """

    def __init__(
        self,
        input_dim=9,
        output_dim=32,
        dropout_prob=0.2,
        d_model=128,
        head_layers=None,
        conv_channels=None,
        patch_shape=None,
        n_enc=6,
        n_sat=3,
        residual=False,
        **kwargs,
    ):
        super().__init__()
        if head_layers is None:
            head_layers = [1024, 1024]
        if conv_channels is None:
            conv_channels = [32, 64]

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_enc = n_enc
        self.n_sat = n_sat
        self.d_model = d_model
        self.residual = bool(residual)
        self.patch_shape = tuple(patch_shape) if patch_shape else None

        self.enc_proj = nn.Linear(n_enc, d_model)

        if self.patch_shape is None:
            self.sat_proj = nn.Linear(n_sat, d_model)
            self.conv = None
        else:
            c, t, h, w = self.patch_shape
            per_var = t * h * w
            expected_sat = c * per_var
            if input_dim != n_enc + expected_sat:
                raise ValueError(
                    f"PatchConvMLP input_dim={input_dim} != n_enc({n_enc}) + sat({expected_sat})"
                )
            if self.residual:
                blocks = []
                in_ch = c
                for out_ch in conv_channels:
                    blocks.append(ResidualConvBlock(in_ch, out_ch))
                    in_ch = out_ch
                blocks.append(nn.AdaptiveAvgPool2d(1))
                self.conv = nn.Sequential(*blocks)
            else:
                layers = []
                in_ch = c
                for out_ch in conv_channels:
                    layers.extend(
                        [
                            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                            nn.ReLU(inplace=True),
                        ]
                    )
                    in_ch = out_ch
                layers.append(nn.AdaptiveAvgPool2d(1))
                self.conv = nn.Sequential(*layers)
            self.sat_proj = nn.Linear(conv_channels[-1], d_model)

        if self.residual:
            head_blocks = []
            prev = d_model
            for width in head_layers:
                head_blocks.append(ResidualLinearBlock(prev, width, dropout_prob))
                prev = width
            self.head_blocks = nn.ModuleList(head_blocks)
            self.head_out = nn.Linear(prev, output_dim)
            self.head = None
        else:
            head = []
            prev = d_model
            for width in head_layers:
                head.extend([nn.Linear(prev, width), nn.ReLU(), nn.Dropout(dropout_prob)])
                prev = width
            head.append(nn.Linear(prev, output_dim))
            self.head = nn.Sequential(*head)
            self.head_blocks = None
            self.head_out = None

    def _encode_sat_point(self, sat_flat):
        return self.sat_proj(sat_flat)

    def _encode_sat_patch(self, sat_flat):
        b = sat_flat.size(0)
        c, t, h, w = self.patch_shape
        per_var = t * h * w
        sat = sat_flat.view(b, c, per_var).view(b, c, t, h, w)
        sat = sat.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        sat = self.conv(sat).view(b, t, -1).mean(dim=1)
        return self.sat_proj(sat)

    def forward(self, x):
        enc = x[:, : self.n_enc]
        sat_flat = x[:, self.n_enc :]
        h = self.enc_proj(enc)
        if self.patch_shape is None:
            h = h + self._encode_sat_point(sat_flat)
        else:
            h = h + self._encode_sat_patch(sat_flat)
        if self.head_blocks is not None:
            for block in self.head_blocks:
                h = block(h)
            return self.head_out(h)
        return self.head(h)


class PatchMaskConvMLP(BaseModel):
    """
    Spatio-temporal patch encoder + MLP head.

    With ``use_mask_channels=True``, flat satellite columns are pixel-interleaved
    ``val, mask`` per variable (6 conv channels). With ``use_mask_channels=False``
    (default), only value planes are used (3 channels); NaN pixels are zero-filled in preproc.
    """

    def __init__(
        self,
        input_dim=535,
        output_dim=32,
        dropout_prob=0.2,
        d_model=128,
        head_layers=None,
        head_hidden=512,
        head_depth=2,
        head_dropout=None,
        conv_channels=None,
        patch_shape=None,
        pool_output=None,
        fusion="concat",
        conv_norm_groups=8,
        n_enc=10,
        n_sat=3,
        use_mask_channels=False,
        residual=False,
        **kwargs,
    ):
        super().__init__()
        if conv_channels is None:
            conv_channels = [32, 64]
        if pool_output is None:
            pool_output = [1, 1, 1]
        if head_dropout is None:
            head_dropout = dropout_prob

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_enc = n_enc
        self.n_sat = n_sat
        self.d_model = d_model
        self.residual = bool(residual)
        self.fusion = fusion
        self.use_mask_channels = bool(use_mask_channels)
        self.patch_shape = tuple(patch_shape) if patch_shape else None
        self.pool_output = tuple(pool_output)

        if self.patch_shape is None:
            raise ValueError("PatchMaskConvMLP requires patch_shape")
        if fusion not in ("add", "concat"):
            raise ValueError(f"fusion must be 'add' or 'concat', got {fusion!r}")

        c, t, h, w = self.patch_shape
        per_ch = t * h * w
        expected_sat = c * per_ch
        if input_dim != n_enc + expected_sat:
            raise ValueError(
                f"PatchMaskConvMLP input_dim={input_dim} != n_enc({n_enc}) + sat({expected_sat})"
            )
        if self.use_mask_channels and c % 2 != 0:
            raise ValueError(
                f"use_mask_channels requires even patch_shape channels (value/mask pairs), got c={c}"
            )

        self.enc_proj = nn.Linear(n_enc, d_model)

        layers = []
        in_ch = c
        for out_ch in conv_channels:
            layers.extend(
                [
                    nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.GroupNorm(conv_norm_groups, out_ch),
                    nn.ReLU(inplace=True),
                ]
            )
            in_ch = out_ch
        # Default [1,1,1] collapses the volume; e.g. [2,2,2] retains coarse structure
        # at the cost of a larger sat_proj input (last_conv_ch * prod(pool_output)).
        layers.append(nn.AdaptiveAvgPool3d(self.pool_output))
        self.conv3d = nn.Sequential(*layers)
        pool_flat = conv_channels[-1] * math.prod(self.pool_output)
        self.sat_proj = nn.Linear(pool_flat, d_model)

        if head_layers is not None:
            widths = list(head_layers)
        else:
            widths = [head_hidden] * head_depth

        head_in = d_model if fusion == "add" else 2 * d_model
        head = []
        prev = head_in
        for width in widths:
            head.extend([nn.Linear(prev, width), nn.ReLU(), nn.Dropout(head_dropout)])
            prev = width
        head.append(nn.Linear(prev, output_dim))
        self.head = nn.Sequential(*head)

    def _unpack_sat_channels(self, sat_flat):
        """Decode flat satellite block to ``(B, C, T, H, W)``."""
        b = sat_flat.size(0)
        c, t, h, w = self.patch_shape
        per_ch = t * h * w
        if not self.use_mask_channels:
            return sat_flat.view(b, c, per_ch).view(b, c, t, h, w)
        n_pairs = c // 2
        channels = []
        for v in range(n_pairs):
            block = sat_flat[:, v * 2 * per_ch : (v + 1) * 2 * per_ch]
            channels.append(block[:, 0::2])
            channels.append(block[:, 1::2])
        return torch.stack(channels, dim=1).view(b, c, t, h, w)

    def _encode_sat_patch(self, sat_flat):
        b = sat_flat.size(0)
        sat_vol = self._unpack_sat_channels(sat_flat)
        feat = self.conv3d(sat_vol).view(b, -1)
        return self.sat_proj(feat)

    def forward(self, x):
        enc = x[:, : self.n_enc]
        sat_flat = x[:, self.n_enc :]
        enc_h = self.enc_proj(enc)
        sat_h = self._encode_sat_patch(sat_flat)
        if self.fusion == "add":
            h = enc_h + sat_h
        else:
            h = torch.cat([enc_h, sat_h], dim=-1)
        return self.head(h)


def _zero_init_linear(layer: nn.Linear) -> None:
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


def _load_warmstart_state(module: nn.Module, ckpt_path: str, *, prefix: str = "") -> None:
    """Load compatible weights from a checkpoint into ``module``."""
    import os

    if not ckpt_path or not os.path.isfile(ckpt_path):
        return
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
    own = module.state_dict()
    mapped = {}
    for key, val in state.items():
        k = key
        if k.startswith("module."):
            k = k[len("module.") :]
        if prefix and not k.startswith(prefix):
            target = prefix + k
        else:
            target = k
        if target in own and own[target].shape == val.shape:
            mapped[target] = val
    module.load_state_dict(mapped, strict=False)


class ResidualPatchEncoder(nn.Module):
    """Small Conv3d trunk preserving coarse spatial structure."""

    def __init__(
        self,
        patch_shape,
        conv_channels=None,
        pool_output=None,
        d_model=128,
        conv_norm_groups=8,
        dropout_prob=0.2,
    ):
        super().__init__()
        if conv_channels is None:
            conv_channels = [16, 32]
        if pool_output is None:
            pool_output = [2, 3, 3]
        self.patch_shape = tuple(patch_shape)
        self.pool_output = tuple(pool_output)
        c, t, h, w = self.patch_shape
        layers = []
        in_ch = c
        for out_ch in conv_channels:
            layers.extend(
                [
                    nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.GroupNorm(conv_norm_groups, out_ch),
                    nn.ReLU(inplace=True),
                ]
            )
            in_ch = out_ch
        layers.append(nn.AdaptiveAvgPool3d(self.pool_output))
        self.conv3d = nn.Sequential(*layers)
        pool_flat = conv_channels[-1] * math.prod(self.pool_output)
        self.proj = nn.Sequential(
            nn.Linear(pool_flat, d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
        )

    def forward(self, sat_flat: torch.Tensor) -> torch.Tensor:
        b = sat_flat.size(0)
        c, t, h, w = self.patch_shape
        per_ch = t * h * w
        vol = sat_flat.view(b, c, per_ch).view(b, c, t, h, w)
        feat = self.conv3d(vol).view(b, -1)
        return self.proj(feat)


class ResidualPatchModel(BaseModel):
    """
    Point-anchored residual patch model: ``Output = Base + g * ΔPC``.

    Base branch is a warm-startable ``PatchConvMLP`` in point mode.
    Patch branch emits a zero-initialized residual correction.
    """

    def __init__(
        self,
        input_dim=534,
        output_dim=32,
        base_dim=9,
        patch_offset=9,
        patch_shape=None,
        d_model=128,
        dropout_prob=0.2,
        conv_channels=None,
        pool_output=None,
        head_hidden=256,
        head_depth=2,
        gate_per_pc=False,
        freeze_base=False,
        warmstart_ckpt=None,
        warmstart_patch_ckpt=None,
        base_head_layers=None,
        conv_norm_groups=8,
        n_enc=6,
        n_sat=3,
        **kwargs,
    ):
        super().__init__()
        if base_head_layers is None:
            base_head_layers = [1024, 1024]
        if patch_shape is None:
            raise ValueError("ResidualPatchModel requires patch_shape")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.base_dim = int(base_dim)
        self.patch_offset = int(patch_offset)
        self.patch_shape = tuple(patch_shape)
        self.gate_per_pc = bool(gate_per_pc)

        self.base = PatchConvMLP(
            input_dim=base_dim,
            output_dim=output_dim,
            dropout_prob=dropout_prob,
            d_model=d_model,
            head_layers=base_head_layers,
            patch_shape=None,
            n_enc=n_enc,
            n_sat=n_sat,
        )
        self.patch_enc = ResidualPatchEncoder(
            patch_shape=self.patch_shape,
            conv_channels=conv_channels or [16, 32],
            pool_output=pool_output or [2, 3, 3],
            d_model=d_model,
            conv_norm_groups=conv_norm_groups,
            dropout_prob=dropout_prob,
        )
        widths = [head_hidden] * head_depth
        head = []
        prev = d_model
        for width in widths:
            head.extend([nn.Linear(prev, width), nn.ReLU(), nn.Dropout(dropout_prob)])
            prev = width
        self.patch_head_hidden = nn.Sequential(*head) if head else nn.Identity()
        self.patch_head_out = nn.Linear(prev if widths else d_model, output_dim)
        _zero_init_linear(self.patch_head_out)

        if gate_per_pc:
            self.gate = nn.Parameter(torch.zeros(output_dim))
        else:
            self.gate = nn.Parameter(torch.zeros(1))

        if warmstart_ckpt:
            _load_warmstart_state(self.base, warmstart_ckpt)
        if warmstart_patch_ckpt:
            _load_warmstart_state(self.patch_enc, warmstart_patch_ckpt, prefix="patch_enc.")

        self.set_freeze_base(freeze_base)

    def set_freeze_base(self, freeze: bool = True) -> None:
        for param in self.base.parameters():
            param.requires_grad = not freeze

    def forward_base(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x[:, : self.base_dim])

    def forward_delta(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.patch_enc(x[:, self.patch_offset :])
        if isinstance(self.patch_head_hidden, nn.Sequential) and len(self.patch_head_hidden) > 0:
            feat = self.patch_head_hidden(feat)
        return self.patch_head_out(feat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.forward_base(x)
        delta = self.forward_delta(x)
        if self.gate_per_pc:
            return base + delta * self.gate
        return base + delta * self.gate.squeeze()


class Autoencoder(nn.Module):
    def __init__(
        self,
        encoding_dim,
        encoder_layers=None,
        decoder_layers=None,
        input_dim=187,
        residual=False,
        dropout_prob=0.0,
    ):
        super(Autoencoder, self).__init__()
        self.encoding_dim = encoding_dim
        self.input_dim = input_dim
        self.residual = bool(residual)
        if encoder_layers is None:
            encoder_layers = [128, 64, 32]
        if decoder_layers is None:
            decoder_layers = [32, 64, 128]

        if self.residual:
            enc_blocks = nn.ModuleList()
            prev_dim = input_dim
            for h in encoder_layers:
                enc_blocks.append(ResidualLinearBlock(prev_dim, h, dropout_prob))
                prev_dim = h
            self.enc_blocks = enc_blocks
            self.enc_out = nn.Linear(prev_dim, self.encoding_dim)
            self.encoder = None

            dec_blocks = nn.ModuleList()
            prev_dim = self.encoding_dim
            for h in decoder_layers:
                dec_blocks.append(ResidualLinearBlock(prev_dim, h, dropout_prob))
                prev_dim = h
            self.dec_blocks = dec_blocks
            self.dec_out = nn.Linear(prev_dim, input_dim)
            self.decoder = None
        else:
            encoder_modules = []
            prev_dim = input_dim
            for h in encoder_layers:
                encoder_modules.append(nn.Linear(prev_dim, h))
                encoder_modules.append(nn.ReLU())
                prev_dim = h
            encoder_modules.append(nn.Linear(prev_dim, self.encoding_dim))
            encoder_modules.append(nn.ReLU())
            self.encoder = nn.Sequential(*encoder_modules)

            decoder_modules = []
            prev_dim = self.encoding_dim
            for h in decoder_layers:
                decoder_modules.append(nn.Linear(prev_dim, h))
                decoder_modules.append(nn.ReLU())
                prev_dim = h
            decoder_modules.append(nn.Linear(prev_dim, input_dim))
            self.decoder = nn.Sequential(*decoder_modules)
            self.enc_blocks = None
            self.dec_blocks = None
            self.enc_out = None
            self.dec_out = None

    def _encode(self, x):
        if self.encoder is not None:
            return self.encoder(x)
        h = x
        for block in self.enc_blocks:
            h = block(h)
        return F.relu(self.enc_out(h))

    def _decode_body(self, encoded):
        if self.decoder is not None:
            return self.decoder(encoded)
        h = encoded
        for block in self.dec_blocks:
            h = block(h)
        return self.dec_out(h)

    def forward(self, x, mask=None):
        if mask is not None:
            x_masked = x * (~mask).float()
        else:
            x_masked = x
        encoded = self._encode(x_masked)
        decoded = self._decode_body(encoded)
        if mask is not None:
            decoded = torch.where(mask, x, decoded)
        return decoded

    def encode(self, x, mask=None):
        if mask is not None:
            x = x * (~mask).float()
        return self._encode(x)

    def decode(self, x, mask=None):
        return self._decode_body(x)


class ResAutoencoder(Autoencoder):
    """Profile AE with residual hidden blocks and optional surface scalar skip at decode."""

    def __init__(self, *args, variable="temperature", surface_residual=True, **kwargs):
        kwargs["residual"] = True
        super().__init__(*args, **kwargs)
        self.variable = variable
        self.surface_residual = bool(surface_residual)

    def _apply_surface_residual(self, decoded, surface_residual):
        if not self.surface_residual or surface_residual is None:
            return decoded
        return decoded + surface_residual.unsqueeze(-1)

    def forward(self, x, mask=None, surface_residual=None):
        if mask is not None:
            x_masked = x * (~mask).float()
        else:
            x_masked = x
        encoded = self._encode(x_masked)
        decoded = self._apply_surface_residual(self._decode_body(encoded), surface_residual)
        if mask is not None:
            decoded = torch.where(mask, x, decoded)
        return decoded

    def decode(self, x, mask=None, surface_residual=None):
        decoded = self._decode_body(x)
        return self._apply_surface_residual(decoded, surface_residual)
    
class KANLinear(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3, 
                 scale_noise=0.1, scale_base=1.0, scale_spline=1.0,
                 enable_standalone_scale_spline=True, base_activation=nn.SiLU,
                 grid_eps=0.02, grid_range=[-1, 1]):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0]).expand(in_features, -1).contiguous()
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features, grid_size + spline_order))
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(torch.Tensor(out_features, in_features))

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 1 / 2) * self.scale_noise / self.grid_size
            self.spline_weight.data.copy_((self.scale_spline if not self.enable_standalone_scale_spline else 1.0) * self.curve2coeff(self.grid.T[self.spline_order: -self.spline_order], noise))
            if self.enable_standalone_scale_spline:
                nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x):
        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = ((x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)]) * bases[:, :, :-1]) + \
                    ((grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:(-k)]) * bases[:, :, 1:])
        return bases.contiguous()

    def curve2coeff(self, x, y):
        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution
        result = solution.permute(2, 0, 1)
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (self.spline_scaler.unsqueeze(-1) if self.enable_standalone_scale_spline else 1.0)

    def forward(self, x):
        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(self.b_splines(x).view(x.size(0), -1), self.scaled_spline_weight.view(self.out_features, -1))
        return base_output + spline_output


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, bottleneck_size):
        super(Encoder, self).__init__()
        self.fc1 = KANLinear(input_size, hidden_size)
        self.fc2 = KANLinear(hidden_size, bottleneck_size)

    def forward(self, x, mask=None):
        if mask is not None:
            x = x * mask.float()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class Decoder(nn.Module):
    def __init__(self, bottleneck_size, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.fc1 = KANLinear(bottleneck_size, hidden_size)
        self.fc2 = KANLinear(hidden_size, output_size)

    def forward(self, x, mask=None):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # Removed tanh activation
        return x


class KAN_Autoencoder(nn.Module):
    def __init__(self, encoding_dim, input_dim=187, hidden_size=64):
        super(KAN_Autoencoder, self).__init__()
        self.encoding_dim = encoding_dim
        self.input_dim = input_dim
        bottleneck_size = encoding_dim

        self.encoder = Encoder(input_dim, hidden_size, bottleneck_size)
        self.decoder = Decoder(bottleneck_size, hidden_size, input_dim)

    def forward(self, x, mask=None):
        # Store original values for masked points
        if mask is not None:
            # Zero out masked values before encoding
            x_masked = x * (~mask).float()
        else:
            x_masked = x
            
        # Encode
        encoded = self.encoder(x_masked, mask)
        
        # Decode
        decoded = self.decoder(encoded, mask)
        
        # Restore original values for masked points
        if mask is not None:
            decoded = torch.where(mask, x, decoded)
        
        return decoded

    def encode(self, x, mask=None):
        if mask is not None:
            x = x * mask.float()
        return self.encoder(x, mask)

    def decode(self, x, mask=None):
        return self.decoder(x, mask)


class _UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class FieldUNet(BaseModel):
    """2-level U-Net for gridded SSH -> PCA fields (Phase C)."""

    def __init__(self, in_channels=7, out_channels=32, base_width=32, **kwargs):
        super().__init__()
        w = int(base_width)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.enc1 = _UNetBlock(in_channels, w)
        self.pool = nn.MaxPool2d(2)
        self.enc2 = _UNetBlock(w, w * 2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = _UNetBlock(w * 2 + w, w)
        self.head = nn.Conv2d(w, out_channels, kernel_size=1)

    def forward(self, x):
        h, w = x.shape[-2:]
        pad_h = (4 - h % 4) % 4
        pad_w = (4 - w % 4) % 4
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        d1 = self.dec1(torch.cat([self.up(e2), e1], dim=1))
        out = self.head(d1)
        if pad_h or pad_w:
            out = out[..., :h, :w]
        return out