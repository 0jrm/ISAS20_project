import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import math

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

        head = []
        prev = d_model
        for width in head_layers:
            head.extend([nn.Linear(prev, width), nn.ReLU(), nn.Dropout(dropout_prob)])
            prev = width
        head.append(nn.Linear(prev, output_dim))
        self.head = nn.Sequential(*head)

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
        return self.head(h)


class Autoencoder(nn.Module):
    def __init__(self, encoding_dim, encoder_layers=None, decoder_layers=None, input_dim=187):
        super(Autoencoder, self).__init__()
        self.encoding_dim = encoding_dim
        self.input_dim = input_dim
        # Default layers if not provided
        if encoder_layers is None:
            encoder_layers = [128, 64, 32]
        if decoder_layers is None:
            decoder_layers = [32, 64, 128]

        # Build encoder
        encoder_modules = []
        prev_dim = input_dim
        for h in encoder_layers:
            encoder_modules.append(nn.Linear(prev_dim, h))
            encoder_modules.append(nn.ReLU())
            prev_dim = h
        encoder_modules.append(nn.Linear(prev_dim, self.encoding_dim))
        encoder_modules.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_modules)

        # Build decoder
        decoder_modules = []
        prev_dim = self.encoding_dim
        for h in decoder_layers:
            decoder_modules.append(nn.Linear(prev_dim, h))
            decoder_modules.append(nn.ReLU())
            prev_dim = h
        decoder_modules.append(nn.Linear(prev_dim, input_dim))
        # No activation at the end
        self.decoder = nn.Sequential(*decoder_modules)

    def forward(self, x, mask=None):
        # Store original values for masked points
        if mask is not None:
            # Zero out masked values before encoding
            x_masked = x * (~mask).float()
        else:
            x_masked = x
        # Encode
        encoded = self.encoder(x_masked)
        # Decode
        decoded = self.decoder(encoded)
        # Restore original values for masked points
        if mask is not None:
            decoded = torch.where(mask, x, decoded)
        return decoded

    def encode(self, x, mask=None):
        if mask is not None:
            x = x * mask.float()
        return self.encoder(x)

    def decode(self, x, mask=None):
        decoded = self.decoder(x)
        return decoded
    
    
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
    def __init__(self, encoding_dim):
        super(KAN_Autoencoder, self).__init__()
        input_size = 187  # Number of levels
        hidden_size = 64
        bottleneck_size = encoding_dim

        self.encoder = Encoder(input_size, hidden_size, bottleneck_size)
        self.decoder = Decoder(bottleneck_size, hidden_size, input_size)

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