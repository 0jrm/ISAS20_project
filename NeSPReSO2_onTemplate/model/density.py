"""Frozen density surrogate and differentiable training penalties."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class RhoMLP(nn.Module):
    """Small MLP used to approximate density from salinity/temperature/pressure."""

    def __init__(self, in_dim=6, hidden=64, depth=2):
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_dim, hidden))
        layers.append(nn.ReLU())
        for _ in range(max(depth - 1, 0)):
            layers.append(nn.Linear(hidden, hidden))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DensityConstraint:
    """Applies differentiable penalties based on densities computed by a frozen surrogate."""

    def __init__(self, dataset, device, config):
        self.device = device
        self.config = config
        self.stab_weight = config.get("stab_weight", 0.0)
        self.smooth_weight = config.get("smooth_weight", 0.0)
        self.tol = config.get("stability_tol", 0.0)

        stats = np.load(config["stats_path"])
        self.x_mean = torch.tensor(stats["x_mean"], dtype=torch.float32, device=device)
        self.x_std = torch.tensor(stats["x_std"], dtype=torch.float32, device=device)
        self.y_mean = torch.tensor(stats["y_mean"], dtype=torch.float32, device=device)
        self.y_std = torch.tensor(stats["y_std"], dtype=torch.float32, device=device)

        checkpoint = torch.load(config["checkpoint"], map_location=device)
        hidden = checkpoint.get("width", 64)
        depth = checkpoint.get("depth", 2)
        self.rho_model = RhoMLP(in_dim=6, hidden=hidden, depth=depth).to(device)
        self.rho_model.load_state_dict(checkpoint["model_state_dict"])
        self.rho_model.eval()
        for param in self.rho_model.parameters():
            param.requires_grad_(False)

        lat = np.asarray(dataset.LAT).squeeze()
        lon = np.asarray(dataset.LON).squeeze()
        self.latitudes = torch.tensor(lat, dtype=torch.float32, device=device)
        self.longitudes = torch.tensor(lon, dtype=torch.float32, device=device)

        if hasattr(dataset, "PRES") and dataset.PRES is not None:
            self.pressures = torch.tensor(dataset.PRES, dtype=torch.float32, device=device)
        else:
            depth_axis = np.arange(dataset.min_depth, dataset.max_depth + 1)
            self.pressures = torch.tensor(depth_axis[:, None], dtype=torch.float32, device=device).expand(
                -1, len(self.latitudes)
            )

        self.min_depth = dataset.min_depth
        self.max_depth = dataset.max_depth
        self.depth_count = self.max_depth - self.min_depth + 1

        smooth_window = config.get("smooth_window", (self.min_depth, self.max_depth))
        start = max(smooth_window[0], self.min_depth)
        end = min(smooth_window[1], self.max_depth)
        self.smooth_start = max(start - self.min_depth, 1)
        self.smooth_end = min(end - self.min_depth, self.depth_count - 2)

    def _gather_pressure(self, indices, depth):
        if self.pressures.dim() == 2:
            gathered = self.pressures[:, indices]
            return gathered.transpose(0, 1)
        return self.pressures.unsqueeze(0).expand(indices.shape[0], depth)

    def __call__(self, temp_profiles, sal_profiles, indices):
        if (self.stab_weight <= 0 and self.smooth_weight <= 0) or indices is None:
            return temp_profiles.new_tensor(0.0)

        if not torch.is_tensor(indices):
            indices = torch.tensor(indices, dtype=torch.long, device=self.device)
        else:
            indices = indices.to(self.device)

        batch_size, depth = temp_profiles.shape
        pressure = self._gather_pressure(indices, depth)

        lat = self.latitudes[indices]
        lon = self.longitudes[indices]
        lon_rad = torch.deg2rad(lon)
        sin_lon = torch.sin(lon_rad)
        cos_lon = torch.cos(lon_rad)

        lat = lat.unsqueeze(1).expand(-1, depth)
        sin_lon = sin_lon.unsqueeze(1).expand_as(lat)
        cos_lon = cos_lon.unsqueeze(1).expand_as(lat)
        pressure = pressure.to(temp_profiles.device)

        feature_stack = torch.stack([sal_profiles, temp_profiles, pressure, sin_lon, cos_lon, lat], dim=-1)
        X = feature_stack.reshape(-1, 6)
        X_norm = (X - self.x_mean) / self.x_std
        rho_norm = self.rho_model(X_norm)
        rho = rho_norm * self.y_std + self.y_mean
        rho = rho.view(batch_size, depth)

        total_penalty = temp_profiles.new_tensor(0.0, device=temp_profiles.device)

        second = rho[:, 2:] - 2 * rho[:, 1:-1] + rho[:, :-2]

        if self.stab_weight > 0:
            stab_penalty = second.pow(2).mean()
            total_penalty = total_penalty + self.stab_weight * stab_penalty

        if self.smooth_weight > 0 and self.smooth_end > self.smooth_start:
            start_idx = self.smooth_start - 1
            end_idx = self.smooth_end
            smooth_slice = second[:, start_idx:end_idx]
            smooth_penalty = smooth_slice.pow(2).mean()
            total_penalty = total_penalty + self.smooth_weight * smooth_penalty

        return total_penalty
