"""Optional scalar diffusion-time conditioning for denoiser queries."""

from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalTimeFeatures(nn.Module):
    """Fixed Fourier features for scalar ``t`` in ``[0, 1]``."""

    def __init__(self, feature_dim: int = 128, max_period: float = 10_000.0) -> None:
        super().__init__()
        if feature_dim < 2:
            raise ValueError("feature_dim must be >= 2")
        if max_period <= 1:
            raise ValueError("max_period must be > 1")
        self.feature_dim = int(feature_dim)
        half_dim = feature_dim // 2
        denominator = max(half_dim - 1, 1)
        frequencies = torch.exp(
            -math.log(max_period) * torch.arange(half_dim, dtype=torch.float32) / denominator
        )
        self.register_buffer("frequencies", frequencies, persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim != 1:
            raise ValueError(f"t must have shape [batch], got {tuple(t.shape)}")
        if not t.is_floating_point():
            raise TypeError("t must be floating point")
        if not torch.isfinite(t).all() or (t < 0).any() or (t > 1).any():
            raise ValueError("t must contain finite values in [0, 1]")
        angles = 2 * math.pi * t.float()[:, None] * self.frequencies[None, :]
        features = torch.cat((angles.sin(), angles.cos()), dim=-1)
        if self.feature_dim % 2:
            features = torch.cat((features, t.float()[:, None]), dim=-1)
        return features


class TimeConditioningMLP(nn.Module):
    """Sinusoidal features followed by Linear-SiLU-Linear to model width.

    The final projection is zero-initialized by default, so enabling the module
    initially preserves the AR checkpoint's denoiser embeddings exactly while
    still allowing gradients to train the projection.
    """

    def __init__(
        self,
        hidden_size: int,
        *,
        feature_dim: int = 128,
        mlp_hidden_size: int | None = None,
        zero_init_output: bool = True,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be > 0")
        inner_size = hidden_size if mlp_hidden_size is None else int(mlp_hidden_size)
        if inner_size <= 0:
            raise ValueError("mlp_hidden_size must be > 0")
        self.features = SinusoidalTimeFeatures(feature_dim=feature_dim)
        self.in_proj = nn.Linear(feature_dim, inner_size)
        self.activation = nn.SiLU()
        self.out_proj = nn.Linear(inner_size, hidden_size)
        if zero_init_output:
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        features = self.features(t).to(dtype=self.in_proj.weight.dtype)
        return self.out_proj(self.activation(self.in_proj(features)))


__all__ = ["SinusoidalTimeFeatures", "TimeConditioningMLP"]
