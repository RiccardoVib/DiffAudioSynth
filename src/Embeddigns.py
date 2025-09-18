# Copyright (C) 2025 Riccardo Simionato, University of Oslo
# Inquiries: riccardo.simionato.vib@gmail.com.com
#
# This code is free software: you can redistribute it and/or modify it under the terms
# of the GNU Lesser General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
#
# This code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Less General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License along with this code.
# If not, see <http://www.gnu.org/licenses/>.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange
from torch import Tensor
from typing import Union, Sequence
from math import pi


class NumberEmbedder(nn.Module):
    def __init__(self, features: int, dim: int = 256, device='cpu'):
        super().__init__()
        assert dim % 2 == 0, f"dim must be divisible by 2, found {dim}"
        self.features = features
        self.weights = nn.Parameter(torch.randn(dim // 2, device=device))
        self.to_out = nn.Linear(in_features=dim + 1, out_features=features).to(device)

        # self.to(device)

    def to_embedding(self, x: Tensor) -> Tensor:
        x = rearrange(x, "b -> b 1")
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return self.to_out(fouriered)

    def forward(self, x: Union[Sequence[float], Tensor]) -> Tensor:
        if not torch.is_tensor(x):
            x = torch.tensor(x, device=self.weights.device)
        assert isinstance(x, Tensor)
        shape = x.shape
        x = rearrange(x, "... -> (...)")
        return self.to_embedding(x).view(*shape, self.features)  # type: ignore


class RandomFourierEmbedding(nn.Module):
    """
    Random Fourier Features (RFF) embedding for continuous inputs.

    This is particularly useful for:
    - Time embeddings in diffusion models
    - Positional encodings
    - Function approximation with neural networks

    The embedding maps scalar inputs to high-dimensional feature vectors
    using random Fourier features: [cos(2πBx), sin(2πBx)]
    where B is a random matrix sampled from a Gaussian distribution.
    """

    def __init__(self, input_dim=1, embedding_dim=256, scale=1.0, learnable=False):
        """
        Args:
            input_dim (int): Dimension of input (usually 1 for time/scalar)
            embedding_dim (int): Dimension of output embedding (should be even)
            scale (float): Scale parameter for the Gaussian distribution
            learnable (bool): Whether the random frequencies are learnable parameters
        """
        super().__init__()

        assert embedding_dim % 2 == 0, "embedding_dim must be even"

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.scale = scale

        # Random frequencies matrix B ~ N(0, scale²I)
        B = torch.randn(embedding_dim // 2, input_dim) * scale

        if learnable:
            self.register_parameter('B', nn.Parameter(B))
        else:
            self.register_buffer('B', B)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (..., input_dim)

        Returns:
            Embedding tensor of shape (..., embedding_dim)
        """
        # Ensure x has the right shape
        if x.dim() == 1:
            x = x.unsqueeze(-1)  # Add feature dimension

        # Compute 2πBx
        projections = 2 * np.pi * torch.mm(x.view(-1, self.input_dim), self.B.t())

        # Compute [cos(2πBx), sin(2πBx)]
        cos_proj = torch.cos(projections)
        sin_proj = torch.sin(projections)

        # Concatenate cos and sin components
        embeddings = torch.cat([cos_proj, sin_proj], dim=-1)

        # Reshape back to original batch dimensions
        original_shape = x.shape[:-1]
        return embeddings.view(*original_shape, self.embedding_dim)


class GaussianFourierProjection(nn.Module):
    """
    Gaussian Fourier Projection layer used in many diffusion models.
    This is a specific type of Random Fourier Features.
    """

    def __init__(self, embedding_dim=256, scale=16.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        """
        Args:
            x: Time steps, shape (batch_size,) or (batch_size, 1)
        """
        if x.dim() > 1:
            x = x.squeeze(-1)

        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class SinusoidalPositionalEmbedding(nn.Module):
    """
    Sinusoidal positional embeddings as used in Transformers.
    This is a deterministic version of Fourier embeddings.
    """

    def __init__(self, embedding_dim=256, max_len=10000):
        super().__init__()

        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()

        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() *
                             -(np.log(10000.0) / embedding_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Position indices, shape (batch_size,) or (batch_size, seq_len)
        """
        if x.dim() == 1:
            return self.pe[x.long()]
        else:
            return self.pe[x.long()]


class TimeEmbedding(nn.Module):
    """
    Complete time embedding module for diffusion models.
    Combines Random Fourier Features with MLPs.
    """

    def __init__(self, embedding_dim=256, hidden_dim=512, scale=16.0):
        super().__init__()

        self.fourier_proj = GaussianFourierProjection(embedding_dim, scale)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t):
        """
        Args:
            t: Time steps, shape (batch_size,)

        Returns:
            Time embeddings, shape (batch_size, hidden_dim)
        """
        t_emb = self.fourier_proj(t)
        return self.mlp(t_emb)
