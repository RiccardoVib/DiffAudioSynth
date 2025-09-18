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
import math

class TimeEmbedding(nn.Module):
    """Simple sinusoidal time embedding"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=time.device) * -emb)
        emb = time[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class Modulation(nn.Module):
    """Simple feature modulation (FiLM)"""

    def __init__(self, channels, cond_dim):
        super().__init__()
        self.proj = nn.Linear(cond_dim, channels * 2)

    def forward(self, x, cond):
        scale, shift = self.proj(cond).chunk(2, dim=-1)
        scale = scale.unsqueeze(-1)
        shift = shift.unsqueeze(-1)
        return x * (1 + scale) + shift


class SelfAttention(nn.Module):
    """Simple self-attention with dropout"""

    def __init__(self, channels, dropout=0.1):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj = nn.Conv1d(channels, channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, C, L = x.shape
        residual = x
        x = self.norm(x)

        qkv = self.qkv(x).view(B, 3, C, L)
        q, k, v = qkv.unbind(1)

        # Simple attention with dropout
        attn = torch.matmul(q.transpose(-2, -1), k) / (C ** 0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)  # Dropout on attention weights
        out = torch.matmul(v, attn.transpose(-2, -1))

        out = self.proj(out)
        out = self.dropout(out)  # Dropout on output

        return residual + out


class FeedForward(nn.Module):
    """Simple feedforward layer"""

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.ff = nn.Sequential(
            nn.Conv1d(channels, channels * 2, 1),
            nn.GELU(),
            nn.Conv1d(channels * 2, channels, 1)
        )

    def forward(self, x):
        return x + self.ff(self.norm(x))


class InjectionBlock(nn.Module):
    """Simple feature injection"""

    def __init__(self, channels, inject_channels):
        super().__init__()
        self.proj = nn.Conv1d(inject_channels, channels, 1)

    def forward(self, x, inject_feat):
        if inject_feat is None:
            return x
        # Match length
        if inject_feat.shape[-1] != x.shape[-1]:
            inject_feat = F.interpolate(inject_feat, size=x.shape[-1], mode='linear')
        return x + self.proj(inject_feat)


class ResNetBlock(nn.Module):
    """Simple ResNet block with all components"""

    def __init__(self, in_ch, out_ch, time_dim, use_attention=False, inject_ch=None):
        super().__init__()

        # Main conv blocks
        self.block1 = nn.Sequential(
            nn.GroupNorm(num_groups=8, num_channels=in_ch),
            nn.SiLU(),
            nn.Conv1d(in_ch, out_ch, 3, padding=1)
        )

        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_ch)
        )

        self.block2 = nn.Sequential(
            nn.GroupNorm(num_groups=8, num_channels=out_ch),
            nn.SiLU(),
            nn.Conv1d(out_ch, out_ch, 3, padding=1)
        )

        # Residual connection
        self.shortcut = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

        # Components
        self.modulation = Modulation(out_ch, time_dim)
        self.attention = SelfAttention(out_ch) if use_attention else None
        self.feedforward = FeedForward(out_ch)
        self.injection = InjectionBlock(out_ch, inject_ch) if inject_ch else None

    def forward(self, x, time_emb, inject_feat=None):
        # Main path
        h = self.block1(x)
        h = h + self.time_proj(time_emb).unsqueeze(-1)
        h = self.block2(h)

        # Add residual
        h = h + self.shortcut(x)

        # Apply components
        h = self.modulation(h, time_emb)

        if inject_feat is not None:
            h = self.injection(h, inject_feat)

        if self.attention:
            h = self.attention(h)

        h = self.feedforward(h)

        return h


class SimpleAudioUNet(nn.Module):
    """Simplified Audio DiffusionTransformation U-Net"""

    def __init__(
            self,
            in_channels=1,
            out_channels=1,
            base_channels=64,
            time_dim=128,
            inject_channels=None
    ):
        super().__init__()

        # Time embedding
        self.time_emb = nn.Sequential(
        #    TimeEmbedding(time_dim // 4),
            #nn.Linear(time_dim // 4, time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        # Input
        self.input_conv = nn.Conv1d(in_channels*2, base_channels, 3, padding=1)
        self.c_conv = nn.Conv1d(88, base_channels, 3, padding=1)

        # Encoder
        self.down1 = ResNetBlock(base_channels, base_channels * 2, time_dim, inject_ch=base_channels)
        self.down2 = ResNetBlock(base_channels * 2, base_channels * 4, time_dim, use_attention=False, inject_ch=base_channels)
        self.down3 = ResNetBlock(base_channels * 4, base_channels * 8, time_dim, use_attention=False, inject_ch=base_channels)

        self.downsample = nn.Conv1d(base_channels * 8, base_channels * 8, 3, stride=2, padding=1)

        # Middle
        self.mid = ResNetBlock(base_channels * 8, base_channels * 8, time_dim, use_attention=True, inject_ch=base_channels)

        # Decoder
        self.upsample = nn.ConvTranspose1d(base_channels * 8, base_channels * 8, 4, stride=2, padding=1)

        self.up1 = ResNetBlock(base_channels * 16, base_channels * 4, time_dim, use_attention=False, inject_ch=base_channels)
        self.up2 = ResNetBlock(base_channels * 8, base_channels * 2, time_dim, use_attention=False, inject_ch=base_channels)
        self.up3 = ResNetBlock(base_channels * 4, base_channels, time_dim, inject_ch=base_channels)

        # Output
        self.output = nn.Sequential(
            nn.GroupNorm(num_groups=8, num_channels=base_channels),
            nn.SiLU(),
            nn.Conv1d(base_channels, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, noise, time, pre_frames, inject_features=None):
        # Time embedding
        t = self.time_emb(time)
        # Input
        x = torch.cat([pre_frames, noise], dim=1)

        x = self.input_conv(x)
        inject_features = self.c_conv(inject_features)

        # Encoder with skip connections
        skip1 = self.down1(x, t, inject_features)
        skip2 = self.down2(skip1, t, inject_features)
        skip3 = self.down3(skip2, t, inject_features)

        # Downsample
        x = self.downsample(skip3)

        # Middle
        x = self.mid(x, t, inject_features)

        # Upsample
        x = self.upsample(x)

        # Decoder with skip connections
        x = self.up1(torch.cat([x, skip3], dim=1), t, inject_features)
        x = self.up2(torch.cat([x, skip2], dim=1), t, inject_features)
        x = self.up3(torch.cat([x, skip1], dim=1), t, inject_features)

        # Output
        return self.output(x)
