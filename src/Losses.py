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

def audio_autoencoder_loss(original, reconstructed):
    # Spectral loss (primary)
    spec_loss = spectral_loss(original, reconstructed)

    # Time domain loss (secondary)
    time_loss = F.l1_loss(original, reconstructed)  # L1 often better than MSE

    # Optional: perceptual loss using pre-trained features
    # perceptual_loss = some_pretrained_loss(original, reconstructed)

    return spec_loss + 0.1 * time_loss

def spectral_loss(y_true, y_pred):
    loss = 0
    # Multiple FFT window sizes
    for n_fft in [512, 1024, 2048]:
        # Spectral convergence loss
        stft_true = torch.stft(y_true, n_fft=n_fft, return_complex=True)
        stft_pred = torch.stft(y_pred, n_fft=n_fft, return_complex=True)

        sc_loss = torch.norm(stft_true - stft_pred, 'fro') / torch.norm(stft_true, 'fro')

        # Log magnitude loss
        mag_true = torch.abs(stft_true)
        mag_pred = torch.abs(stft_pred)
        logmag_loss = F.l1_loss(torch.log(mag_true + 1e-7), torch.log(mag_pred + 1e-7))

        loss += sc_loss + logmag_loss

    return loss

class SpectralLoss(nn.Module):
    """
    
    """

    def __init__(self, eps=1e-8):
        super(SpectralLoss, self).__init__()
        self.eps = eps

    def forward(self, pred, target):
        loss = 0
        # Multiple FFT window sizes
        for n_fft in [512, 1024, 2048]:
            # Spectral convergence loss
            stft_true = torch.stft(y_true, n_fft=n_fft, return_complex=True)
            stft_pred = torch.stft(y_pred, n_fft=n_fft, return_complex=True)

            sc_loss = torch.norm(stft_true - stft_pred, 'fro') / torch.norm(stft_true, 'fro')

            # Log magnitude loss
            mag_true = torch.abs(stft_true)
            mag_pred = torch.abs(stft_pred)
            logmag_loss = F.l1_loss(torch.log(mag_true + 1e-7), torch.log(mag_pred + 1e-7))

            loss += sc_loss + logmag_loss

        return loss


class NormalizedMSELoss(nn.Module):
    """
    Custom MSE loss normalized by the target values.

    Loss = MSE(pred, target) / |target|^2
    where MSE = mean((pred - target)^2)
    """

    def __init__(self, eps=1e-8):
        super(NormalizedMSELoss, self).__init__()
        self.eps = eps  # Small epsilon to avoid division by zero

    def forward(self, pred, target):
        # Compute standard MSE
        mse = torch.mean((pred - target) ** 2)

        # Normalize by target magnitude squared
        target_norm_sq = torch.mean(target ** 2)

        # Add epsilon to avoid division by zero
        normalized_mse = mse / (target_norm_sq + self.eps)

        return normalized_mse


# Alternative implementation with element-wise normalization
class ElementwiseNormalizedMSELoss(nn.Module):
    """
    Element-wise normalized MSE loss where each squared error is
    normalized by the corresponding target value squared.

    Loss = mean((pred - target)^2 / (target^2 + eps))
    """

    def __init__(self, eps=1e-8):
        super(ElementwiseNormalizedMSELoss, self).__init__()
        self.eps = eps

    def forward(self, pred, target):
        squared_errors = (pred - target) ** 2
        target_sq = target ** 2 + self.eps
        normalized_errors = squared_errors / target_sq
        return torch.mean(normalized_errors)


# Functional versions
def normalized_mse_loss(pred, target, eps=1e-8):
    """Functional version of normalized MSE loss"""
    mse = torch.mean((pred - target) ** 2)
    target_norm_sq = torch.mean(target ** 2)
    return mse / (target_norm_sq + eps)


def elementwise_normalized_mse_loss(pred, target, eps=1e-8):
    """Functional version of element-wise normalized MSE loss"""
    squared_errors = (pred - target) ** 2
    target_sq = target ** 2 + eps
    return torch.mean(squared_errors / target_sq)
