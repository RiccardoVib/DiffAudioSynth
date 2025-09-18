import os
import librosa
import soundfile as sf
import scipy.ndimage
import numpy as np
import torch

import torch
import torch.nn.functional as F


def load_audio_pairs(folder, file_ext=".wav"):
    # Collect files in folder
    files = [f for f in os.listdir(folder) if f.endswith(file_ext)]

    # Group files by numeric prefix (e.g., "001_something.wav" -> "001")
    pairs = {}
    for f in files:
        # Extract numeric prefix (digits at start of filename)
        prefix = ''.join(ch for ch in f if ch.isdigit())
        if prefix:
            pairs.setdefault(prefix, []).append(f)

    # Load pairs where exactly 2 files share the prefix
    audio_pairs = []
    for prefix, files_group in pairs.items():
        if len(files_group) == 2:
            path1 = os.path.join(folder, files_group[0])
            path2 = os.path.join(folder, files_group[1])
            audio1, sr1 = librosa.load(path1, sr=None)
            audio2, sr2 = librosa.load(path2, sr=None)
            if sr1 != sr2:
                print(f"Warning: sampling rates differ in pair {prefix}")
            audio_pairs.append(((audio1, sr1), (audio2, sr2), prefix, (path1, path2)))

    return audio_pairs

class MultiResolutionSTFTLoss:
    def __init__(self, fft_sizes=None, hop_sizes=None, win_lengths=None, window='hann', device='cpu'):
        self.device = device
        self.fft_sizes = fft_sizes if fft_sizes is not None else [1024, 2048, 4096]
        self.hop_sizes = hop_sizes if hop_sizes is not None else [256, 512, 1024]
        self.win_lengths = win_lengths if win_lengths is not None else [1024, 2048, 4096]
        self.window = window
        self.windows = [torch.hann_window(win).to(device) for win in self.win_lengths]

    def stft(self, x, n_fft, hop_length, win_length, window):
        return torch.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                          window=window, return_complex=True, center=True, pad_mode='reflect')

    def spectral_convergence(self, x_mag, y_mag):
        return torch.norm(y_mag - x_mag, p='fro') / (torch.norm(y_mag, p='fro') + 1e-8)

    def magnitude_loss(self, x_mag, y_mag):
        return F.l1_loss(x_mag, y_mag)

    def __call__(self, x, y):
        """
        x, y: tensors of shape (batch, 1, time)
        """
        sc_loss = 0.0
        mag_loss = 0.0
        for i, n_fft in enumerate(self.fft_sizes):
            hop_length = self.hop_sizes[i]
            win_length = self.win_lengths[i]
            window = self.windows[i]

            X = self.stft(x.squeeze(1), n_fft, hop_length, win_length, window)
            Y = self.stft(y.squeeze(1), n_fft, hop_length, win_length, window)

            X_mag = torch.abs(X)
            Y_mag = torch.abs(Y)

            sc_loss += self.spectral_convergence(X_mag, Y_mag)
            mag_loss += self.magnitude_loss(X_mag, Y_mag)

        sc_loss /= len(self.fft_sizes)
        mag_loss /= len(self.fft_sizes)

        total_loss = sc_loss + mag_loss
        return sc_loss# total_loss

def compute_metrcis(y1, y2, sr):
    # Load audio

    min_len = min(len(y1), len(y2))
    y1, y2 = y1[:min_len], y2[:min_len]

    # L1/L2 Loss
    l1_loss = np.mean(np.abs(y1 - y2))
    l2_loss = np.mean((y1 - y2) ** 2)

    print(f"L1 Loss: {l1_loss:.6f}")
    print(f"L2 Loss: {l2_loss:.6f}")

    # Multi-Scale STFT Loss
    # x, y are torch tensors with shape (batch, 1, time)
    loss_fn = MultiResolutionSTFTLoss()
    loss_value = loss_fn(torch.from_numpy(y1[None, None]), torch.from_numpy(y2[None, None]))
    print(f"Multi-Scale STFT Loss: {loss_value.item():.6f}")


# Example usage
if __name__ == "__main__":
    import os
    import librosa

    # Example use:
    folders = ['../../TrainedModels/AudioFiles/512/',
               '../../TrainedModels/AudioFiles/1024/',
               '../../TrainedModels/AudioFiles/2048/',
               '../../TrainedModels/AudioFiles/4096/']

    for folder in folders:
        pairs = load_audio_pairs(folder)
        for (audio1, sr1), (audio2, sr2), prefix, (path1, path2) in pairs:
            print(f"Loaded pair {prefix} with sr {sr1}")
            compute_metrcis(audio1, audio2, sr1)