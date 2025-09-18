import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from CheckpointManager import CheckpointManager
from utils import save_audio_files
from math import pi
from torch import Tensor
from einops import repeat, rearrange
from typing import Any, Tuple
from Embeddigns import NumberEmbedder
from U_NET_MIDI import SimpleAudioUNet
import time


def extend_dim(x: Tensor, dim: int):
    # e.g. if dim = 4: shape [b] => [b, 1, 1, 1],
    return x.view(*x.shape + (1,) * (dim - x.ndim))


# Function to save losses to file
def save_losses(train_losses, val_losses, filename='losses.json'):
    losses_dict = {
        'train_losses': train_losses,
        'val_losses': val_losses
    }
    with open(filename, 'w') as f:
        json.dump(losses_dict, f)
    print(f"Losses saved to {filename}")


# Function to load losses from file
def load_losses(filename='losses.json'):
    with open(filename, 'r') as f:
        losses_dict = json.load(f)
    return losses_dict['train_losses'], losses_dict['val_losses']


class Schedule(nn.Module):
    """Interface used by different sampling schedules"""

    def forward(self, num_steps: int, device: torch.device) -> Tensor:
        raise NotImplementedError()


class LinearSchedule(Schedule):
    def __init__(self, start: float = 1.0, end: float = 0.0):
        super().__init__()
        self.start, self.end = start, end

    def forward(self, num_steps: int, device: Any) -> Tensor:
        return torch.linspace(self.start, self.end, num_steps, device=device)


class Distribution:
    """Interface used by different distributions"""

    def __call__(self, num_samples: int, device: torch.device):
        raise NotImplementedError()


class UniformDistribution(Distribution):
    def __init__(self, vmin: float = 0.0, vmax: float = 1.0):
        super().__init__()
        self.vmin, self.vmax = vmin, vmax

    def __call__(self, num_samples: int, device: torch.device = torch.device("cpu")):
        vmax, vmin = self.vmax, self.vmin
        return (vmax - vmin) * torch.rand(num_samples, device=device) + vmin


class DiffusionModel:
    """Simple diffusion model implementation."""

    def __init__(self, model, audio_length=64, audio_size=1, noise_steps=100, beta_start=1e-4, beta_end=0.02,
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model.to(device)
        self.audio_length = audio_length
        self.audio_size = audio_size
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device
        self.encoder = NumberEmbedder(128, device=device)
        self.max_grad_norm = 1.0  # Prevent exploding gradients

    def prepare_noise_schedule(self):
        """Linear noise schedule."""
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def get_alpha_beta(self, sigmas: Tensor) -> Tuple[Tensor, Tensor]:
        angle = sigmas * pi / 2
        alpha, beta = torch.cos(angle), torch.sin(angle)
        return alpha, beta

    def noise_audios(self, x, sigmas_batch):
        # Get noise
        noise = torch.randn_like(x)
        noise /= noise.max()

        # Combine input and noise weighted by half-circle
        alphas, betas = self.get_alpha_beta(sigmas_batch)
        x_noisy = alphas * x + betas * noise
        v_target = alphas * noise - betas * x
        return x_noisy, noise, v_target

    def sample_timesteps(self, batch_size, device, dim):
        """Sample random timesteps."""
        sigmas = UniformDistribution()(num_samples=batch_size, device=device)
        sigmas_batch = extend_dim(sigmas, dim=dim)
        return sigmas, sigmas_batch

    def sample_timesteps_val(self, batch_size, device, dim):
        """Sample random timesteps."""
        sigmas = torch.Tensor([0.2, 0.4, 0.6, 0.8]).to(device)
        sigmas_batch = extend_dim(sigmas, dim=dim)
        return sigmas, sigmas_batch

    def val_step(self, optimizer, audios, criterion=nn.MSELoss()):
        """Training step for diffusion model."""
        optimizer.zero_grad()

        # Sample noise level
        sigmas_t, sigmas_batch_t = self.sample_timesteps_val(audios[1].shape[0], self.device, audios[1].ndim)

        # Add noise to the audios according to timestep
        x_t, noise, y_t = self.noise_audios(audios[1], sigmas_batch_t)

        # Predict the noise
        sigmas_t_encoded = self.encoder.to_embedding(sigmas_t)
        predicted_noise = self.model(x_t, sigmas_t_encoded, audios[0], audios[2])

        # Calculate loss and update model weights
        loss = criterion(predicted_noise, y_t)

        return loss.item()

    def train_step(self, optimizer, audios, criterion=nn.MSELoss()):
        """Training step for diffusion model."""
        optimizer.zero_grad()

        # Sample noise level
        sigmas_t, sigmas_batch_t = self.sample_timesteps(audios[1].shape[0], self.device, audios[1].ndim)

        # Add noise to the audios according to timestep
        x_t, noise, y_t = self.noise_audios(audios[1], sigmas_batch_t)

        # Predict the noise
        sigmas_t_encoded = self.encoder.to_embedding(sigmas_t)
        predicted_noise = self.model(x_t, sigmas_t_encoded, audios[0], audios[2])

        # Calculate loss and update model weights
        loss = criterion(predicted_noise, y_t)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        optimizer.step()

        return loss.item()

    @torch.no_grad()
    def sample(self, input, target, cond, num_steps):
        """Sample new audios from the diffusion model."""
        # Start with pure noise
        x_noisy = torch.randn((input.shape[0], target.shape[1], target.shape[2])).to(self.device)
        x_noisy /= x_noisy.max()
        b = x_noisy.shape[0]
        input = input.to(self.device)

        sigmas = LinearSchedule()(num_steps + 1, device=x_noisy.device)
        sigmas_batch = repeat(sigmas, "i -> i b", b=b)
        sigmas_batch = extend_dim(sigmas_batch, dim=x_noisy.ndim + 1)
        alphas, betas = self.get_alpha_beta(sigmas_batch)

        sigmas_t_encoded = self.encoder.to_embedding(sigmas)
        sigmas_t_encoded_batch = repeat(sigmas_t_encoded, "l i -> l b i", b=b)

        progress_bar = tqdm(range(num_steps))
        self.model.eval()
        # Progressively denoise the audios
        for i in progress_bar:
            v_pred = self.model(x_noisy, sigmas_t_encoded_batch[i], input, cond)
            x_pred = alphas[i] * x_noisy - betas[i] * v_pred
            noise_pred = betas[i] * x_noisy + alphas[i] * v_pred
            x_noisy = alphas[i + 1] * x_pred + betas[i + 1] * noise_pred
            progress_bar.set_description(f"Sampling (noise={sigmas[i + 1]:.2f})")

        return x_noisy


# Example usage
def test_diffusion_model(dataset_val, model_path, noise_steps):
    """Train the diffusion model on a dataset."""
    # Setup dataloader

    val_dataloader = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=True)

    base_channels, inject_channels = 128, 128
    # Initialize checkpoint manager
    ckpt_manager = CheckpointManager(model_path / "my_checkpoints")

    # Define model components
    audio_length = dataset_val.audio_chunks_outputs.shape[-1]
    model = SimpleAudioUNet(in_channels=5, out_channels=5, base_channels=base_channels, inject_channels=inject_channels)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params}")

    print('\n val batch_size', dataset_val.batch_size)
    print('\n resolution', dataset_val.resolution)
    print('\n base_channels ', base_channels)
    print('\n inject_channels ', inject_channels)
    print('\n noise_steps ', noise_steps)
    print('\n')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('cuda available :', torch.cuda.is_available())

    diffusion = DiffusionModel(model, audio_length=audio_length, audio_size=1, noise_steps=noise_steps, device=device)

    n_fft = audio_length - 1

    # Load best checkpoint
    best_checkpoint = ckpt_manager.load_best_checkpoint(diffusion, device=device)
    if best_checkpoint:
        print(f"Loaded best model with metric: {best_checkpoint.get('best_val_loss', 0)}")

    max = (48000 * 5) // (audio_length - dataset_val.stride)
    window = torch.hann_window(window_length=n_fft, device=diffusion.device)

    predictions = torch.zeros(max - 1, 5, audio_length // 2)
    targets = torch.zeros(max - 1, 5, audio_length // 2)
    # targets_images = torch.zeros(max, 5, audio_length // 2)

    input = torch.from_numpy(val_dataloader.sampler.data_source.audio_chunks_outputs[0: max - 1]).to(diffusion.device)
    cond = torch.from_numpy(val_dataloader.sampler.data_source.audio_chunks_inputs[1: max]).to(
        diffusion.device)
    target = torch.from_numpy(val_dataloader.sampler.data_source.audio_chunks_outputs[1: max]).to(
        diffusion.device)

    input_stft = torch.stft(input.squeeze(), n_fft=n_fft, win_length=n_fft, hop_length=n_fft // 4,
                            window=window, return_complex=True)  # b, f, t

    # input_stft = input_stft[None]
    input_stft = input_stft.real
    input_stft = rearrange(input_stft, 'b f t -> b t f')

    target_stft = torch.stft(target.squeeze(), n_fft=n_fft, win_length=n_fft, hop_length=n_fft // 4,
                             window=window, return_complex=True)  # b, f, t

    # target_stft = target_stft[None]
    target_image = target_stft.imag
    target_stft = target_stft.real
    target_stft = rearrange(target_stft, 'b f t -> b t f')
    target_image = rearrange(target_image, 'b f t -> b t f')

    # input_stft_ = input_stft[0:1]
    # For each audio chunk
    for n in range(max - 1):
        print('iteration n: ', n)
        print('over ', max)

        start_time = time.time()

        input_stft_, target_stft_ = visualize_samples(input_stft[n:n + 1], target_stft[n:n + 1], cond[n:n + 1], n_fft,
                                                      window, diffusion)
        # input_stft_, target_stft_ = visualize_samples(input_stft_, target_stft[n:n+1], cond[n:n+1], n_fft, window, diffusion)
        end_time = time.time()

        processing_time = end_time - start_time
        chunk_duration = audio_length / (dataset_val.fs)  # in seconds

        rtf = processing_time / chunk_duration
        print('rtf: ', rtf)

        predictions[n] = input_stft_
        targets[n] = target_stft_

    targets_images = target_image

    predictions = rearrange(predictions, 'b t f -> b f t')
    targets = rearrange(targets, 'b t f -> b f t')
    targets_images = rearrange(targets_images, 'b t f -> b f t')

    targets = targets.to(diffusion.device)
    targets_images = targets_images.to(diffusion.device)
    window = window.to(diffusion.device)
    predictions = predictions.to(diffusion.device)

    predictions = torch.istft(torch.complex(predictions, targets_images), n_fft=n_fft, hop_length=n_fft // 4,
                              win_length=n_fft, window=window, return_complex=False,
                              length=audio_length)  # Force output to match original length
    targets = torch.istft(torch.complex(targets, targets_images), n_fft=n_fft, hop_length=n_fft // 4, win_length=n_fft,
                          window=window, return_complex=False,
                          length=audio_length)  # Force output to match original length

    predictions = predictions.reshape(-1).cpu().numpy()
    targets = targets.reshape(-1).cpu().numpy()

    predictions = np.array(predictions, dtype=np.float32)
    targets = np.array(targets, dtype=np.float32)

    fig = plt.figure()
    plt.plot(predictions, alpha=0.9, label='prediction')
    plt.plot(targets, alpha=0.7, label='target')
    plt.legend()
    save_audio_files(targets, predictions, model_path, 'real_test', sample_rate=dataset_val.fs)

    fig_path = model_path / ('real_test.pdf')

    fig.savefig(fig_path, format='pdf')
    plt.close('all')

    return diffusion


def visualize_samples(inputs, targets, cond, n_fft, window, diffusion):
    """Visualize samples from the diffusion model."""
    samples = diffusion.sample(input=inputs, target=targets, cond=cond, num_steps=1)  # diffusion.noise_steps)
    # samples = rearrange(samples, 'b t f -> b f t')
    # targets = rearrange(targets, 'b t f -> b f t')

    # targets = targets.to(samples.device)
    # window = window.to(samples.device)

    # samples = torch.istft(torch.complex(samples, torch.zeros_like(samples)), n_fft=n_fft, hop_length=n_fft//4, win_length=n_fft, window=window, return_complex=False, length=n_fft+1)  # Force output to match original length
    # targets = torch.istft(torch.complex(targets, torch.zeros_like(targets)), n_fft=n_fft, hop_length=n_fft//4, win_length=n_fft, window=window, return_complex=False, length=n_fft+1)  # Force output to match original length

    return samples, targets
