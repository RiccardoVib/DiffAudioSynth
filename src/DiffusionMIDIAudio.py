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

import json
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


# Function to plot losses
def plot_losses(train_losses, val_losses, filename='loss_plot.png'):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.title('Training and Validation Loss Over Time', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    # plt.show()
    print(f"Plot saved to {filename}")


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

        # """Add noise to audios according to diffusion schedule."""
        # sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None]
        # sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None]
        # eps = torch.randn_like(x)
        # return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * eps, eps

    def sample_timesteps(self, batch_size, device, dim):
        """Sample random timesteps."""
        sigmas = UniformDistribution()(num_samples=batch_size, device=device)
        sigmas_batch = extend_dim(sigmas, dim=dim)
        return sigmas, sigmas_batch
        # return torch.randint(low=1, high=self.noise_steps, size=(batch_size,))

    def sample_timesteps_val(self, batch_size, device, dim):
        """Sample random timesteps."""
        # sigmas = UniformDistribution()(num_samples=batch_size, device=device)
        sigmas = torch.Tensor([0.2, 0.4, 0.6, 0.8]).to(device)
        sigmas_batch = extend_dim(sigmas, dim=dim)
        return sigmas, sigmas_batch

    def val_step(self, optimizer, audios, criterion=nn.MSELoss()):
        """Training step for diffusion model."""
        optimizer.zero_grad()

        # Sample noise level
        # sigmas_t, sigmas_batch_t = self.sample_timesteps(audios[1].shape[0], self.device, audios[1].ndim)
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
    def sample(self, input, target, cond, num_steps, n_samples=4, return_process=False):
        """Sample new audios from the diffusion model."""
        # Start with pure noise
        # intermediate_audios = []
        x_noisy = torch.randn((input.shape[0], target.shape[1], target.shape[2])).to(self.device)
        x_noisy /= x_noisy.max()
        b = x_noisy.shape[0]
        sigmas = LinearSchedule()(num_steps + 1, device=x_noisy.device)
        sigmas_batch = repeat(sigmas, "i -> i b", b=b)
        sigmas_batch = extend_dim(sigmas_batch, dim=x_noisy.ndim + 1)
        alphas, betas = self.get_alpha_beta(sigmas_batch)

        sigmas_t_encoded = self.encoder.to_embedding(sigmas)
        sigmas_t_encoded_batch = repeat(sigmas_t_encoded, "l i -> l b i", b=b)
        # sigmas_t_encoded_batch = extend_dim(sigmas_t_encoded_batch, dim=x_noisy.ndim + 1)

        progress_bar = tqdm(range(num_steps))
        # Progressively denoise the audios
        # for i in tqdm(reversed(range(1, num_steps)), desc="Sampling...", total=self.noise_steps - 1):
        for i in progress_bar:
            v_pred = self.model(x_noisy, sigmas_t_encoded_batch[i], input, cond)
            x_pred = alphas[i] * x_noisy - betas[i] * v_pred
            noise_pred = betas[i] * x_noisy + alphas[i] * v_pred
            x_noisy = alphas[i + 1] * x_pred + betas[i + 1] * noise_pred
            progress_bar.set_description(f"Sampling (noise={sigmas[i + 1]:.2f})")
            # if return_process:
            #   intermediate_audios.append(x_noisy.cpu())
        return x_noisy


# Example usage
def train_diffusion_model(dataset, dataset_val, model_path, noise_steps, epochs=10, lr=1e-4):
    """Train the diffusion model on a dataset."""
    # Setup dataloader
    if dataset is not None:
        train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    else:
        train_dataloader = None
    val_dataloader = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=True)

    # cqt = CQT(
    #     num_octaves=8,
    #     num_bins_per_octave=64,
    #     sample_rate=48000,
    #     block_length=100 # input.shape[-1] // 120
    # )
    base_channels, inject_channels = 128, 128
    lr_count = 0

    # Initialize checkpoint manager
    ckpt_manager = CheckpointManager(model_path / "my_checkpoints")

    # Define model components
    audio_length = dataset_val.audio_chunks_outputs.shape[-1]
    model = SimpleAudioUNet(in_channels=5, out_channels=5, base_channels=base_channels, inject_channels=inject_channels)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params}")

    if dataset is not None:
        print('\n train batch_size', dataset.batch_size)

    print('\n val batch_size', dataset_val.batch_size)
    print('\n resolution', dataset_val.resolution)
    print('\n base_channels ', base_channels)
    print('\n inject_channels ', inject_channels)
    print('\n noise_steps ', noise_steps)
    print('\n epochs ', epochs)
    print('\n')
    # loss_fn = ElementwiseNormalizedMSELoss()
    # loss_fn = audio_autoencoder_loss

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('cuda available :', torch.cuda.is_available())

    diffusion = DiffusionModel(model, audio_length=audio_length, audio_size=1, noise_steps=noise_steps, device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=1e-2
    )
    # Define the scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )

    # Load last checkpoint
    checkpoint = ckpt_manager.load_last_checkpoint(diffusion, optimizer, device=device)

    if checkpoint:
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}")
        best_loss = checkpoint['best_val_loss']
        print(f"Loaded best model with metric: {best_loss}")

    else:
        print("Starting training from scratch")
        best_loss = float('inf')

    n_fft = audio_length
    train_losses, val_losses = [], []
    # Training loop
    epoch = 0
    for epoch in range(epochs):
        train_batches = 0
        train_loss, val_loss = 0, 0
        model.train()
        for input, target, cond in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
            input = input[0].to(diffusion.device)
            target = target[0].to(diffusion.device)
            cond = cond[0].to(diffusion.device)
            window = torch.hann_window(window_length=n_fft, device=target.device)

            input_stft = torch.stft(input.squeeze(), n_fft=n_fft, win_length=n_fft, hop_length=n_fft // 4,
                                    window=window, return_complex=True).real  # b, f, t
            input_stft = rearrange(input_stft, 'b f t -> b t f')
            target_stft = torch.stft(target.squeeze(), n_fft=n_fft, win_length=n_fft, hop_length=n_fft // 4,
                                     window=window, return_complex=True).real  # b, f, t
            target_stft = rearrange(target_stft, 'b f t -> b t f')

            loss = diffusion.train_step(optimizer=optimizer,
                                        audios=[input_stft, target_stft, cond])  # , criterion=loss_fn)
            train_loss += loss
            train_batches += 1
            scheduler.step()

        avg_train_loss = train_loss / train_batches
        train_losses.append(avg_train_loss)

        # Validation phase
        if (epoch + 1) % 1 == 0:
            total_val_loss = 0
            val_batches = 0
            model.eval()
            with torch.no_grad():
                for input, target, cond in tqdm(val_dataloader, desc=f"Validation Epoch {epoch + 1}"):
                    input = input[0].to(diffusion.device)
                    target = target[0].to(diffusion.device)
                    cond = cond[0].to(diffusion.device)

                    input_stft = torch.stft(input.squeeze(), n_fft=n_fft, win_length=n_fft, hop_length=n_fft // 4,
                                            window=window, return_complex=True).real  # b, f, t
                    input_stft = rearrange(input_stft, 'b f t -> b t f')
                    target_stft = torch.stft(target.squeeze(), n_fft=n_fft, win_length=n_fft, hop_length=n_fft // 4,
                                             window=window, return_complex=True).real  # b, f, t
                    target_stft = rearrange(target_stft, 'b f t -> b t f')

                    val_loss = diffusion.val_step(optimizer=optimizer,
                                                  audios=[input_stft, target_stft, cond])  # , criterion=loss_fn)

                    total_val_loss += val_loss
                    val_batches += 1

            avg_val_loss = total_val_loss / val_batches
            val_losses.append(avg_val_loss)

            # Update learning rate scheduler with validation loss
            scheduler.step()
            print(f'Epoch {epoch + 1}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
            print(f'Learning Rate {optimizer.param_groups[0]["lr"]:.2e}')

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            # Save best checkpoint (assuming this is the best model so far)
            state_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'encoder_state_dict': diffusion.encoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'best_val_loss': best_loss
            }
            ckpt_manager.save_checkpoint(state_dict, is_best=True)
            print(f"Epoch {epoch + 1}, Validation loss improved: ", best_loss)
            lr_count = 0
        else:
            lr_count += 1
            # print(f"Epoch {epoch + 1}, Validation loss did not improved. Best val loss: ", best_loss)
            if lr_count == 500:
                print(f'No improvements over 500 epochs -> stopping...')
                break
        # Save latest checkpoint
        state_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'encoder_state_dict': diffusion.encoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'best_val_loss': best_loss
        }

        #print(diffusion.encoder.state_dict())
        # Save last checkpoint
        ckpt_manager.save_last_checkpoint(state_dict)

        # Generate and visualize samples
        if (epoch + 1) % 20 == 0:
            visualize_samples(input_stft, target_stft, cond, n_fft, window, diffusion, model_path, 'val', epoch + 1,
                              dataset_val.fs)
            filename = model_path / ('loss_plot_' + str(epoch) + '.png')
            plot_losses(train_losses=train_losses, val_losses=val_losses, filename=filename)

    filename = model_path / ('losses' + str(epoch) + '.json')
    save_losses(train_losses=train_losses, val_losses=val_losses, filename=filename)
    filename = model_path / ('loss_plot' + str(epoch) + '.png')
    plot_losses(train_losses=train_losses, val_losses=val_losses, filename=filename)

    # Load best checkpoint
    best_checkpoint = ckpt_manager.load_best_checkpoint(diffusion, device=device)
    if best_checkpoint:
        print(f"Loaded best model with metric: {best_checkpoint.get('best_val_loss', 0)}")

    inputs, targets, conds = [], [], []
    # Visualize the diffusion process
    for input, target, cond in tqdm(val_dataloader):
        input = input[0].to(diffusion.device)
        target = target[0].to(diffusion.device)
        cond = cond[0].to(diffusion.device)
        window = torch.hann_window(window_length=n_fft, device=target.device)

        input_stft = torch.stft(input.squeeze(), n_fft=n_fft, win_length=n_fft, hop_length=n_fft // 4,
                                window=window, return_complex=True).real  # b, f, t
        input_stft = rearrange(input_stft, 'b f t -> b t f')

        target_stft = torch.stft(target.squeeze(), n_fft=n_fft, win_length=n_fft, hop_length=n_fft // 4,
                                 window=window, return_complex=True).real  # b, f, t
        target_stft = rearrange(target_stft, 'b f t -> b t f')

        inputs.append(input_stft)
        targets.append(target_stft)
        conds.append(cond)

    visualize_samples(input_stft, target_stft, cond, n_fft, window, diffusion, model_path, 'test', epoch + 1,
                      dataset_val.fs)

    # visualize_diffusion_process(diffusion, inputs[:4], targets[:4], conds[:4], n_fft, window, model_path, diffusion.noise_steps, dataset_val.fs)

    return diffusion


def visualize_samples(inputs, targets, cond, n_fft, window, diffusion, model_path, filename, epoch=None, fs=48000):
    """Visualize samples from the diffusion model."""
    samples = diffusion.sample(input=inputs, target=targets, cond=cond, n_samples=inputs[0].shape[0],
                               num_steps=diffusion.noise_steps, return_process=False)
    samples = rearrange(samples, 'b t f -> b f t')
    targets = rearrange(targets, 'b t f -> b f t')

    samples = torch.istft(torch.complex(samples, torch.zeros_like(samples)), n_fft=n_fft, hop_length=n_fft // 4,
                          win_length=n_fft, window=window, return_complex=False, length=n_fft+1)
    targets = torch.istft(torch.complex(targets, torch.zeros_like(targets)), n_fft=n_fft, hop_length=n_fft // 4,
                          win_length=n_fft, window=window, return_complex=False, length=n_fft+1)

    fig = plt.figure()
    for i, (input, sample, target) in enumerate(zip(inputs, samples, targets)):
        plt.plot(sample.cpu().squeeze(), alpha=0.9, label='prediction')
        plt.plot(target.cpu().squeeze(), alpha=0.7, label='target')
        plt.legend(())
        save_audio_files(target.cpu(), sample.cpu(), model_path, filename + str(i), sample_rate=fs)

        fig_path = model_path / (filename + str(i) + '.pdf')

        fig.savefig(fig_path, format='pdf')
        plt.close('all')

    if epoch:
        plt.suptitle(f"Samples at Epoch {epoch}")


def visualize_diffusion_process(diffusion, inputs, targets, conds, n_fft, window, model_path, num_inference_steps=50,
                                fs=48000):
    """Visualize the diffusion process."""

    for j, (target, input, cond) in enumerate(zip(targets, inputs, conds)):
        intermediate_audios = diffusion.sample(input=input, target=target, cond=cond, num_steps=num_inference_steps,
                                               n_samples=input.shape[0], return_process=True)
        last_audios = intermediate_audios

        last_audios = rearrange(last_audios, 'b t f -> b f t')
        target = rearrange(target, 'b t f -> b f t')
        last_audios = torch.istft(torch.complex(last_audios, torch.zeros_like(last_audios)), n_fft=n_fft,
                                  hop_length=n_fft // 4, win_length=n_fft, window=window,
                                  return_complex=False, length=n_fft+1)

        target = torch.istft(torch.complex(target, torch.zeros_like(target)), n_fft=n_fft, hop_length=n_fft // 4,
                             win_length=n_fft, window=window,
                             return_complex=False, length=n_fft+1)

        fig = plt.figure()
        for i, (audio, tar, inp) in enumerate(zip(last_audios, target, input)):
            plt.plot(audio.cpu().squeeze(), alpha=0.9, label='prediction')
            plt.plot(tar.cpu().squeeze(), alpha=0.7, label='target')
            plt.legend()

            fig_path = model_path / ('inference' + str(j + i) + '.pdf')

            save_audio_files(tar.cpu(), audio.cpu(), model_path, 'inference' + str(j + i), sample_rate=fs)

            plt.suptitle("DiffusionTransformation Process: Noise → Audio")
            plt.tight_layout()
            fig.savefig(fig_path, format='pdf')
            plt.close('all')
