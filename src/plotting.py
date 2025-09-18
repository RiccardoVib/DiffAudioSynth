import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple


def visualize_midi_tensor(midi_tensor: torch.Tensor,
                          fs: int = 48000,
                          note_range: Tuple[int, int] = (21, 108),
                          figsize: Tuple[int, int] = (15, 8),
                          max_time: Optional[float] = None,
                          colormap: str = 'viridis',
                          show_note_names: bool = True,
                          title: str = "MIDI Tensor Visualization") -> plt.Figure:
    """
    Visualize a MIDI tensor as a piano roll.

    Args:
        midi_tensor: Tensor of shape (time_steps, num_notes)
        fs: Sample rate used to create the tensor
        note_range: (min_note, max_note) MIDI note numbers
        figsize: Figure size (width, height)
        max_time: Maximum time to display in seconds (None for full length)
        colormap: Matplotlib colormap name
        show_note_names: Whether to show note names on y-axis
        title: Plot title

    Returns:
        matplotlib Figure object
    """
    # Convert tensor to numpy if needed
    if isinstance(midi_tensor, torch.Tensor):
        midi_data = midi_tensor.numpy()
    else:
        midi_data = midi_tensor

    # Get dimensions
    time_steps, num_notes = midi_data.shape
    min_note, max_note = note_range

    # Create time axis
    time_axis = np.linspace(0, time_steps / fs, time_steps)

    # Limit time if specified
    if max_time is not None:
        max_samples = int(max_time * fs)
        if max_samples < time_steps:
            midi_data = midi_data[:max_samples]
            time_axis = time_axis[:max_samples]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create the piano roll visualization
    im = ax.imshow(midi_data.T,
                   aspect='auto',
                   origin='lower',
                   extent=[time_axis[0], time_axis[-1], min_note, max_note],
                   cmap=colormap,
                   interpolation='nearest')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Velocity', rotation=270, labelpad=15)

    # Set labels and title
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('MIDI Note Number')
    ax.set_title(title)

    # Optionally add note names
    if show_note_names and num_notes <= 88:  # Only for reasonable number of notes
        note_names = []
        note_labels = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

        # Create note name labels at octave boundaries
        octave_notes = []
        octave_positions = []

        for note_num in range(min_note, max_note + 1, 12):  # Every octave
            if note_num <= max_note:
                octave = (note_num // 12) - 1
                note_name = note_labels[note_num % 12]
                octave_notes.append(f'{note_name}{octave}')
                octave_positions.append(note_num)

        ax.set_yticks(octave_positions)
        ax.set_yticklabels(octave_notes)

    # Add grid for better readability
    ax.grid(True, alpha=0.3)

    # Tight layout
    plt.tight_layout()

    return fig


def visualize_midi_comparison(midi_tensor1: torch.Tensor,
                              midi_tensor2: torch.Tensor,
                              fs: int = 48000,
                              note_range: Tuple[int, int] = (21, 108),
                              figsize: Tuple[int, int] = (15, 10),
                              max_time: Optional[float] = None,
                              titles: Tuple[str, str] = ("MIDI 1", "MIDI 2")) -> plt.Figure:
    """
    Compare two MIDI tensors side by side.

    Args:
        midi_tensor1, midi_tensor2: Tensors to compare
        fs: Sample rate
        note_range: MIDI note range
        figsize: Figure size
        max_time: Maximum time to display
        titles: Titles for the two plots

    Returns:
        matplotlib Figure object
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    for i, (tensor, ax, title) in enumerate([(midi_tensor1, ax1, titles[0]),
                                             (midi_tensor2, ax2, titles[1])]):
        # Convert to numpy if needed
        if isinstance(tensor, torch.Tensor):
            midi_data = tensor.numpy()
        else:
            midi_data = tensor

        time_steps = midi_data.shape[0]
        min_note, max_note = note_range

        # Create time axis
        time_axis = np.linspace(0, time_steps / fs, time_steps)

        # Limit time if specified
        if max_time is not None:
            max_samples = int(max_time * fs)
            if max_samples < time_steps:
                midi_data = midi_data[:max_samples]
                time_axis = time_axis[:max_samples]

        # Plot
        im = ax.imshow(midi_data.T,
                       aspect='auto',
                       origin='lower',
                       extent=[time_axis[0], time_axis[-1], min_note, max_note],
                       cmap='viridis',
                       interpolation='nearest')

        ax.set_ylabel('MIDI Note')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Velocity', rotation=270, labelpad=15)

    ax2.set_xlabel('Time (seconds)')
    plt.tight_layout()

    return fig


def visualize_midi_notes_over_time(midi_tensor: torch.Tensor,
                                   fs: int = 48000,
                                   note_range: Tuple[int, int] = (21, 108),
                                   figsize: Tuple[int, int] = (15, 6),
                                   max_time: Optional[float] = None,
                                   threshold: float = 0.1) -> plt.Figure:
    """
    Visualize MIDI tensor as individual note lines over time.

    Args:
        midi_tensor: MIDI tensor
        fs: Sample rate
        note_range: MIDI note range
        figsize: Figure size
        max_time: Maximum time to display
        threshold: Minimum velocity to display a note

    Returns:
        matplotlib Figure object
    """
    # Convert to numpy if needed
    if isinstance(midi_tensor, torch.Tensor):
        midi_data = midi_tensor.numpy()
    else:
        midi_data = midi_tensor

    time_steps = midi_data.shape[0]
    min_note, max_note = note_range

    # Create time axis
    time_axis = np.linspace(0, time_steps / fs, time_steps)

    # Limit time if specified
    if max_time is not None:
        max_samples = int(max_time * fs)
        if max_samples < time_steps:
            midi_data = midi_data[:max_samples]
            time_axis = time_axis[:max_samples]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot each note that has activity above threshold
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    color_idx = 0

    active_notes = []
    for note_idx in range(midi_data.shape[1]):
        note_num = min_note + note_idx
        note_data = midi_data[:, note_idx]

        if np.max(note_data) > threshold:
            # Find note onset and offset times
            active_times = time_axis[note_data > threshold]
            active_velocities = note_data[note_data > threshold]

            if len(active_times) > 0:
                ax.scatter(active_times, [note_num] * len(active_times),
                           s=active_velocities * 100,
                           c=[colors[color_idx % len(colors)]],
                           alpha=0.7,
                           label=f'Note {note_num}')
                active_notes.append(note_num)
                color_idx += 1

    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('MIDI Note Number')
    ax.set_title('MIDI Notes Over Time')
    ax.grid(True, alpha=0.3)

    # Show legend only if reasonable number of notes
    if len(active_notes) <= 20:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    return fig