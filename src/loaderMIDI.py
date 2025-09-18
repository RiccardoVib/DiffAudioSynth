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

import pickle
import sys
import gzip
import torch
import torchaudio
import numpy as np
import os
import glob
from utils import natural_sort_key
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import pretty_midi
from typing import Tuple
import librosa.display

def plot_piano_roll(piano_roll, start_pitch, fs=100):
    librosa.display.specshow(piano_roll, hop_length=1, sr=fs, x_axis='time', y_axis='cqt_note',
                             fmin=pretty_midi.note_number_to_hz(start_pitch))

def audio_converter(audio):
    """Convert audio to range -1 to +1 based on bit depth."""
    if audio.dtype == torch.int16:
        return audio.to(torch.float32) / 32768.0
    elif audio.dtype == torch.int32:
        return audio.to(torch.float32) / 2147483648.0
    elif audio.dtype == torch.float32 or audio.dtype == torch.float64:
        return audio.to(torch.float32)
    else:
        raise ValueError(f"Unsupported audio data type: {audio.dtype}")

def midi_to_pianoroll(midi_file: str, fs: int = 48000, note_range: Tuple[int, int] = (21, 108)) -> torch.Tensor:
    """
    Convert MIDI file to piano roll representation.

    Args:
        midi_file: Path to MIDI file
        fs: Sample rate for the piano roll
        note_range: (min_note, max_note) MIDI note numbers to include

    Returns:
        Piano roll tensor of shape (time_steps, num_notes)
    """
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_file)
    except:
        raise ValueError(f"Could not load MIDI file: {midi_file}")
    min_note, max_note = note_range
    piano_roll = midi_data.get_piano_roll(fs=fs)[min_note:max_note+1]/127
    # plot_piano_roll(piano_roll[:, :12000], 21)
    # plt.show()
    return torch.tensor(piano_roll.T, dtype=torch.float32)

def _midi_to_pianoroll(midi_file: str, fs: int = 48000, note_range: Tuple[int, int] = (21, 108)) -> torch.Tensor:
    """
    Convert MIDI file to piano roll representation.

    Args:
        midi_file: Path to MIDI file
        fs: Sample rate for the piano roll
        note_range: (min_note, max_note) MIDI note numbers to include

    Returns:
        Piano roll tensor of shape (time_steps, num_notes)
    """
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_file)
    except:
        raise ValueError(f"Could not load MIDI file: {midi_file}")

    # Get the total duration
    total_time = midi_data.get_end_time()

    # Calculate time steps based on sample rate
    time_steps = int(total_time * fs)

    # Note range
    min_note, max_note = note_range
    num_notes = max_note - min_note + 1

    # Initialize piano roll
    piano_roll = np.zeros((time_steps, num_notes))

    # Fill piano roll for each instrument
    for instrument in midi_data.instruments:
        if instrument.is_drum:
            continue  # Skip drum tracks for now

        for note in instrument.notes:
            if min_note <= note.pitch <= max_note:
                start_time = int(note.start * fs)
                end_time = int(note.end * fs)
                note_idx = note.pitch - min_note

                # Ensure we don't exceed array bounds
                start_time = max(0, min(start_time, time_steps - 1))
                end_time = max(0, min(end_time, time_steps))

                # Set velocity (normalized to 0-1)
                velocity = note.velocity / 127.0
                piano_roll[start_time:end_time, note_idx] = velocity

    return torch.tensor(piano_roll, dtype=torch.float32)


def midi_to_onset_frames(midi_file: str, fs: int = 48000, note_range: Tuple[int, int] = (21, 108)) -> torch.Tensor:
    """
    Convert MIDI file to onset representation (only note onsets, not duration).

    Args:
        midi_file: Path to MIDI file
        fs: Sample rate
        note_range: (min_note, max_note) MIDI note numbers to include

    Returns:
        Onset tensor of shape (time_steps, num_notes)
    """
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_file)
    except:
        raise ValueError(f"Could not load MIDI file: {midi_file}")

    total_time = midi_data.get_end_time()
    time_steps = int(total_time * fs)

    min_note, max_note = note_range
    num_notes = max_note - min_note + 1

    onset_roll = np.zeros((time_steps, num_notes))

    for instrument in midi_data.instruments:
        if instrument.is_drum:
            continue

        for note in instrument.notes:
            if min_note <= note.pitch <= max_note:
                onset_time = int(note.start * fs)
                note_idx = note.pitch - min_note

                if 0 <= onset_time < time_steps:
                    velocity = note.velocity / 127.0
                    onset_roll[onset_time, note_idx] = velocity

    return torch.tensor(onset_roll, dtype=torch.float32)


class MidiAudioDataset(Dataset):
    def __init__(self,
                 root_dir_midi: str,
                 root_dir_audio: str,
                 dataset_path: str,
                 batch_size: int,
                 resolution: int,
                 output_length: int,
                 input_dimension: int,
                 output_dimension: int,
                 stride: int,
                 fs: int,
                 save_dataset: bool,
                 filename: str,
                 load_dataset: bool,
                 shuffle: bool,
                 all_in_memory: bool,
                 mono: bool,
                 data_type: type = torch.float32,
                 midi_representation: str = 'pianoroll',  # 'pianoroll' or 'onset'
                 note_range: Tuple[int, int] = (21, 108),  # MIDI note range
                 ):
        """
        Initializes a MIDI-to-Audio data loader object

        New parameters:
        :param root_dir_midi: the directory in which MIDI input data are stored [string]
        :param root_dir_audio: the directory in which target audio data are stored [string]
        :param midi_representation: 'pianoroll' or 'onset' - how to represent MIDI data
        :param note_range: (min_note, max_note) tuple for MIDI note range
        """
        self.root_dir_midi = root_dir_midi
        self.root_dir_audio = root_dir_audio
        self.dataset_path = dataset_path
        self.filename = filename
        self.batch_size = batch_size
        self.resolution = resolution
        self.output_length = output_length
        self.stride = stride
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.fs = fs
        self.data_type = data_type
        self.shuffle = shuffle
        self.load_dataset = load_dataset
        self.save_dataset = save_dataset
        self.all_in_memory = all_in_memory
        self.mono = mono
        self.midi_representation = midi_representation
        self.note_range = note_range
        self.lim = -1
        self.input_length = self.output_length // self.resolution

        # Additional variables
        self.indices = None
        self.num_samples = 0
        self.num_frames = 0

        # get file paths
        self.midi_files = glob.glob(os.path.join(self.root_dir_midi, "*.mid")) + \
                          glob.glob(os.path.join(self.root_dir_midi, "*.midi"))
        self.audio_files = glob.glob(os.path.join(self.root_dir_audio, "*.wav"))

        # ensure that the sets are ordered correctly
        self.midi_files = sorted(self.midi_files, key=natural_sort_key)
        self.audio_files = sorted(self.audio_files, key=natural_sort_key)

        self.audio_chunks = []
        self.audio_item_target = []
        self.midi_item_input = []
        self.audio_chunks_inputs, self.audio_chunks_outputs = [], []

        # get the already prepared dataset or load the files
        self._load_dataset_or_files()

        # prepare the chunks
        if self.load_dataset is False:
            if self.all_in_memory:
                self._prepare_dataset_and_save()
            else:
                self.idx = 0
                self._prepare_next_file()

        if self.all_in_memory:
            self._prepare_matrix()

        self.__on_epoch_end__()

    def print(self):
        print("\nDataset:")
        print(f"num_examples: {len(self.audio_chunks_total)}")
        print(f"sample_length: {self.output_length}")
        print(f"num_frames: {self.num_frames}")
        print(f"num_minutes: {(self.output_length / self.fs) * self.num_frames}")

    def _load_audio(self, filepath, frame_offset=0):
        """Load audio file"""
        audio, sr = torchaudio.load(filepath, channels_first=False)
        audio = (audio[frame_offset:])

        if len(audio.shape) == 1:
            audio = audio[:, np.newaxis]

        if self.mono and audio.shape[1] > 1:
            audio = torch.mean(audio, dim=-1, keepdim=True)

        if audio.dtype is not self.data_type:
            audio = audio_converter(audio)

        if sr != self.fs:
            audio = torchaudio.functional.resample(audio.T, sr, self.fs).T

        return audio, sr

    def _load_midi(self, filepath):
        """Load and convert MIDI file to tensor representation"""
        if self.midi_representation == 'pianoroll':
            midi_tensor = midi_to_pianoroll(filepath, fs=self.fs//self.resolution, note_range=self.note_range)
        elif self.midi_representation == 'onset':
            midi_tensor = midi_to_onset_frames(filepath, fs=self.fs//self.resolution, note_range=self.note_range)
        else:
            raise ValueError(f"Unknown MIDI representation: {self.midi_representation}")

        return midi_tensor

    def select_audio_by_time_range(self, audio, time_delimiters, percentage):
        """
        Selects a percentage of audio from each input/target pair within specified time ranges.
        """
        if percentage <= 0 or percentage > 1.0:
            raise ValueError("Percentage must be between 0 and 1")

        selected_audio = self._select_audio_segments(
            audio,
            time_delimiters,
            percentage
        )
        return selected_audio

    def _select_audio_segments(self, audio, time_delimiters, percentage):
        """Helper method to select audio segments based on time delimiters."""
        selected_segments = []

        for start_time, end_time in time_delimiters:
            start_sample = int(start_time * self.fs)
            end_sample = int(end_time * self.fs)

            start_sample = max(0, start_sample)
            end_sample = min(audio.shape[0], end_sample)

            if start_sample >= end_sample:
                continue

            segment_length = end_sample - start_sample
            samples_to_take = int(segment_length * percentage)

            if samples_to_take <= 0:
                continue

            start_offset = start_sample + (segment_length - samples_to_take) // 2
            selected_segment = audio[start_offset:start_offset + samples_to_take]
            selected_segments.append(selected_segment)

        if not selected_segments:
            return torch.empty((0,) + tuple(audio.shape[1:]), dtype=audio.dtype, device=audio.device)

        return torch.cat(selected_segments, dim=0)

    def _load_dataset_or_files(self):
        """Load a prepared dataset or the MIDI/audio files."""

        # Check if dataset already exists and should be loaded
        if self.load_dataset:
            dataset_path = os.path.join(self.dataset_path, )
            if os.path.exists(dataset_path):
                print(f"Dataset found at {dataset_path}. Skipping preparation.")
                with gzip.open(os.path.join(dataset_path, self.filename + ".pkl.gz"), 'rb') as file:
                    audio_chunks = pickle.load(file)
                    self.audio_chunks_inputs = np.array(audio_chunks['audio_chunks_inputs'], dtype=np.float32)
                    self.audio_chunks_outputs = np.array(audio_chunks['audio_chunks_outputs'], dtype=np.float32)
                    self.audio_chunks_params = np.array(audio_chunks['audio_chunks_params'], dtype=np.float32)
                    self.num_samples = audio_chunks['num_samples']
        else:
            self.min_min_frames = float('inf')
            # loop over files
            for idx, (midi_file, audio_file) in enumerate(zip(self.midi_files, self.audio_files)):
                print('MIDI input:', midi_file, '- Audio target:', audio_file)

                # Check if files match (optional - you might want different matching logic)
                midi_basename = os.path.splitext(os.path.basename(midi_file))[0]
                audio_basename = os.path.splitext(os.path.basename(audio_file))[0]

                if midi_basename != audio_basename:
                    print(f"Warning: {midi_basename} and {audio_basename} have different names")

                # Get audio metadata
                md = torchaudio.info(audio_file)
                num_frames = md.num_frames
                self.num_frames += num_frames

                sys.stdout.write(f"* Pre-loading... {idx + 1:3d}/{len(self.audio_files):3d} ...\r")
                sys.stdout.flush()

                # Load audio target
                target_audio, sr = self._load_audio(audio_file)

                # Load MIDI input
                try:
                    midi_input = self._load_midi(midi_file)
                except Exception as e:
                    print(f"Error loading MIDI file {midi_file}: {e}")
                    continue


                # Ensure MIDI and audio have compatible lengths
                # Truncate to shorter length
                min_frames = min(midi_input.shape[0]*self.resolution, target_audio.shape[0])
                if midi_input.shape[0]*self.resolution != target_audio.shape[0]:
                    print(
                        f"Length mismatch: MIDI {midi_input.shape[0]}, Audio {target_audio.shape[0]}. Truncating to {min_frames}")

                if min_frames < self.min_min_frames:
                    self.min_min_frames = min_frames

                # Store the processed data
                self.audio_item_target.append({
                    'file': audio_file,
                    'audio': target_audio[:min_frames],
                    'num_frames': min_frames,
                })

                self.midi_item_input.append({
                    'file': midi_file,
                    'midi': midi_input[:min_frames//self.resolution],
                    'num_frames': min_frames//self.resolution,
                })

    def _prepare_dataset_and_save(self):
        """Prepare the MIDI-audio chunks and optionally save."""

        if self.output_length != -1:
            max_frames = 0
            for audio_item in self.audio_item_target:
                num_frames_for_file = (audio_item['num_frames'] - self.output_length) // (
                            self.output_length - self.stride) + 1
                max_frames = max(max_frames, max(0, num_frames_for_file))

            # MIDI input dimensions
            midi_features = self.midi_item_input[0]['midi'].shape[-1]  # Number of MIDI features (notes)

            self.audio_chunks_inputs = np.zeros(
                (len(self.audio_item_target), max_frames, midi_features, self.input_length))
            self.audio_chunks_outputs = np.zeros(
                (len(self.audio_item_target), max_frames, self.output_length))

        else:
            midi_features = self.midi_item_input[0]['midi'].shape[-1]
            self.audio_chunks_inputs = np.zeros(
                (len(self.audio_item_target), midi_features, len(self.midi_item_input[0]['midi'])))
            self.audio_chunks_outputs = np.zeros(
                (len(self.audio_item_target), 1, len(self.audio_item_target[0]['audio'])))

        # loop over files
        for idx, (midi_item, audio_item) in enumerate(zip(self.midi_item_input, self.audio_item_target)):
            if self.output_length == -1:  # take whole file
                self.audio_chunks_inputs[idx, 0] = midi_item['midi'].numpy()
                self.audio_chunks_outputs[idx, 0] = audio_item['audio'].numpy()
                self.num_samples = len(self.audio_item_target)

            else:  # split into chunks
                for n in range(max_frames):
                    hop = int(n * (self.output_length - self.stride))
                    end = hop + self.output_length

                    # Audio output chunk
                    self.audio_chunks_outputs[idx, n:n+1, :] = audio_item['audio'][hop:end, :].numpy().T

                    # MIDI input chunk
                    self.audio_chunks_inputs[idx, n:n+1, :] = midi_item['midi'][hop//self.resolution:end//self.resolution, :].numpy().T

                self.num_samples = max_frames * len(self.audio_item_target)

        self.audio_chunks_inputs = self.audio_chunks_inputs.astype(np.float32)
        self.audio_chunks_outputs = self.audio_chunks_outputs.astype(np.float32)

        # Save dataset if needed
        if self.save_dataset and self.dataset_path is not None:
            data_dict = {
                'audio_chunks_inputs': self.audio_chunks_inputs,
                'audio_chunks_outputs': self.audio_chunks_outputs,
                'num_samples': self.num_samples
            }

            path = os.path.join(self.dataset_path)
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

            with gzip.open(os.path.join(path, self.filename + ".pkl.gz"), 'wb', compresslevel=9) as file:
                pickle.dump(data_dict, file)

            print(f"Dataset saved to {self.dataset_path}")

    def _prepare_matrix(self):
        self.audio_chunks_inputs = np.array(self.audio_chunks_inputs, dtype=np.float32).reshape(-1, self.audio_chunks_inputs.shape[2], self.audio_chunks_inputs.shape[3])
        self.audio_chunks_outputs = np.array(self.audio_chunks_outputs, dtype=np.float32).reshape(-1, 1, self.audio_chunks_outputs.shape[2])


    def _prepare_next_file(self):
        """Prepare the chunks for only one file."""
        if self.input_length != -1:
            max_frames = int(self.min_min_frames // (self.input_length - self.stride))
            midi_features = self.midi_item_input[self.idx]['midi'].shape[-1]

            self.audio_chunks_inputs = np.zeros((1, max_frames, midi_features, self.input_length))
            self.audio_chunks_outputs = np.zeros((1, max_frames, self.output_length))

        else:
            midi_features = self.midi_item_input[self.idx]['midi'].shape[-1]
            self.audio_chunks_inputs = np.zeros((1, midi_features, len(self.midi_item_input[self.idx]['midi'])))
            self.audio_chunks_outputs = np.zeros(
                (1, self.output_dimension, len(self.audio_item_target[self.idx]['audio'])))

        audio_item = self.audio_item_target[self.idx]
        midi_item = self.midi_item_input[self.idx]

        if self.input_length == -1:  # take whole file
            self.audio_chunks_inputs[0] = midi_item['midi'].numpy()
            self.audio_chunks_outputs[0] = audio_item['audio'].numpy()

        else:  # split into chunks
            max_frames = int(self.min_min_frames // (self.input_length - self.stride)) - 1
            for n in range(max_frames):
                offset = self.input_length - self.output_length
                hop = int(n * (self.input_length - self.stride))
                end = hop + self.input_length

                self.audio_chunks_inputs[0, n:n+1, :] = midi_item['midi'][hop:end, :].numpy().T
                self.audio_chunks_outputs[0, n:n+1, :] = audio_item['audio'][offset + hop:end, :].numpy().T

        self.num_samples = max_frames * len(self.audio_item_target)

    def __shuffle__(self):
        """Shuffle the order of data samples"""
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        """Returns the number of batches per epoch"""
        return int(np.floor((self.num_samples - 1) / self.batch_size))

    def __getitem__(self, idx):
        """Get batch at position idx"""
        if idx >= self.__len__():
            raise IndexError(f"Index {idx} out of bounds for dataset with {self.__len__()} batches")

        if self.all_in_memory:
            indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
            input_batch = self.audio_chunks_outputs[indices]
            output_batch = self.audio_chunks_outputs[indices+1]
            cond_batch = self.audio_chunks_inputs[indices+1]

        else:
            inx = idx % (len(self.indices) // self.batch_size)
            if inx == len(self.indices) // self.batch_size:
                self.idx += 1
                self._prepare_next_file()
                inx = 0

            indices = self.indices[inx * self.batch_size:(inx + 1) * self.batch_size]

            input_batch = self.audio_chunks_outputs[indices]
            output_batch = self.audio_chunks_outputs[indices+1]
            cond_batch = self.audio_chunks_inputs[indices+1]

        return [input_batch, output_batch, cond_batch]

    def __on_epoch_end__(self):
        """Called at the end of each epoch"""
        if self.all_in_memory:
            self.indices = np.arange(self.num_samples-1)
        else:
            self.indices = np.arange(self.audio_chunks_inputs.shape[0])
        self.idx = 0
        if self.shuffle:
            self.__shuffle__()
