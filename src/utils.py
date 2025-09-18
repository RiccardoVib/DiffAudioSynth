import math
import numpy as np
import re
from scipy.io.wavfile import write
import os


def save_audio_files(output_audio, prediction_audio, model_path, prefix, sample_rate=48000):
    """
    Save audio files in WAV format.

    Parameters:
        output_audio (np.ndarray): Output audio data array (processed).
        prediction_audio: Predicted labels or values (could be additional info to save).
        model_path (str): The path where to save the audio files (should exist).
    """
    # Create the model path directory if it doesn't exist
    os.makedirs(model_path, exist_ok=True)

    # Saving output audio
    output_file_path = os.path.join(model_path, prefix + '_output_audio.wav')
    output_audio = np.array(output_audio.squeeze(), dtype=np.float32)
    write(output_file_path, sample_rate, output_audio)  # Scale to int16

    # Saving output audio
    output_file_path = os.path.join(model_path, prefix + '_prediction_audio.wav')
    prediction_audio = np.array(prediction_audio.squeeze(), dtype=np.float32)
    write(output_file_path, sample_rate, prediction_audio)  # Scale to int16

    print(f"Audio files saved to {model_path}")

def natural_sort_key(s):
    """
    Function to use as a key for sorting strings in natural order.
    This ensures that strings with numbers are sorted in human-expected order.
    For example: ["file1", "file10", "file2"] -> ["file1", "file2", "file10"]

    Args:
        s: The string to convert to a natural sort key

    Returns:
        A list of string and integer parts that can be used for natural sorting
    """
    # Split the string into text and numeric parts
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]



def compute_lcm(x, y):
    """Compute the least common multiple of two numbers."""
    return (x * y) // math.gcd(x, y)


