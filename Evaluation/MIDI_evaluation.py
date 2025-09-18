import numpy as np
import librosa
import pretty_midi
from scipy.signal import find_peaks
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')




def detect_onsets(audio, sr=22050, hop_length=512):
    """
    Detect onset times in audio using spectral flux
    Returns: Array of onset times in seconds
    """
    # Compute onset strength
    onset_frames = librosa.onset.onset_detect(
        y=audio,
        sr=sr,
        hop_length=hop_length,
        units='frames'
    )

    # Convert frames to time
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)

    return onset_times

def compute_onset_accuracy_audio_ref_gen(ref_audio, gen_audio, sr, hop_length=256, tolerance=0.05):
    ref_onsets = detect_onsets(ref_audio, sr, hop_length)
    gen_onsets = detect_onsets(gen_audio, sr, hop_length)
    return compute_onset_accuracy(ref_onsets, gen_onsets, tolerance)

def compute_onset_accuracy(midi_onsets, detected_onsets, tolerance=0.05):
    """
    Compute onset detection accuracy with tolerance window

    Args:
        midi_onsets: List of ground truth onset times from MIDI
        detected_onsets: List of detected onset times from audio
        tolerance: Time tolerance in seconds for matching

    Returns:
        dict: Precision, recall, F1-score
    """
    midi_onsets = np.array(midi_onsets)
    detected_onsets = np.array(detected_onsets)

    # Find matches within tolerance
    matches = 0

    for detected_onset in zip(detected_onsets):
        # Find closest detected onset within tolerance
        distances = np.abs(detected_onset - midi_onsets)
        idx_min = np.argmin(distances)
        if distances[idx_min] <= tolerance:
            matches += 1

    # Calculate metrics
    precision = matches / len(detected_onsets) if len(detected_onsets) > 0 else 0
    recall = matches / len(midi_onsets) if len(midi_onsets) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'matches': matches,
        'total_midi': len(midi_onsets),
        'total_detected': len(detected_onsets)
    }


def estimate_f0(audio, sr=22050, hop_length=512, fmin=80, fmax=2000):
    """
    Estimate F0 (fundamental frequency) using piptrack
    Returns: (times, frequencies) arrays
    """
    # Estimate pitch using piptrack
    pitches, magnitudes = librosa.piptrack(
        y=audio,
        sr=sr,
        hop_length=hop_length,
        fmin=fmin,
        fmax=fmax,
        threshold=0.1
    )

    # Extract the most confident pitch at each time frame
    times = librosa.frames_to_time(np.arange(pitches.shape[1]), sr=sr, hop_length=hop_length)
    f0_sequence = []

    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t] if magnitudes[index, t] > 0 else 0
        f0_sequence.append(pitch)

    return times, np.array(f0_sequence)


def midi_note_to_freq(midi_note):
    """Convert MIDI note number to frequency in Hz"""
    return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))


def compute_pitch_accuracy_audio_ref_gen(ref_audio, gen_audio, sr, hop_length=512, tolerance_cents=50):
    ref_times, ref_f0s = estimate_f0(ref_audio, sr, hop_length)
    gen_times, gen_f0s = estimate_f0(gen_audio, sr, hop_length)

    # Align time length by truncation
    min_len = min(len(ref_f0s), len(gen_f0s))
    ref_f0s, gen_f0s = ref_f0s[:min_len], gen_f0s[:min_len]

    # Compute pitch accuracy as fraction of frames where pitch differs by less than tolerance
    ref_nonzero = ref_f0s > 0
    gen_nonzero = gen_f0s > 0
    voiced_frames = ref_nonzero & gen_nonzero

    if np.sum(voiced_frames) == 0:
        return {'accuracy': 0, 'mean_error_cents': float('inf')}

    cents_error = 1200 * np.abs(np.log2(gen_f0s[voiced_frames] / ref_f0s[voiced_frames]))
    accuracy = np.mean(cents_error <= tolerance_cents)
    mean_error = np.mean(cents_error)

    return {'accuracy': accuracy, 'mean_error_cents': mean_error}


def compute_polyphony_preservation_audio_ref_gen(ref_audio, gen_audio, sr, hop_length=512):
    # Use multipitch detection (e.g., librosa's chroma or spectral peaks)
    ref_chroma = librosa.feature.chroma_stft(y=ref_audio, sr=sr, hop_length=hop_length)
    gen_chroma = librosa.feature.chroma_stft(y=gen_audio, sr=sr, hop_length=hop_length)

    mean_error = np.mean(np.abs(ref_chroma - gen_chroma))
    accuracy = 1-mean_error
    return {'polyphony_accuracy': accuracy, 'mean_polyphony_error': mean_error}

def evaluate_audio_alignment(ref_audio_file, gen_audio_file, sr=22050):
    # Load audios
    ref_audio, _ = librosa.load(ref_audio_file, sr=sr)
    gen_audio, _ = librosa.load(gen_audio_file, sr=sr)

    # Compute metrics
    onset = compute_onset_accuracy_audio_ref_gen(ref_audio, gen_audio, sr)
    pitch = compute_pitch_accuracy_audio_ref_gen(ref_audio, gen_audio, sr)
    duration = compute_duration_accuracy_audio_ref_gen(ref_audio, gen_audio, sr)
    polyphony = compute_polyphony_preservation_audio_ref_gen(ref_audio, gen_audio, sr)

    results = {
        'onset_detection': onset,
        'pitch_accuracy': pitch,
        'polyphony_preservation': polyphony,
        'summary': {
            'onset_f1': onset.get('f1', None),
            'pitch_accuracy': pitch['accuracy'],
            'polyphony_accuracy': polyphony['polyphony_accuracy']
        }
    }
    return results

def print_evaluation_results(results):
    """Print formatted evaluation results for audio-to-audio comparison"""
    print("\n" + "=" * 50)
    print("Audio-to-Audio Alignment Evaluation Results")
    print("=" * 50)

    print("\n1. Onset Detection:")
    onset = results['onset_detection']
    # Onset accuracy metrics from audio-to-audio comparison uses only precision, recall, f1
    print(f"   Precision: {onset.get('precision', 'N/A'):.3f}" if 'precision' in onset else "   Precision: N/A")
    print(f"   Recall: {onset.get('recall', 'N/A'):.3f}" if 'recall' in onset else "   Recall: N/A")
    print(f"   F1-Score: {onset.get('f1', 'N/A'):.3f}" if 'f1' in onset else "   F1-Score: N/A")
    if 'matches' in onset and 'total_midi' in onset:
        print(f"   Matches: {onset['matches']}/{onset['total_midi']}")
    else:
        print("   Matches info not available")

    print("\n2. Pitch Accuracy:")
    pitch = results['pitch_accuracy']
    print(f"   Accuracy: {pitch.get('accuracy', 0):.3f}")
    print(f"   Mean Error: {pitch.get('mean_error_cents', float('nan')):.1f} cents" if 'mean_error_cents' in pitch else "   Mean Error: N/A")
    print(f"   Std Error: {pitch.get('std_error_cents', float('nan')):.1f} cents" if 'std_error_cents' in pitch else "   Std Error: N/A")


    print("\n3. Polyphony Preservation:")
    polyphony = results['polyphony_preservation']
    print(f"   Accuracy: {polyphony.get('polyphony_accuracy', 0):.3f}")
    print(f"   Mean Polyphony Error: {polyphony.get('mean_polyphony_error', float('nan')):.2f}")

    print("\n" + "=" * 50)
    print("Summary Scores:")
    summary = results.get('summary', {})
    if summary:
        for metric, value in summary.items():
            print(f"   {metric}: {value:.3f}")
    else:
        print("No summary scores available.")


# Example usage
if __name__ == "__main__":
    from FullEvalaution import load_audio_pairs

    # Example use:
    folders = ['../../TrainedModels/AudioFiles/512/',
               '../../TrainedModels/AudioFiles/1024/',
               '../../TrainedModels/AudioFiles/2048/',
               '../../TrainedModels/AudioFiles/4096/']

    for folder in folders:
        pairs = load_audio_pairs(folder)
        for (audio1, sr1), (audio2, sr2), prefix, (path1, path2) in pairs:
            print(f"Loaded pair {prefix} with sr {sr1}")
            results = evaluate_audio_alignment(path1, path2, sr=sr1)
            # Print nicely formatted results
            print_evaluation_results(results)

