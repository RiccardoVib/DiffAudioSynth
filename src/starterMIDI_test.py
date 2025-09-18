from loaderMIDI import MidiAudioDataset
from pathlib import Path
from InferenceTest import test_diffusion_model

noise_steps = 1000

model_path = "../../TrainedModels/"
root_dir_midi = "../../Files/MaestroDataset/TRAIN/"
root_dir_audio = "../../Files/MaestroDataset/TRAIN/"
dataset_path = "../../Files/MaestroDataset/"
batch_size = 1  # 16
fs = 48000 // 4
resolution = 2
output_lengths = [4096, 2024, 1024, 512]  ## 1 second
input_dimension = 1
output_dimension = 1
save_dataset = False
filename = 'Maestro'
load_dataset = False
shuffle = True
all_in_memory = True
data_type = 'torch.float32'
mono = True

for output_length in output_lengths:

    stride = 0#(3*output_length // 4)  # 0

    model_name = "DiffusionMIDI_STFT" + str(output_length)
    model_path_ = Path(model_path) / model_name

    root_dir_midi = "../../Files/MaestroDataset/TEST/"
    root_dir_audio = "../../Files/MaestroDataset/TEST/"

    dataset_val = MidiAudioDataset(root_dir_midi=root_dir_midi,
                                   root_dir_audio=root_dir_audio,
                                   dataset_path=dataset_path,
                                   batch_size=batch_size,
                                   resolution=resolution,
                                   output_length=output_length,
                                   stride=stride,
                                   input_dimension=input_dimension,  # Number of piano keys
                                   output_dimension=output_dimension,  # Mono audio
                                   fs=fs,
                                   save_dataset=save_dataset,
                                   filename=filename,
                                   load_dataset=load_dataset,
                                   shuffle=shuffle,
                                   all_in_memory=all_in_memory,
                                   mono=mono,
                                   midi_representation='pianoroll',  # or 'onset'
                                   note_range=(21, 108)  # A0 to C8
                                   )

    # Train diffusion model for epochs
    diffusion = test_diffusion_model(dataset_val, model_path_, noise_steps=noise_steps)
