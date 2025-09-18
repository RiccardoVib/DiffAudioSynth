from loaderMIDI import MidiAudioDataset
from pathlib import Path
from DiffusionMIDIAudio import train_diffusion_model


noise_steps = 1000
epochs = 0


model_path = "../../TrainedModels/"

root_dir_midi = "../../Files/MaestroDataset/TRAIN/"
root_dir_audio = "../../Files/MaestroDataset/TRAIN/"
dataset_path = "../../Files/MaestroDataset/"
batch_size = 4  # 16
fs = 48000 // 4
resolution = 2
output_lengths = [2048, 1024, 512] ## 1 second
output_lengths = [4096] ## 1 second
stride = 0
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
    model_name = "DiffusionMIDI_STFT" + str(output_length)
    model_path = Path(model_path) / model_name

    if epochs != 0:

        dataset = MidiAudioDataset(root_dir_midi=root_dir_midi,
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
    else:
        dataset = None

    root_dir_midi = "../../Files/MaestroDataset/TEST/"
    root_dir_audio = "../../Files/MaestroDataset/TEST/"
    batch_size = 4
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
    diffusion = train_diffusion_model(dataset, dataset_val, model_path, noise_steps=noise_steps, epochs=epochs)