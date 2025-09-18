# Low-Latency Diffusion-based Audio Synthesizer

This code repository is for the article _Low-Latency Diffusion-based Audio Synthesizer_, in review.

This repository contains all the necessary utilities to use our architectures. Find the code located inside the "./src" folder, and the weights of pre-trained models inside the "./weights" folder

### Folder Structure

```
./
├── src
├── evaluation
├── plots
└── weights
    ├── 512
    ├── 1024
    ├── 2048
    └── 4096
        
```

### Contents

1. [Datasets](#datasets)
2. [Architecture](#architecture)
3. [How to Train and Run Inference](#how-to-train-and-run-inference)
4. [Audio Examples](#audio-examples)

<br/>

# Architecture

![Alt text](plots/DiffSynth.png)
![Alt text](plots/DiffSynth1.png)


# Datasets

Datasets are available [here](https://magenta.withgoogle.com/datasets/maestro)

Our architectures were evaluated on the 2018 folder.


# How To Train and Run Inference 

This code relies on Python 3.9 and torch.
First, install Python dependencies:
```
cd ./src
pip install -r requirements.txt
```

To train models, use the starter.py script.
Ensure you have loaded the dataset into the chosen datasets folder

Available options (to be changed in starterMIDI.py or starterMIDI_test.py) : 
* --model_path - Folder directory in which to store the trained models [str] (default ="./models")
* --root_dir_midi - Folder directory in which the midi datasets are stored [str] (default="./datasets/midi")
* --root_dir_audio - Folder directory in which the audio datasets are stored [str] (default="./datasets/audio")
* --dataset_path - Folder directory in which the dataset is stored [str] (default="./datasets")
* --filename - The names of the datasets to use. [str] (default="Maestro"])
* --epochs - Number of training epochs. [int] (default=1000)
* --batch_size - The size of each batch [int] (default=4)
* --fs - The desired sample rate [int] (default=48000)
* --stride - Number of samples of the overlap [int] (default=0)
* --noise_steps = The number of denoising steps. [ [int] ] (default=1000)
* --mono - If mono or stereo [bool] (default=True)
* --data_type - Data type considered. [str] (default='torch.float32')
* --resolution - MIDI resolution. [in] (default=2)
* --output_lengths - Frame size to produce each iteration [[int]]. (default=[4096, 2048, 1024, 512])
* --save_dataset - If save the dataset [bool] (default=False)
* --load_dataset - If load an already saved dataset [bool] (default=False)
* --shuffle - If shuffle the dataset [bool] (default=False)
* --all_in_memory - If load all the data in the memory [bool] (default=True)

Example training case: 
```
cd ./code/

python starterMIDI.py
```

To only run inference on an existing pre-trained model, use the starterMIDI_test.py script. In this case, ensure you have the existing model and dataset (to use for inference) both in their respective directories with corresponding names.

Example inference case:
```
cd ./code/

python starterMIDI_test.py
```

# Audio Example

[Listen to Audio]("audio/real_output.wav")

# Bibtex

If you use the code included in this repository or any part of it, please acknowledge 
its authors by adding a reference to these publications:

```


```
