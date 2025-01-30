"""
===============================================================================
    Project:        JarvisAI Bachelorthesis
    File:           feature_extraction.py
    Description:    This script contains the for the Mel-Spectrogram-model 
                    necessary feature extraction of the prepared audio files
    Author:         Carlo Berger, Aalen University
    Email:          Carlo.Berger@studmail.htw-aalen.de
    Created:        2024-10-30
    Last Modified:  2025-01-30
    Version:        2.0
===============================================================================

    Copyright (c) 2025 Carlo Berger

    This software is provided "as is", without warranty of any kind, express
    or implied, including but not limited to the warranties of merchantability,
    fitness for a particular purpose, and non-infringement. In no event shall
    the authors or copyright holders be liable for any claim, damages, or other
    liability, whether in an action of contract, tort, or otherwise, arising
    from, out of, or in connection with the software or the use or other dealings
    in the software.

    All code is licenced under the opensource License. You may not use this file except
    in compliance with the License.

===============================================================================
"""


import os
import random
import numpy as np
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
from utils import find_files
from IPython.display import Audio
from matplotlib.patches import Rectangle
from torchaudio.utils import download_asset
import librosa
import matplotlib.pyplot as plt

# print(torch.__version__)
# print(torchaudio.__version__)

# Set the seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

'''
# Function to normalize a Tensor
# divide through zero debug
# was not necessary for training at the end
def normalize(tensor):
    mean = tensor.mean()
    std = tensor.std()
    if std == 0:
        return tensor  # Avoid division by zero
    return (tensor - mean) / std
'''

def plot_waveform(waveform, sr, title="Waveform", ax=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sr

    if ax is None:
        _, ax = plt.subplots(num_channels, 1)
    ax.plot(time_axis, waveform[0], linewidth=1)
    ax.grid(True)
    ax.set_xlim([0, time_axis[-1]])
    ax.set_title(title)


def plot_spectrogram(specgram, title=None, ylabel="freq_bin", ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto", interpolation="nearest")


def plot_fbank(fbank, title=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Filter bank")
    axs.imshow(fbank, aspect="auto")
    axs.set_ylabel("frequency bin")
    axs.set_xlabel("mel bin")


# Directory of preprocessed Audio data before feature extraction
# PROCESSED_DATA_DIR = '../outputs/processed_audio/'
# FEATURE_OUTPUT_DIR = '../outputs/features/'
# VALIDATION_OUTPUT_DIR = '../outputs/features_val'

# Directory of preprocessed Audio data after feature extraction
PROCESSED_DATA_DIR = '../outputs/test_audio/'
FEATURE_OUTPUT_DIR = '../outputs/features_test/'
VALIDATION_OUTPUT_DIR = '../outputs/features_test_val'

# Define / create working dir
os.makedirs(FEATURE_OUTPUT_DIR, exist_ok=True)
os.makedirs(VALIDATION_OUTPUT_DIR, exist_ok=True)

# Scan for all .wav files in defined directorys
audio_files = find_files(PROCESSED_DATA_DIR, '.wav')

# Extract features from preprocessed audio data
for audio_file in audio_files:
    # Load audio data from directory
    SPEECH_WAVEFORM, SAMPLE_RATE = torchaudio.load(audio_file)
    
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        n_fft=512,    # Anzahl der FFT-Punkte (sollte hoch genug sein)
        hop_length=160,  # Schrittweite (in Samples) zwischen den Fenstern
        f_min=0,
        f_max=8000, 
        n_mels=80  # Anzahl der Mel-Bänder (reduziert, um Warnungen zu vermeiden)
    )

    # Normalisierung des Mel-Spektrogramms
    # durchführen da loss noch sehr hoch ist
    # mel_spectrogram = normalize(mel_spectrogram)

    # Perform transform
    spec = mel_spectrogram(SPEECH_WAVEFORM)

    # prepare save of the features into directory
    feature_file_path = os.path.join(FEATURE_OUTPUT_DIR, os.path.basename(audio_file).replace('.wav', '.pt'))
    
    # Save the features into directory
    torch.save(spec, feature_file_path)

    # Plot and save the spectrogram as a PNG file
    fig, axs = plt.subplots(2, 1)
    plot_waveform(SPEECH_WAVEFORM, SAMPLE_RATE, title="Original waveform", ax=axs[0])
    plot_spectrogram(spec[0], title="Spectrogram", ax=axs[1])
    fig.tight_layout()

    # Prepare save for .png files into Directory
    png_file_path = os.path.join(VALIDATION_OUTPUT_DIR, os.path.basename(audio_file).replace('.wav', '_spectrogram.png'))
    
    # Save the plot as a PNG file into directory
    fig.savefig(png_file_path)
    plt.close(fig)  # Close the figure after saving to free up memory


print("success")