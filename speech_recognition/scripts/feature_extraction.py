# scripts/feature_extraction.py

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
# Funktion zur Normalisierung eines Tensors
# vorschlag von gpt um hohen loss zu verbessern
# weiter ergänzt so das std nicht null sein kann um fehler zu vermeiden
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


# Verzeichnis der verarbeiteten Audiodaten Entwicklungsdaten
# PROCESSED_DATA_DIR = '../outputs/processed_audio/'
# FEATURE_OUTPUT_DIR = '../outputs/features/'
# VALIDATION_OUTPUT_DIR = '../outputs/features_val'

# Verzeichnis der verarbeiteten Audiodaten Entwicklungsdaten
PROCESSED_DATA_DIR = '../outputs/test_audio/'
FEATURE_OUTPUT_DIR = '../outputs/features_test/'
VALIDATION_OUTPUT_DIR = '../outputs/features_test_val'

# Sicherstellen, dass das Ausgabe-Verzeichnis existiert
os.makedirs(FEATURE_OUTPUT_DIR, exist_ok=True)
os.makedirs(VALIDATION_OUTPUT_DIR, exist_ok=True)

# Suche nach allen WAV-Dateien
audio_files = find_files(PROCESSED_DATA_DIR, '.wav')

# Extrahiere Features aus jeder Audiodatei
for audio_file in audio_files:
    # Lade die Audiodatei
    SPEECH_WAVEFORM, SAMPLE_RATE = torchaudio.load(audio_file)
    
    # Define transform
    # spectrogram = T.Spectrogram(n_fft=512)  
    # features = torchaudio.compliance.kaldi.fbank(SPEECH_WAVEFORM, n_fft=512, num_mel_bins=80, snip_edges=False)
    
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
    # spec = spectrogram(SPEECH_WAVEFORM)
    spec = mel_spectrogram(SPEECH_WAVEFORM)

    # Speicherpfad für die Features
    feature_file_path = os.path.join(FEATURE_OUTPUT_DIR, os.path.basename(audio_file).replace('.wav', '.pt'))
    
    # Speichere die Features als Torch-Tensor
    torch.save(spec, feature_file_path)

    # Plot and save the spectrogram as a PNG file
    fig, axs = plt.subplots(2, 1)
    plot_waveform(SPEECH_WAVEFORM, SAMPLE_RATE, title="Original waveform", ax=axs[0])
    plot_spectrogram(spec[0], title="Spectrogram", ax=axs[1])
    fig.tight_layout()

    # Speicherpfad für die PNG-Datei
    png_file_path = os.path.join(VALIDATION_OUTPUT_DIR, os.path.basename(audio_file).replace('.wav', '_spectrogram.png'))
    
    # Save the plot as a PNG file
    fig.savefig(png_file_path)
    plt.close(fig)  # Close the figure after saving to free up memory


print("success")