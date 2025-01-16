# scripts/prepare_data.py

import os
from utils import load_transcriptions, load_audio, find_files
import torchaudio
import random
import numpy as np
import torch

# Set the seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Verzeichnis der rohen Daten für Entwicklungsdaten
DATA_DIR = '../data/dev/'  # Annahme: Skript wird aus dem Verzeichnis "scripts" ausgeführt
OUTPUT_DIR = '../outputs/processed_audio/'

# Verzeichnis der rohen Daten für Testdaten
# DATA_DIR = '../data/test-clean/'  # Annahme: Skript wird aus dem Verzeichnis "scripts" ausgeführt
# OUTPUT_DIR = '../outputs/test_audio/'

# Sicherstellen, dass das Ausgabe-Verzeichnis existiert
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# Suche nach allen Transkriptionsdateien
transcription_files = find_files(DATA_DIR, '.trans.txt')


if not transcription_files:
    print("No transcription files found!")

# Verarbeitung jeder Transkriptionsdatei
for trans_file in transcription_files:
    # Lade Transkriptionen
    transcriptions = load_transcriptions(trans_file)
    
    # Verzeichnis der Audiodateien
    audio_dir = os.path.dirname(trans_file)
    
    # Suche nach allen Audiodateien
    audio_files = find_files(audio_dir, '.flac')
    
    for audio_file in audio_files:
        # Lade die Audiodatei
        waveform, sample_rate = load_audio(audio_file)
        
        # Resample auf 16kHz, falls notwendig
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
        
        # Speicherpfad für die konvertierte Audiodatei
        output_file_path = os.path.join(OUTPUT_DIR, os.path.basename(audio_file).replace('.flac', '.wav'))
        
        # Speichere die konvertierte Audiodatei im WAV-Format
        torchaudio.save(output_file_path, waveform, 16000)
        
        # Optional: Transkription speichern
        file_id = os.path.basename(audio_file).replace('.flac', '')
        transcription = transcriptions.get(file_id, "")
        with open(os.path.join(OUTPUT_DIR, 'transcriptions.txt'), 'a') as f:
            f.write(f"{output_file_path}\t{transcription}\n")

