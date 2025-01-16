# scripts/utils.py

import os
from typing import List, Tuple
import torchaudio
import soundfile as sf

def load_transcriptions(file_path: str) -> dict:
    """
    Lädt die Transkriptionen aus einer gegebenen Textdatei.
    
    :param file_path: Pfad zur Transkriptionsdatei
    :return: Ein Wörterbuch mit der Zuordnung von Dateinamen zu Transkriptionen
    """
    transcriptions = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                file_id, transcription = parts
                transcriptions[file_id] = transcription
    return transcriptions

def load_audio(file_path: str) -> Tuple:
    """
    Lädt eine Audio-Datei und gibt die Waveform und die Sampling-Rate zurück.
    
    :param file_path: Pfad zur Audiodatei
    :return: Waveform-Daten und Sampling-Rate
    """
    waveform, sample_rate = torchaudio.load(file_path)
    return waveform, sample_rate

def find_files(root_dir: str, file_extension: str) -> List[str]:
    """
    Durchsucht ein Verzeichnis rekursiv nach Dateien mit einer bestimmten Erweiterung.
    
    :param root_dir: Wurzelverzeichnis, das durchsucht werden soll
    :param file_extension: Dateierweiterung, nach der gesucht wird (z.B. '.flac')
    :return: Liste der gefundenen Dateipfade
    """
    files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.endswith(file_extension):
                files.append(os.path.join(dirpath, f))
    return files
