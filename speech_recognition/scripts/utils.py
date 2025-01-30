"""
===============================================================================
    Project:        JarvisAI Bachelorthesis
    File:           utils.py
    Description:    This script contains the transciption loading, 
                    audio data locating and loading for the Mel-Spectrogram-model 1.0
    Author:         Carlo Berger, Aalen University
    Email:          Carlo.Berger@studmail.htw-aalen.de
    Created:        2024-11-15
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
