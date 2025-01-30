"""
===============================================================================
    Project:        JarvisAI Bachelorthesis
    File:           prepare_data.py
    Description:    This script loads the dataset, transforms it into .wav format
                    and resamples data to 16 kHz as well as aligns the transcriptions 
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

# Directory of training purpose audio data before pre-processing and output directory after processing
DATA_DIR = '../data/dev/'  # check working dir alligns
OUTPUT_DIR = '../outputs/processed_audio/'

# Directory of testing purpose audio data before pre-processing and output directory after processing
# DATA_DIR = '../data/test-clean/'  # check working dir alligns
# OUTPUT_DIR = '../outputs/test_audio/'

# Define or create output directorys
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# search for transcription .txt file
transcription_files = find_files(DATA_DIR, '.trans.txt')


if not transcription_files:
    print("No transcription files found!")

# Process all transciptions
for trans_file in transcription_files:
    # Load transcriptions
    transcriptions = load_transcriptions(trans_file)
    
    # Link directory for audio files
    audio_dir = os.path.dirname(trans_file)
    
    # Search directory for all audio files with .flac format
    audio_files = find_files(audio_dir, '.flac')
    
    for audio_file in audio_files:
        # Load audio data
        waveform, sample_rate = load_audio(audio_file)
        
        # Resample to 16kHz (if necessary)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
        
        # Link output directory for saving results
        output_file_path = os.path.join(OUTPUT_DIR, os.path.basename(audio_file).replace('.flac', '.wav'))
        
        # Safe transformed data into output directory
        torchaudio.save(output_file_path, waveform, 16000)
        
        # Safe transcriptions into output directory
        file_id = os.path.basename(audio_file).replace('.flac', '')
        transcription = transcriptions.get(file_id, "")
        with open(os.path.join(OUTPUT_DIR, 'transcriptions.txt'), 'a') as f:
            f.write(f"{output_file_path}\t{transcription}\n")

