import os
import csv
from utils import load_audio, find_files
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

# Input and output directories
DATA_DIR = '../data/train/'  # Directory containing .mp3 files
OUTPUT_DIR = '../outputs/train_audio/'  # Directory to save .wav files and transcriptions

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Path to the transcription file
transcription_file_path = os.path.join(DATA_DIR, 'train.csv')

# Load transcriptions into a dictionary (semicolon-delimited file)
transcriptions = {}
with open(transcription_file_path, 'r', encoding='latin1') as f:  # Use latin1 encoding
    reader = csv.reader(f, delimiter=';')  # Specify semicolon as the delimiter
    for row in reader:
        # Skip rows with incorrect number of columns
        if len(row) >= 2:
            file_name, transcription = row[0].strip(), row[1].strip()
            transcriptions[file_name] = transcription

# Debug: Print the first few entries in the transcriptions dictionary
print("Loaded transcriptions:", list(transcriptions.items())[:5])

# Find all .mp3 files in the directory
audio_files = find_files(DATA_DIR, '.mp3')

# Open the transcription output file once and write to it in each iteration
transcription_output_path = os.path.join(OUTPUT_DIR, 'transcriptions.txt')

with open(transcription_output_path, 'w', encoding='utf-8') as transcription_file:  # Open in write mode
    for i, audio_file in enumerate(audio_files, start=1):
        print(f"Processing file {i}/{len(audio_files)}: {audio_file}")  # Debug: Show progress

        try:
            # Load the audio file
            print(f"Loading audio file: {audio_file}")  # Debug: File being loaded
            waveform, sample_rate = load_audio(audio_file)

            # Resample to 16kHz if necessary
            if sample_rate != 16000:
                print(f"Resampling {audio_file} from {sample_rate} Hz to 16kHz")  # Debug: Resampling info
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                waveform = resampler(waveform)

            # Construct the output file path
            file_name = os.path.basename(audio_file).replace('.mp3', '.wav')
            output_file_path = os.path.join(OUTPUT_DIR, file_name)

            # Save the audio file in WAV format
            torchaudio.save(output_file_path, waveform, 16000)
            print(f"Saved WAV file: {output_file_path}")  # Debug: Save confirmation

            # Retrieve and save the corresponding transcription
            file_id = os.path.basename(audio_file)
            transcription = transcriptions.get(file_id, None)

            if transcription is None:
                print(f"WARNING: No transcription found for {file_id}")  # Debug: Missing transcription
                transcription = ""
            else:
                print(f"Found transcription for {file_id}: {transcription}")  # Debug: Found transcription


            # Write the transcription and audio file path to the output file
            transcription_file.write(f"{output_file_path}\t{transcription}\n")
            print(f"Transcription saved for: {file_id}")  # Debug: Transcription confirmation

        except Exception as e:
            print(f"Error processing file {audio_file}: {e}")  # Debug: Error info
