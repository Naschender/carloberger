"""
===============================================================================
    Project:        JarvisAI Bachelorthesis
    File:           read_checkpoints.py
    Description:    This script manually overwrites model parameters and saves best model 
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

import pytorch_lightning as pl
import torch
import os
from model import SpeechRecognitionModel
from model_lightning import LitSpeechRecognitionModel
import torch.nn as nn

# Define the checkpoint directory and output file
CHECKPOINT_DIR = "checkpoints"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "checkpoint_info.txt")

# Function to write information to the output file
def write_to_file(text):
    with open(OUTPUT_FILE, "a") as file:
        file.write(text + "\n")

# Function to load and print model and training parameters from a checkpoint
def read_checkpoint_parameters(checkpoint_path, num_classes):
    # Load the checkpoint using LitSpeechRecognitionModel
    criterion = nn.CTCLoss(blank=num_classes - 1)
    model = LitSpeechRecognitionModel(num_classes=num_classes, criterion=criterion)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    
    # Load the state dictionary into the Lightning model
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    
    # Write all available keys in the checkpoint to the file
    write_to_file(f"\nCheckpoint Keys: {list(checkpoint.keys())}\n")

    # Write hyperparameters (if available)
    if 'hparams' in checkpoint:
        write_to_file("Hyperparameters:")
        for key, value in checkpoint['hparams'].items():
            write_to_file(f"  {key}: {value}")

    # Write optimizer state (if available)
    if 'optimizer_states' in checkpoint:
        write_to_file("\nOptimizer State:")
        for state in checkpoint['optimizer_states']:
            write_to_file(str(state))

    # Write learning rate scheduler state (if available)
    if 'lr_schedulers' in checkpoint:
        write_to_file("\nLearning Rate Scheduler State:")
        write_to_file(str(checkpoint['lr_schedulers']))

    # Write callbacks state (if available)
    if 'callbacks' in checkpoint:
        write_to_file("\nCallbacks State:")
        write_to_file(str(checkpoint['callbacks']))

    # Access the underlying SpeechRecognitionModel
    speech_model = model.model

    write_to_file(f"\nLoaded model parameters from checkpoint: {checkpoint_path}\n")
    for name, param in speech_model.named_parameters():
        write_to_file(f"Parameter: {name}, Shape: {param.shape}, Requires Grad: {param.requires_grad}")

# Function to iterate through all checkpoints and read parameters
def read_all_checkpoints(checkpoint_dir, num_classes):
    # Clear the output file if it already exists
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
    
    if not checkpoint_files:
        write_to_file("No checkpoint files found.")
        return
    
    for ckpt_file in checkpoint_files:
        checkpoint_path = os.path.join(checkpoint_dir, ckpt_file)
        write_to_file(f"\nProcessing checkpoint: {ckpt_file}\n")
        read_checkpoint_parameters(checkpoint_path, num_classes)

if __name__ == '__main__':
    NUM_CLASSES = 29  
    read_all_checkpoints(CHECKPOINT_DIR, NUM_CLASSES)
    print(f"Checkpoint information has been written to {OUTPUT_FILE}")
