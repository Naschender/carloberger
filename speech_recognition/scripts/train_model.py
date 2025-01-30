"""
===============================================================================
    Project:        JarvisAI Bachelorthesis
    File:           prepare_mp3.py
    Description:    This script contains the whole training process for 
                    training the Mel-Spectrogram-model 1.0
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

import warnings
import torch
import os
import sys
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchaudio.transforms import MelSpectrogram
from model import SpeechRecognitionModel
import torchaudio
import random
import os
import numpy as np
from torch.nn.utils.rnn import pad_sequence #(1)
from tqdm import tqdm
from data import SpeechDataset, custom_collate, vocab

import torchaudio

# Set the seed for reproducibility
seed = 42

# For Python's built-in random module
random.seed(seed)

# For NumPy
np.random.seed(seed)

# For PyTorch
torch.manual_seed(seed)

# For CUDA (GPU)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # For multiple GPUs

# To ensure deterministic behavior in convolution layers and other operations (this might slightly affect performance)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Define Hyperparameters
EPOCHS = 200
BATCH_SIZE = 1
LEARNING_RATE = 0.001 # Adam optimizer standard learning rate 0,001 
NUM_CLASSES = len(vocab) # total of 29: 26 Letters + Blank <sp> (Leerzeichen), Apostroph ', End of Sentence
WEIGHT_DECAY=1e-4

# Function to validate model on test data
def validate_model(model, test_loader):
    model.eval()  # Set the model to evaluation mode
    validation_loss = 0.0
    with torch.no_grad():  # No need to track gradients for validation
        for i, (inputs, targets, input_lens, output_lens) in enumerate(test_loader):
            # Forward pass
            outputs = model(inputs.cuda().permute(0, 2, 1))

            # Calculate the loss
            try:
                # Training loss calculation
                outputs = torch.clamp(outputs, min=-1e10, max=1e10).log_softmax(-1)
                loss = criterion(outputs.transpose(1, 0), targets.cuda(), input_lens.cuda(), output_lens.cuda())

            except RuntimeError as e:
                print(f"CTC loss calculation failed during validation at batch {i+1} with error: {e}")
                continue

            # Add to total validation loss
            validation_loss += loss.item()

    # Return average validation loss
    avg_validation_loss = validation_loss / len(test_loader)
    return avg_validation_loss


def add_weight_noise(model, stddev):
    """Adds Gaussian noise to the RNN weights of the model."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'rnn' in name:  # Only add noise to RNN layers
                noise = torch.randn_like(param) * stddev
                param.add_(noise)


# Training data directory and loading with dataset
feature_dir = '../outputs/processed_audio/'
transcription_file = '../outputs/processed_audio/transcriptions.txt'
train_dataset = SpeechDataset(feature_dir, transcription_file, apply_specaugment=True)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate, num_workers=1)
# train_dataset.__getitem__(0)

# Validation data directory and loading with dataset
# Validation dataset and dataloader
test_feature_dir = '../outputs/test_audio/'
test_transcription_file = '../outputs/test_audio/transcriptions.txt'
test_dataset = SpeechDataset(test_feature_dir, test_transcription_file, apply_specaugment=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate, num_workers=4)
# test_dataset.__getitem__(0)

# waveform, sample_rate = torchaudio.load('../outputs/processed_audio/84-121123-0000.wav')  
# features = torchaudio.compliance.kaldi.fbank(waveform, num_mel_bins=80, snip_edges=False)

# Model, criterion, optimizer and rate scheduler initialisieren
model = SpeechRecognitionModel(num_classes=NUM_CLASSES)
model.cuda()
criterion = nn.CTCLoss(blank=NUM_CLASSES - 1)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)


# Variables for optimizing
progress = tqdm(range(EPOCHS))
minimum_loss = 5.0
early_stop_counter = 0
early_stop_patience = 100
best_val_loss = float('inf')

# Training (Gradient Descent Algorithmn)
for epoch in progress:
    model.train()
    # Overall training Loss
    running_loss = 0.0
    
    # variable i equals to often used "batch_index" 
    for i, (inputs, targets, input_lens, output_lens) in enumerate(train_loader):
        # resetting the gradients from previous iteration
        # because without it would accumilate the gradients instead of learning from its faults
        # inputs = inputs / inputs.abs().max()  # Normalize the inputs
        optimizer.zero_grad()
        
        # Debug outputs for ensuring valid values
        if i % 10 == 0:
            # Check for NaN or Inf values in inputs or input_lens
            if torch.isnan(inputs).any() or torch.isinf(inputs).any():
                print(f"NaN or Inf detected in inputs at batch {i+1}")
                continue
        
            if torch.isnan(input_lens).any() or torch.isinf(input_lens).any() or (input_lens <= 0).any():
                print(f"Invalid input lengths at batch {i+1}")
                continue

            if torch.isnan(output_lens).any() or torch.isinf(output_lens).any() or (output_lens <= 0).any():
                print(f"Invalid output lengths at batch {i+1}")
                continue

            # Ensure input lengths are greater than target lengths
            if (input_lens < output_lens).any():
                print(f"Input lengths are shorter than target lengths at batch {i+1}")
                continue
        # print("input_lens_before_output:", input_lens)
        # print("output_lens_before_output:", output_lens)            
        # Modellvorhersage
        outputs = model(inputs.cuda().permute(0, 2, 1))
        
        # Check for NaN or Inf in model outputs
        if torch.isnan(outputs).any() or torch.isinf(outputs).any():
            print(f"NaN or Inf detected in model outputs at batch {i+1}")
            continue

        # Loss calculation
        try:
            # print("input_len_before_loss:", input_lens)
            # print("output_lens_before_loss:", output_lens)
            # Training loss calculation
            outputs = torch.clamp(outputs, min=-1e10, max=1e10).log_softmax(-1)
            loss = criterion(outputs.transpose(1, 0), targets.cuda(), input_lens.cuda(), output_lens.cuda())

        except RuntimeError as e:
            print(f"CTC loss calculation failed at batch {i+1} with error: {e}")
            continue

        # Compute Partial Derivates or gradient  of the loss?
        loss.backward()

        # Gradient clipping to prevent gradients to explode
        # comparison between 5.0, 4.0, 3.0, 2.0, 1.0 resulted in 3.0 as the best value with best stability and most less overfitting
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

        # Optimizer call
        optimizer.step()

        # Add each loss to the overall training loss
        running_loss += loss.item()
        
        # if i % 10 == 0:  # Ausgabe nach jedem 10. Batch
            # print(f'Epoch {epoch+1}, Batch {i+1}, Loss: {loss.item()}')

    # Apply weight noise after each epoch
    #    if epoch % 10 == 0:  # Add weight noise every 10 batches
    add_weight_noise(model, stddev=0.0001)

    # run model validation from test training data
    validation_loss = validate_model(model, test_loader)
    print(f'\nEpoch {epoch+1}, Validation Loss: {validation_loss:.4f}')

    if validation_loss < best_val_loss:
        best_val_loss = validation_loss
        early_stop_counter = 0
        torch.save(model.state_dict(), "best_model.pt")  # Save best model based on validation loss
        print(f'Early stop counter resetted and best model so far saved.')
    else:
        early_stop_counter += 1
        
    if early_stop_counter >= early_stop_patience:
        print(f'Early stopping triggered at EPOCH {epoch} after a plateau of {early_stop_patience} epochs with no improvement.')
        break

    # learning rate scheduler calculation
    avg_loss = running_loss / len(train_loader)

    # for more information set this command into the second for loop
    progress.set_description(f'Epoch {epoch+1} complete. Average Loss: {running_loss / len(train_loader)}')
    
    if minimum_loss > avg_loss:
        minimum_loss = avg_loss
        # safe model if average loss is on minimum
        torch.save(model.state_dict(), "speechRecognition1.pt")

    # Step the scheduler based on the average loss
    scheduler.step(avg_loss)
print("Training complete!")
