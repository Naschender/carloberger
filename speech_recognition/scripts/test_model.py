"""
===============================================================================
    Project:        JarvisAI Bachelorthesis
    File:           test_model.py
    Description:    debugging script to test model results
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

import torch
from data import custom_collate, vocab, SpeechDataset
from torch.utils.data import DataLoader
from model import SpeechRecognitionModel
from torch.utils.data import Dataset
import os
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import torchaudio
import math
from heapq import heappush, heappop
import torch.nn.functional as F
from model_lightning import LitSpeechRecognitionModel
import torch.nn as nn


# Define Hyperparameters
BATCH_SIZE = 1
NUM_CLASSES = len(vocab)
test_feature_dir = '../outputs/test_audio/'
test_transcription_file = '../outputs/test_audio/transcriptions.txt'
test_dataset = SpeechDataset(test_feature_dir, test_transcription_file)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate, num_workers=4)
test_dataset.__getitem__(0)

# load model from train_model.py
# model = SpeechRecognitionModel(NUM_CLASSES)
# model.load_state_dict(torch.load('best_model.pt', weights_only=True))
# load model from model_lightning.py
model = LitSpeechRecognitionModel.load_from_checkpoint('checkpoints/model_epoch=44-val_loss=0.70.ckpt', num_classes=NUM_CLASSES, criterion=nn.CTCLoss(blank=28))
model = model.model
model.eval()
model.cuda()

'''
# Lade Daten
feature_dir = '../outputs/features/'
transcription_file = '../outputs/processed_audio/transcriptions.txt'
train_dataset = SpeechDataset(feature_dir, transcription_file)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate, num_workers=4)
train_dataset.__getitem__(0)
'''

def ctc_greedy_decode(output, vocab, blank_id=NUM_CLASSES - 1):
    """Greedy decoding for CTC output."""
    decoded = []
    prev_token = None

    for timestep in output.argmax(dim=-1).squeeze():  # Get most likely token at each timestep
        token = timestep.item()
        if token != blank_id and token != prev_token:
            if vocab.index2word[token] == '<sp>':
                decoded.append(' ')  # Replace <sp> with space
            else:
                decoded.append(vocab.index2word[token])
        prev_token = token

    return ''.join(decoded)


def ctc_beam_search_decode(output, vocab, blank_id=NUM_CLASSES - 1, beam_width=7):
    """Beam search decoding for CTC output with <sp> replacement for spaces."""
    sequences = [([], 0)]  # Each entry is a pair (sequence, score)

    for timestep in output.squeeze().cpu():  # Assuming output is on CUDA
        all_candidates = []
        
        # Generate candidates for each existing sequence
        for seq, score in sequences:
            for idx, prob in enumerate(timestep):
                if prob > 0:  # Ensure non-zero probability
                    candidate_seq = seq + [idx]
                    candidate_score = score + math.log(prob + 1e-10)  # Robust log
                    all_candidates.append((candidate_seq, candidate_score))
        
        # Sort by score and select top sequences
        all_candidates = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]
        sequences = all_candidates

    # Process the best sequence to remove blanks and repeated characters
    best_seq, _ = sequences[0]
    decoded = []
    prev_token = None
    for token in best_seq:
        if token != blank_id and token != prev_token:
            if vocab.index2word[token] == '<sp>':
                decoded.append(' ')
            else:
                decoded.append(vocab.index2word[token])
        prev_token = token

    return ''.join(decoded)



input_feature, target, input_len, target_len = next(iter(test_loader))
output = model(input_feature.cuda().permute(0, 2, 1))

print(output)
print("Logit statistics - Mean:", output.mean().item(), "Std:", output.std().item())

# Apply softmax along the last dimension to get probabilities
probabilities = F.softmax(output, dim=-1)

# Display prediction matrix
print("\nPrediction Matrix (Time Steps x Characters):")
predicted_indices = probabilities.argmax(dim=-1).squeeze().cpu().detach().numpy()  # Shape: (time_steps,)
predicted_chars = [vocab.index2word[idx] if idx in vocab.index2word else '<unk>' for idx in predicted_indices]

# Print the prediction matrix with time steps and detected characters
print("Time Step | Predicted Character")
print("-" * 30)
for i, char in enumerate(predicted_chars):
    print(f"{i:9} | {char}")

# Decode the output using the CTC greedy decoding function
decoded_text = ctc_beam_search_decode(probabilities, vocab)
    
# Display the original transcription and the predicted transcription
original_text = ''.join([vocab.index2word[idx.item()] for idx in target.squeeze()])
print("Original transcription:", original_text)
print("Predicted transcription:", decoded_text)

pass


# # Run predictions on test set
# for input_feature, target, input_len, target_len in test_loader:
#     input_feature = input_feature.cuda().permute(0, 2, 1)  # Reshape input for model
#     with torch.no_grad():
#         output = model(input_feature)  # Get raw model output logits

#     # Decode the output using the CTC greedy decoding function
#     decoded_text = ctc_greedy_decode(output, vocab)
    
#     # Display the original transcription and the predicted transcription
#     original_text = ''.join([vocab.index2word[idx.item()] for idx in target.squeeze()])
#     print("Original transcription:", original_text)
#     print("Predicted transcription:", decoded_text)
    
#     break  # Exit after first batch for demonstration
