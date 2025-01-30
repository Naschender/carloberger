"""
===============================================================================
    Project:        JarvisAI Bachelorthesis
    File:           model.py
    Description:    This script containing the Mel-Spectrogram-model 1.0 
                    which uses PyTorch package 
    Author:         Carlo Berger, Aalen University
    Email:          Carlo.Berger@studmail.htw-aalen.de
    Created:        2024-11-15
    Last Modified:  2025-01-30
    Version:        4.0
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

import random
import numpy as np
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
import torch
import torch.nn as nn
import pytorch_lightning as pl

# Set the seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class SpeechRecognitionModel(nn.Module):
    def __init__(self, num_classes):
        # here the model parameters are getting defined
        super(SpeechRecognitionModel, self).__init__()
        
        # Convolutional Layer zur Merkmalsextraktion
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.38),  # Droup out rate 0.2, 0.3 are too low while 0.4 is to high
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.38), # Droup out rate 0.2, 0.3 are too low while 0.4 is to high
            # nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(64),
            # nn.ReLU(),
            # nn.Dropout(0.38), # Droup out rate 0.2, 0.3 are too low while 0.4 is to high
        )

        
        # Recurrent Layer for Sequence processing
        # adjustment of input size and hidden_size because of CNN form and more layers 32 * 40 / 64 * 64
        self.rnn = nn.GRU(input_size=32 * 40, hidden_size=128, num_layers=3, batch_first=True, bidirectional=True)
        
        # Fully connected layer to reduce to vocab of 29 possible outputs
        self.fc = nn.Linear(128 * 2, num_classes)  # 128 * 2 wegen Bidirectional GRU
        
    def forward(self, x):
        # here is defined, how the model produces outputs
        # Add one channel-dimension (Batch, Channel, Feature, Time)
        x = x.unsqueeze(1)
        
        # CNN Forward Pass
        x = self.cnn(x)
        
        # Print the shape of x after the CNN and before the GRU
        # print("Shape before permute and flatten:", x.shape)

        # Readjust dimensions for RNN (Batch, Time, Features)
        x = x.permute(0, 3, 1, 2)  # (Batch, Feature, Channel, Time) -> (Batch, Time, Feature, Channel)
        x = x.flatten(2)  # flattening apllied to last two dimensions
        
        # Print the shape of x after the CNN and before the GRU
        # print("Shape before GRU:", x.shape)

        # RNN Forward Pass
        x, _ = self.rnn(x)
        
        # Fully Connected Layer
        x = self.fc(x)

        # print("RNN output shape:", x.shape)

        return x

