# scripts/model.py

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

        
        # Recurrent Layer (z.B. GRU) zur Sequenzverarbeitung
        # adjustment of input size and hidden_size because of CNN form and more layers 32 * 40 / 64 * 64
        self.rnn = nn.GRU(input_size=32 * 40, hidden_size=128, num_layers=3, batch_first=True, bidirectional=True)
        
        # Voll verbundene Schicht zur Ausgabe
        self.fc = nn.Linear(128 * 2, num_classes)  # 128 * 2 wegen Bidirectional GRU
        
    def forward(self, x):
        # here is defined, how the model produces outputs
        # Füge eine Kanal-Dimension hinzu (Batch, Channel, Feature, Time)
        x = x.unsqueeze(1)
        
        # CNN Forward Pass
        x = self.cnn(x)
        
        # Print the shape of x after the CNN and before the GRU
        # print("Shape before permute and flatten:", x.shape)

        # Anpassung der Dimensionen für RNN (Batch, Time, Features)
        x = x.permute(0, 3, 1, 2)  # (Batch, Feature, Channel, Time) -> (Batch, Time, Feature, Channel)
        x = x.flatten(2)  # Verflachen der letzten beiden Dimensionen
        
        # Print the shape of x after the CNN and before the GRU
        # print("Shape before GRU:", x.shape)

        # RNN Forward Pass
        x, _ = self.rnn(x)
        
        # Fully Connected Layer
        x = self.fc(x)

        # print("RNN output shape:", x.shape)

        return x

