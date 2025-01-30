"""
===============================================================================
    Project:        JarvisAI Bachelorthesis
    File:           data.py
    Description:    This script contains the Mel-Spectrogram-model 
                    custom collate function, vocab function the old 
                    and the new dataset class
    Author:         Carlo Berger, Aalen University
    Email:          Carlo.Berger@studmail.htw-aalen.de
    Created:        2024-10-30
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

from torch.utils.data import Dataset
import os
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import torchaudio
import torchaudio.transforms as T


def custom_collate(data): #(2)
    inputs = [d[0] for d in data] #(3)
    # input_lens for only on conv2d layer
    # input_lens = torch.from_numpy(np.array([len(d[0]) // 2 for d in data])) # max pool (2) -> reduce width and height for 2 two times so 2x2
    # input_lens for 2^2 Time Dimension Reduction so  2x MaxPool2d(2) 
    input_lens = torch.from_numpy(np.array([len(d[0]) // 4 for d in data])) # max pool (2) -> reduce width and height for 2 two times so 2x2
    labels = [d[1] for d in data]
    label_lens = torch.from_numpy(np.array([len(d[1]) for d in data]))

    inputs = pad_sequence(inputs, batch_first=True) #(4)
    labels = pad_sequence(labels, batch_first=True, padding_value=vocab.word2index['<sp>'])

    return inputs, labels, input_lens, label_lens


class Vocab(object):
    def __init__(self, vocab_file=None):
        self.word2index = {}
        self.index2word = {}

        if vocab_file:
            self.load_vocab(vocab_file)
    
    def load_vocab(self, vocab_file):
        # load vocab from file
        with open(vocab_file, 'r') as f:
            for i, line in enumerate(f):
                word = line.strip()
                self.word2index[word] = i
                self.index2word[i] = word
        # add blank word
        self.word2index['<blank>'] = len(self.word2index)
        self.index2word[self.word2index['<blank>']] = '<blank>'
    def __len__(self):
        return len(self.index2word)
    def encode(self, text):
        out = []
        for c in text:
            if c == ' ':
                out.append(vocab.word2index['<sp>'])
            else:
                out.append(vocab.word2index[c])
        return out


vocab = Vocab('vocab.txt')


# # Eigener Dataset-Klasse zur Verarbeitung der Daten
# class SpeechDataset(Dataset):
#     def __init__(self, feature_dir, transcription_file):
#         self.feature_files = [os.path.join(feature_dir, f) for f in os.listdir(feature_dir) if f.endswith('.pt')]
#         self.transcriptions = self.load_transcriptions(transcription_file)
        
#     def load_transcriptions(self, file_path):
#         transcriptions = {}
#         with open(file_path, 'r') as f:
#             for line in f:
#                 parts = line.strip().split('\t')
#                 if len(parts) == 2:
#                     id = os.path.basename(parts[0])[:-4]
#                     transcriptions[id] = parts[1]
#         return transcriptions

#     def __len__(self):
#         return len(self.feature_files)

#     def __getitem__(self, idx):
#         feature_file = self.feature_files[idx]
#         features = torch.load(feature_file, weights_only=True)
#         id = os.path.basename(feature_file)[:-3]
#         transcription = self.transcriptions.get(id, "")
#         transcription_idx = vocab.encode(transcription)
#         return features.squeeze(0).transpose(1,0), torch.from_numpy(np.array(transcription_idx))


# Eigener Dataset-Klasse zur Verarbeitung der Daten
# Change: load audio file -> process audio file
class SpeechDataset(Dataset):
    def __init__(self, audio_dir, transcription_file, apply_specaugment=False):
        self.audio_files = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith('.wav')]
        self.transcriptions = self.load_transcriptions(transcription_file)
        self.apply_specaugment = apply_specaugment
        self.specaugment = T.SpecAugment(
            freq_mask_param=30,    # Adjustable parameter
            time_mask_param=45,    # Adjustable parameter
            n_time_masks=1,        # Number of time masks to apply
            n_freq_masks=1         # Number of frequency masks to apply
        )
        
    def load_transcriptions(self, file_path):
        transcriptions = {}
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    id = os.path.basename(parts[0])[:-4]
                    transcriptions[id] = parts[1]
        return transcriptions

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_file = self.audio_files[idx]
        SPEECH_WAVEFORM, SAMPLE_RATE = torchaudio.load(audio_file)
        features = torchaudio.compliance.kaldi.fbank(SPEECH_WAVEFORM, num_mel_bins=80, snip_edges=False)

        # Apply SpecAugment only to training samples if apply_specaugment is True
        if self.apply_specaugment:
            features = self.specaugment(features)

        id = os.path.basename(audio_file)[:-4]
        transcription = self.transcriptions.get(id, "")
        transcription_idx = vocab.encode(transcription)
        return features, torch.from_numpy(np.array(transcription_idx))
