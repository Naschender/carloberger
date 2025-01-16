# scripts/wav2vec/dataset.py
# author: Carlo Berger

# imports dataset.py
import os
import random
import torch
import torchaudio
from torch.utils.data import Dataset
from config import processor, encode, decode
from dataclasses import dataclass
from typing import Dict, List, Union
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from pytorch_lightning.callbacks import ModelCheckpoint


# Set the seed for reproducibility
seed = 42
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

'''
class SpeechDataset(Dataset):
    def __init__(self, audio_dir, transcription_file, apply_specaugment=False, num_samples_per_epoch=10000):
        """
        :param audio_dir: Directory containing audio files.
        :param transcription_file: Path to the transcription file.
        :param apply_specaugment: Whether to apply SpecAugment.
        :param num_samples_per_epoch: Number of random files to sample per epoch.
        """
        self.audio_dir = audio_dir
        self.transcriptions = self.load_transcriptions(transcription_file, processor)
        self.apply_specaugment = apply_specaugment
        self.num_samples_per_epoch = num_samples_per_epoch
        self.audio_files = [
            f for f in os.listdir(audio_dir) if f.endswith('.wav')
        ]

    def load_transcriptions(self, file_path, processor):
        transcriptions = {}
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    id = os.path.basename(parts[0])[:-4]
                    transcription = parts[1]

                    # Preprocess transcription:
                    transcription = transcription.upper().replace(" ", "|")
                    transcription = "".join(
                        char for char in transcription if char in processor.tokenizer.get_vocab()
                    )
                    transcriptions[id] = transcription
        return transcriptions

    def __getitem__(self, idx):
        # Randomly select a file each time __getitem__ is called
        random_file = random.choice(self.audio_files)
        audio_file_path = os.path.join(self.audio_dir, random_file)

        # Load audio
        speech_waveform, sample_rate = torchaudio.load(audio_file_path)

        # Preprocess transcription
        file_id = os.path.basename(random_file)[:-4]
        transcription = self.transcriptions.get(file_id, "")
        transcription_idx = encode(transcription)
        transcription_test = decode(transcription_idx)
        # Return data in the required dictionary format
        return {
            "input_values": speech_waveform.numpy(),  # Use NumPy array for compatibility with processor
            "labels": transcription_idx,
        }

    def __len__(self):
        # Define dataset length as the number of samples per epoch
        return self.num_samples_per_epoch

'''
class SpeechDataset(Dataset):
    def __init__(self, audio_dir, transcription_file):
        self.audio_files = [
            os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith('.wav')
        ]
        self.transcriptions = self.load_transcriptions(transcription_file, processor)
        # self.apply_specaugment = apply_specaugment

    def load_transcriptions(self, file_path, processor):
        transcriptions = {}
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    id = os.path.basename(parts[0])[:-4]
                    transcription = parts[1]

                    # Preprocess transcription:
                    transcription = transcription.upper().replace(" ", "|")
                    transcription = "".join(
                        char for char in transcription if char in processor.tokenizer.get_vocab()
                    )
                    transcriptions[id] = transcription
        return transcriptions

    def __getitem__(self, idx):
        # Load audio
        audio_file = self.audio_files[idx]
        speech_waveform, sample_rate = torchaudio.load(audio_file)

        # Preprocess transcription
        file_id = os.path.basename(audio_file)[:-4]
        transcription = self.transcriptions.get(file_id, "")
        transcription_idx = encode(transcription)
        # Debugging
        transcription_test = decode(transcription_idx)
        # Return data in the required dictionary format
        return {
            "input_values": speech_waveform.numpy(),  # Use NumPy array for compatibility with processor
            "labels": transcription_idx,
        }

    def __len__(self):
        return len(self.audio_files)



@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: str = "longest"

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"][0]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # Pad input features using the processor
        batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")

        # Manually pad the labels
        labels = [feature["input_ids"] for feature in label_features]
        max_label_length = max(len(label) for label in labels)

        padded_labels = []
        for label in labels:
            padded_label = label + [-100] * (max_label_length - len(label))
            padded_labels.append(padded_label)

        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)

        # Add attention_mask if it exists, otherwise create a default mask
        if "attention_mask" not in batch:
            batch["attention_mask"] = torch.ones(batch["input_values"].shape, dtype=torch.long)

        # Debug prints
        # print(features)  # Print out features to check its structure
        # print(features[0])  # Print the first feature to see its structure

        return batch



class HuggingFaceSaveCallback(ModelCheckpoint):
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        # Call the default save logic
        super().on_save_checkpoint(trainer, pl_module, checkpoint)
        
        # Save in Hugging Face format
        if self.best_model_path:  # Ensure the current model is the best
            hf_save_dir = os.path.join(
                os.path.dirname(self.best_model_path), "huggingface_model"
            )
            
            # Map the PyTorch Lightning model state_dict to Hugging Face model
            hf_model = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-base-960h')
            new_state_dict = {
                k.replace('model.', ''): v
                for k, v in pl_module.state_dict().items()
                if k.startswith('model.')
            }
            hf_model.load_state_dict(new_state_dict)
            
            # Save the Hugging Face model
            hf_model.save_pretrained(hf_save_dir)
            print(f"Hugging Face model saved to {hf_save_dir}")