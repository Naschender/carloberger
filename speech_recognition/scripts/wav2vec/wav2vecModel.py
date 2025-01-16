# SpeechRecognition/scripts/wav2vec/wav2vecModel.py
# author: carlo berger
# Topic: Speech Recognition Model in context of Bachelorthesis WS25
#        this model transforms raw audio in waveform format into vector format
#        the vector format then gets used for predicting the best path through calculation matrix

### imports
# general
import torch
import torch.nn as nn
import random
import numpy as np
import pytorch_lightning as pl
import soundfile as sf
from torch.optim.lr_scheduler import ReduceLROnPlateau

# from files
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from dataclasses import dataclass
from typing import Dict, List, Union
from dataset import SpeechDataset, DataCollatorCTCWithPadding

from config import MODEL_ID, OUTPUT_DIR, BATCH_SIZE, LEARNING_RATE, MAX_STEPS, WARMUP_STEPS, EVAL_STEPS, SAVE_STEPS, MAX_EPOCHS, DEVICE, processor, encode, decode

from metrics import compute_metrics

# Set the seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


# check status of computing unit
print(f"Using device: {DEVICE}")


# initialize the model
class SpeechRecognition(pl.LightningModule):
    def __init__(self, LEARNING_RATE):
        super().__init__()
        self.processor = processor
        self.model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID).to(DEVICE)
        self.learning_rate = LEARNING_RATE
        self.validation_outputs = {"preds": [], "labels": []}
        self.ctc_loss_fn = nn.CTCLoss(blank=self.processor.tokenizer.pad_token_id, reduction="mean", zero_infinity=True)
        
        # save hyper-parameters to self.hparams (auto-logged by W&B)
        self.save_hyperparameters()

    def forward(self, inputs, attention_mask=None):
        return self.model(inputs, attention_mask=attention_mask).logits


    def training_step(self, batch, batch_idx):
        inputs = batch["input_values"].to(self.device)
        attention_mask = batch.get("attention_mask", None)
        labels = batch["labels"].to(self.device)

        # Forward pass
        outputs = self(inputs, attention_mask)
        log_probs = outputs.log_softmax(-1).to(inputs.device)

        # Check for NaN/Inf in outputs
        if torch.isnan(log_probs).any() or torch.isinf(log_probs).any():
            print("Error: NaN or Inf detected in log_probs during training_step.")
            print(f"log_probs shape: {log_probs.shape}")
            return float('inf')

        # Compute input_lengths and target_lengths
        input_lengths = torch.full(size=(outputs.size(0),), fill_value=outputs.size(1), dtype=torch.long).to(inputs.device)
        target_lengths = torch.sum(labels != -100, dim=1).to(inputs.device)

        ### Debug print statements
        # print("Training Step Debug:")
        # print(f"outputs shape: {outputs.shape}")
        # print(f"Logits shape: {log_probs.shape}")
        # print(f"Logits sample: {log_probs[0, :10, :10]}")  # Print the first 10 timesteps and classes
        # print(f"log_probs content: {log_probs}")
        # print(f"Input lengths: {input_lengths}")
        # print(f"Target lengths: {target_lengths}")
        # print(f"Labels shape: {labels.shape}")
        # Check outputs distribution
        # print(f"outputs min: {log_probs.min()}, max: {log_probs.max()}, mean: {log_probs.mean()}")
        
         # Check for inconsistencies
        if target_lengths.max() > input_lengths.max():
            print("Error: Target lengths exceed input lengths.")
            return float('inf')

        # Compute CTC loss
        try:
            loss = self.ctc_loss_fn(log_probs.transpose(0, 1), labels, input_lengths, target_lengths)
            self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        except RuntimeError as e:
            print(f"RuntimeError during CTC loss computation in training_step: {e}")
            return float('inf')

        return loss


    def validation_step(self, batch, batch_idx):
        with torch.no_grad():  # Disable gradient computations
            inputs = batch["input_values"].to(self.device)
            attention_mask = batch.get("attention_mask", None)
            labels = batch["labels"].to(self.device)

            # Forward pass
            outputs = self(inputs, attention_mask)
            log_probs = outputs.log_softmax(dim=-1).to(inputs.device)

            # Check for NaN/Inf in outputs
            if torch.isnan(log_probs).any() or torch.isinf(log_probs).any():
                print("Error: NaN or Inf detected in log_probs during validation_step.")
                print(f"log_probs shape: {log_probs.shape}")
                return float('inf')

            # Compute input_lengths and target_lengths
            input_lengths = torch.full(size=(outputs.size(0),), fill_value=outputs.size(1), dtype=torch.long).to(inputs.device)
            target_lengths = torch.sum(labels != -100, dim=1).to(inputs.device)

            ### Debug print statements
            # print("Validation Step Debug:")
            # print(f"outputs shape: {outputs.shape}")
            # Debugging raw logits
            # print(f"Log-Probabilities shape: {log_probs.shape}")
            # print(f"Log-Probabilities sample: {log_probs[0, :10, :10]}")
            # print(f"log_probs content: {log_probs}")
            # print(f"Input lengths: {input_lengths}")
            # print(f"Target lengths: {target_lengths}")
            # print(f"Labels shape: {labels.shape}")
            # Check outputs distribution
            # print(f"outputs min: {log_probs.min()}, max: {log_probs.max()}, mean: {log_probs.mean()}")
            
            # Check for inconsistencies
            if target_lengths.max() > input_lengths.max():
                print("Error: Target lengths exceed input lengths.")
                return float('inf')

            # Decode predictions and labels
            preds = torch.argmax(log_probs, dim=-1)
            
            ### Debugging predictions
            # print(f"Predictions (raw logits argmax): {preds}")
            # print(f"Predictions shape: {preds.shape}")
            # print(f"Target lengths: {target_lengths}")
            
            # Ensure preds and labels match in size before accumulating
            assert preds.shape[0] == labels.shape[0], "Mismatch in batch sizes of predictions and labels"
    
            # Truncate predictions to target lengths
            # Use min to ensure truncation does not go out of bounds
            preds_trimmed = [
                pred[:min(target_lengths[i], pred.shape[0])] for i, pred in enumerate(preds)
            ]


            # Decode predictions and labels
            decoded_predictions = [
                self.processor.batch_decode([pred.tolist()], skip_special_tokens=True)[0]
                for pred in preds_trimmed
            ]

            decoded_labels = []
            for label in labels.tolist():
                cleaned_label = [id for id in label if id != -100]
                decoded_labels.append(decode(cleaned_label))
            
            # Debugging decoded outputs
            print(f"Decoded Predictions: {decoded_predictions[:5]}")
            print(f"Decoded Labels: {decoded_labels[:5]}")

            # Debugging mismatches
            if len(decoded_predictions) != len(decoded_labels):
                print(f"Skipping batch: Preds: {len(decoded_predictions)}, Labels: {len(decoded_labels)}")
                return

            # Add to validation outputs
            self.validation_outputs["preds"].extend(decoded_predictions)
            self.validation_outputs["labels"].extend(decoded_labels)

            # Debugging size
            # print(f"Validation Step - Batch {batch_idx}:")
            # print(f"Predictions shape: {preds.shape}, Labels shape: {labels.shape}")

            # Compute CTC loss
            try:
                loss = self.ctc_loss_fn(log_probs.transpose(0, 1), labels, input_lengths, target_lengths)
                self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            except RuntimeError as e:
                print(f"RuntimeError during CTC loss computation in validation_step: {e}")
                return float('inf')

            return loss

    def on_validation_epoch_end(self):
        preds = self.validation_outputs["preds"]
        labels = self.validation_outputs["labels"]

        # Ensure the lengths match
        if len(preds) != len(labels):
            print(f"Error: Mismatch in predictions ({len(preds)}) and labels ({len(labels)}) during aggregation")
            return

        outputs = [{"preds": pred, "labels": label} for pred, label in zip(preds, labels)]

        # Compute WER
        try:
            wer = compute_metrics(outputs)
            self.log("val_wer", wer, prog_bar=True, on_epoch=True, logger=True)
        except ValueError as e:
            print(f"Error computing WER: {e}")

        # Clear validation outputs for the next epoch
        self.validation_outputs = {"preds": [], "labels": []}
        

        avg_val_loss = self.trainer.callback_metrics.get("val_loss", None)
        if avg_val_loss is not None:
            self.log("val_loss_epoch", avg_val_loss, logger=True)


    def configure_optimizers(self):
        ### Adam Optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
    


        ### learning rate scheduler
        # schedule on plateau (old values factor=0.1, patience=5)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=1)
        '''
        ### schedule on scheduler CyclicLR (Batch size of 4 compared with dataset around 2.500 results in 625 iterations)
        # good for escaping local minima (periodically increasing learning rate)
        # scheduler = CyclicLR(optimizer, base_lr=1e-4, max_lr=LEARNING_RATE, step_size_up=300, mode="exp_range") # vllt auch mal linear??
         
        
        ### Schedule on Scheduler One-Cycle-LR
        # Good on generalization (linear approach)
        # Get the total number of optimizer steps per epoch
        total_steps = self.trainer.estimated_stepping_batches
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=1e-4,
            steps_per_epoch=total_steps,
            epochs=EPOCHS,
            anneal_strategy='linear'
        )
        
    '''    
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'        # monitor is needed with schedulers, means that it will reduce the learning rate based on the validation loss not the training loss 
        }