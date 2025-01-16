# SpeechRecognition/scripts/wav2vec/train_wa2vec.py

### imports
# general imports
import torch
torch.set_float32_matmul_precision("medium")
print(torch.version.cuda)

# CUDA exeption debugging
import os
# Enable CUDA launch blocking and device-side assertions for better debugging
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

import evaluate
from wav2vecModel import SpeechRecognition
from dataset import SpeechDataset, DataCollatorCTCWithPadding, HuggingFaceSaveCallback
from transformers import TrainingArguments
from datasets import load_dataset, Audio

# Imports for custom dataset
import os
import numpy as np
import torchaudio
from torch.utils.data import Dataset

# training function imports
import pytorch_lightning as pl
import random
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


# Logging
# from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers import WandbLogger

from config import MODEL_ID, OUTPUT_DIR, BATCH_SIZE, LEARNING_RATE, MAX_STEPS, WARMUP_STEPS, EVAL_STEPS, SAVE_STEPS, MAX_EPOCHS, DEVICE, processor, encode, decode


def prep_dataset():
    # load dataset
    minds = load_dataset("PolyAI/minds14", name="en-US", split="train[:1000]").train_test_split(test_size=0.2)
    
    # Remove unwanted columns but keep "transcription"
    minds = minds.remove_columns(["english_transcription", "intent_class", "lang_id"])
    
    minds = minds.cast_column("audio", Audio(sampling_rate=16000))
    return minds

def prep_batch(batch):
    audio = batch["audio"]
    transcription = batch["transcription"]
    
    # Print original transcription for debugging
    # print(f"Original transcription: {transcription}")
    
    # check vocab ob processor
    # print(processor.tokenizer.get_vocab())
    # output: {'<pad>': 0, '<s>': 1, '</s>': 2, '<unk>': 3, '|': 4, 'E': 5, 'T': 6, 'A': 7, 'O': 8, 'N': 9, 'I': 10, 'H': 11, 'S': 12, 'R': 13, 'D': 14, 'L': 15, 'U': 16, 'M': 17, 'W': 18, 'C': 19, 'F': 20, 'G': 21, 'Y': 22, 'P': 23, 'B': 24, 'V': 25, 'K': 26, "'": 27, 'X': 28, 'J': 29, 'Q': 30, 'Z': 31}
    # Thats why we need to adjust,
    # Preprocess transcription:
    # 1. Convert to uppercase
    # 2. Replace spaces with "|"
    # 3. Remove unsupported characters
    transcription = transcription.upper().replace(" ", "|")
    transcription = "".join(char for char in transcription if char in processor.tokenizer.get_vocab())
    
    # Debug: Print processed transcription
    # print(f"Processed transcription: {transcription}")
    labels = encode(transcription)
    # Wav2Vec2Processor: Generate inputs from audio and preprocessed transcription
    inputs = processor(audio["array"], sampling_rate=audio["sampling_rate"])
    inputs['labels'] = labels
    # Calculate and store input length for audio features
    inputs["input_length"] = len(inputs["input_values"][0])
    
    return inputs



# Initialize the Speech Recognition model using PyTorch Lightning
model = SpeechRecognition.load_from_checkpoint('/home/tuancuong/workspaces/carloberger/speech_recognition/scripts/checkpoints/model_epoch=3-val_loss=0.31.ckpt', LEARNING_RATE=LEARNING_RATE)

# Logging
# TensorBoard
# logger = TensorBoardLogger("speech_recognition/scripts/", name="lightning_logs")
# CSV Logger
csv_logger = CSVLogger("speech_recognition/scripts/", name="lightning_logs")
# Wandb Logger
wandb_logger = WandbLogger(project='Bachelor_Thesis_Berger', name='MozillaCommonVoiceFull', log_model='all')
wandb_logger.experiment.config["batch_size"] = BATCH_SIZE




# initialize and generate dataset
minds = prep_dataset()
print("Starting dataset preprocessing with .map()...")
# .map() does:
# current minds structure: {'train': ['path', 'audio', 'transcription'], 'test': ['path', 'audio', 'transcription']}
# 1: removing path
# 2: audio into input_values
# 3: transcription into Numbers and called labels
# 4: calculates input_length
minds = minds.map(prep_batch, remove_columns=minds.column_names["train"], num_proc=2, load_from_cache_file=True)
# minds = [prep_batch(batch) for batch in minds]
# new minds structure: # {'train': ['input_values', 'labels', 'input_length'], 'test': ['input_values', 'labels', 'input_length']}
print("Dataset preprocessing completed.")


def get_dataloaders(use_wav2vec_processor=True):
    """
    Returns train and validation DataLoaders based on the selected configuration.
    Args:
        use_wav2vec_processor (bool): If True, use Wav2Vec2Processor and DataCollatorCTCWithPadding.
                                      If False, use SpeechDataset and custom_collate.
    Returns:
        train_loader, val_loader: DataLoaders for training and validation.
    """
    if use_wav2vec_processor:
        # Using Wav2Vec2Processor and DataCollatorCTCWithPadding
        print("Using DataLoader with Wav2Vec2Processor and DataCollatorCTCWithPadding.")
        collator = DataCollatorCTCWithPadding(processor)

        train_loader = DataLoader(
            minds["train"],
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=collator,
            pin_memory=True,
        )

        val_loader = DataLoader(
            minds["test"],
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=collator,
            pin_memory=True,
        )
    else:
        # Using SpeechDataset and Wav2Vec2Processor (replacing custom_collate)
        print("Using DataLoader with Mozilla Common Voice DataSet and Wav2Vec2Processor (DataCollatorCTCWithPadding).")

        feature_dir = 'speech_recognition/outputs/train_audio'
        transcription_file = 'speech_recognition/outputs/train_audio/transcriptions.txt'
        test_feature_dir = 'speech_recognition/outputs/test_audio/'
        test_transcription_file = 'speech_recognition/outputs/test_audio/transcriptions.txt'

        # train_dataset = SpeechDataset(feature_dir, transcription_file, apply_specaugment=True, num_samples_per_epoch=196608)
        train_dataset = SpeechDataset(feature_dir, transcription_file)
        val_dataset = SpeechDataset(test_feature_dir, test_transcription_file)

        # Load dataclass from modelfile
        collator = DataCollatorCTCWithPadding(processor)


        train_loader = DataLoader(
            train_dataset,
            batch_size=4*BATCH_SIZE,
            shuffle=True,
            collate_fn=collator,
            num_workers=4,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=collator,
            num_workers=2,
            pin_memory=True,
        )

    return train_loader, val_loader



def train_model():
    # Definition of specific seed for all computing instances
    SEED = 42
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    """
    Trains the model using the specified DataLoader configuration.
    Args:
        use_minds_dataset (bool): 
        If True, it uses the Minds Dataset with its configuration setup. 
        If False, it uses custom integrated Mozilla CommonVoice Corpus 4.0 EN SpeechDataset and its configuration setup.
    """
    use_minds_dataset=False
    # Get DataLoaders based on the selected configuration
    print("Starting DataLoader initialization...")
    train_loader, val_loader = get_dataloaders(use_minds_dataset)
    print("DataLoaders initialized.")


    '''
    # checkpoint initialization without HuggingFace compatible Model save
    checkpoint_callback = ModelCheckpoint(
        dirpath= 'speech_recognition/scripts/checkpoints',
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        filename='model_{epoch}-{val_loss:.2f}',
        # verbose=True,
    )
    '''
    # Updated checkpoint initialization
    checkpoint_callback = HuggingFaceSaveCallback(
        dirpath='speech_recognition/scripts/checkpoints',
        monitor='val_wer',
        mode='min',
        save_top_k=1,
        filename='model_{epoch}-{val_loss:.2f}',
    )
    # Initialize EarlyStopping callback
    early_stopping_callback = EarlyStopping(
        monitor='val_wer',
        patience=10,
        mode='min',
        verbose=True
    )

    # Dynamically set the device and accelerator based on GPU availability
    if torch.cuda.is_available():
        accelerator = "gpu"
        print("GPU USED")
    else:
        accelerator = "cpu"
        print("CPU USED")

    # Set up PyTorch Lightning Trainer
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback, early_stopping_callback],
        devices=[1],
        max_epochs=MAX_EPOCHS,
        accumulate_grad_batches=2,
        accelerator=accelerator,
        log_every_n_steps=1,  # Log frequently enough but not overly granular
        check_val_every_n_epoch=1,  # Validate at the end of each epoch
        gradient_clip_val=0.3,
        gradient_clip_algorithm="norm",
        enable_progress_bar=True,
        logger=wandb_logger,
        fast_dev_run=False
    )

    # Train the model
    trainer.fit(model, train_loader, val_loader)

    # Evaluate on the validation set
    val_results = trainer.validate(model, val_loader)
    for metric in val_results:
        for key, value in metric.items():
            print(f"{key}: {value}")  # Optional: Keep for console output
            trainer.logger.log_metrics({key: value})



if __name__ == "__main__":
    torch.cuda.empty_cache()
    train_model()