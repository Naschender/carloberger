from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from model import SpeechRecognitionModel
from data import SpeechDataset, custom_collate, vocab
import pytorch_lightning as pl
import random
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import CyclicLR

# Hyperparameters
# Define Hyperparameters
EPOCHS = 100
LEARNING_RATE = 0.0002 # Adam learning rate 0,001 
NUM_CLASSES = len(vocab) # total of 29: 26 Letters + Blank <sp> (Leerzeichen), Apostroph ', End of Sentence
WEIGHT_DECAY=1e-4

# pytorch_lightning model for aboved defined model
class LitSpeechRecognitionModel(pl.LightningModule):
    def __init__(self, num_classes, criterion):
        super().__init__()
        self.model = SpeechRecognitionModel(num_classes)
        self.criterion = criterion

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets, input_lens, output_lens = batch
        outputs = self(inputs.permute(0, 2, 1))
        outputs = torch.clamp(outputs, min=-1e10, max=1e10).log_softmax(-1)
        loss = self.criterion(outputs.transpose(1, 0), targets, input_lens, output_lens)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets, input_lens, output_lens = batch
        outputs = self(inputs.permute(0, 2, 1))
        outputs = torch.clamp(outputs, min=-1e10, max=1e10).log_softmax(-1)
        loss = self.criterion(outputs.transpose(1, 0), targets, input_lens, output_lens)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        ### Adam Optimizer
        optimizer = Adam(self.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)


        ### learning rate scheduler
        # schedule on plateau (old values factor=0.1, patience=5)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.4, patience=4)

        ### schedule on scheduler CyclicLR (Batch size of 4 compared with dataset around 2.500 results in 625 iterations)
        # good for escaping local minima (periodically increasing learning rate)
        # scheduler = CyclicLR(optimizer, base_lr=1e-4, max_lr=LEARNING_RATE, step_size_up=300, mode="exp_range") # vllt auch mal linear??
         
        '''
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

    def train_dataloader(self):
        feature_dir = '../outputs/processed_audio/'
        transcription_file = '../outputs/processed_audio/transcriptions.txt'
        train_dataset = SpeechDataset(feature_dir, transcription_file, apply_specaugment=True)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=custom_collate, num_workers=2) 
        return train_loader
    
    def val_dataloader(self):
        test_feature_dir = '../outputs/test_audio/'
        test_transcription_file = '../outputs/test_audio/transcriptions.txt'
        test_dataset = SpeechDataset(test_feature_dir, test_transcription_file, apply_specaugment=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=custom_collate, num_workers=2)
        return test_loader

# train with trainer class
def train():
    # Definition of specific seed for all computing instances
    SEED = 42
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Initialization of a new Model
    model = LitSpeechRecognitionModel(num_classes=NUM_CLASSES, criterion=nn.CTCLoss(blank=NUM_CLASSES - 1))

    # Initialization of the Model from a specific Checkpoint
    # model = LitSpeechRecognitionModel.load_from_checkpoint('checkpoints/model_epoch=80-val_loss=1.21.ckpt', num_classes=NUM_CLASSES, criterion=nn.CTCLoss(blank=28))
    
    # Initialization of the trained model train_model.py
    # model.model.load_state_dict(torch.load('best_model.pt', weights_only=True))

    # Definition of the trainer
    trainer = pl.Trainer(callbacks=[pl.callbacks.ModelCheckpoint(dirpath='checkpoints', monitor='val_loss', mode='min', save_top_k=1, filename='model_{epoch}-{val_loss:.2f}'), 
                                    pl.callbacks.EarlyStopping(monitor='val_loss', patience=10)], 
                         devices=1, max_epochs=EPOCHS, gradient_clip_val=1.2, gradient_clip_algorithm="norm",
                         fast_dev_run=False)
    
    # Call of definition above
    trainer.fit(model)

if __name__ == '__main__':
    train()