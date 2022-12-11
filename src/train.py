import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import RichProgressBar, ModelCheckpoint
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torchvision import models

from pathlib import Path

import yaml
import munch
import torchmetrics as tm
from src import data
from src import constants as C
from src import configuration

class ResNetMod(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.conf = config
        self.acc = tm.Accuracy(task='multiclass', num_classes=config.n_classes)

        # define model and loss
        match config.resnet_variant.lower():
            case C.RESNET18:
                self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            case C.RESNET50:
                self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            case _:
                raise NotImplementedError(f'The variant {config.resnet_variant}hasn\'t been implemented')

        _first_conv_weights = self.model.conv1.weight
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # Going from 3D to 1D - mean original weight.
        # https://datascience.stackexchange.com/questions/65783/pytorch-how-to-use-pytorch-pretrained-for-single-channel-image
        # The alternative is to duplicate the single channel data to 3 channels,
        # but that consumes more memory
        self.model.conv1.weight = nn.Parameter(
            torch.mean(_first_conv_weights, dim=1, keepdim=True)
        )
        
        # Freeze entire backbone by default
        for param in self.model.parameters():
            param.requires_grad = False
            
        # model.parameters() are sequential from the input so
        # we want to iterate in reverse. Cast the generator to
        # a list and reverse what is needed.
        for param in self._get_params_reversed(self.conf.unfreeze_layers):
            param.requires_grad = True
                
        self.loss = nn.CrossEntropyLoss()
        n_inputs = self.model.fc.in_features
        top_layers = [
            nn.Linear(in_features=n_inputs, out_features=512),
            nn.LeakyReLU(),
            nn.Dropout(p=self.conf.dropout),
            nn.Linear(in_features=512, out_features=256),
            nn.LeakyReLU(),
            nn.Linear(in_features=256, out_features=self.conf.n_classes), # TODO: Get this n_classes from dataset.
        ]
        self.model.fc = nn.Sequential(
            *top_layers
        )
    
    
    def _get_params_reversed(self, num_layers):
        reversed_layers = list(reversed(list(self.model.parameters())))
        if num_layers < 0:
            return reversed_layers
        return reversed_layers[:num_layers]
    
    #@auto_move_data # this decorator automatically handles moving your tensors to GPU if required
    def forward(self, x):
        return self.model(x)
  
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        self.log('train_acc', self.acc(logits, y), prog_bar=True,
            on_step=True,
            on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        self.log('val_acc', self.acc(logits, y), prog_bar=True,
            on_step=False,
            on_epoch=True)
        self.log('val_loss', loss, prog_bar=True,
            on_step=True,
            on_epoch=True)
        return loss
  
    def configure_optimizers(self):
        opt_name = list(self.conf.optimiser)[0]
        opt_config = self.conf.optimiser[opt_name]
        match opt_name.lower():
            case 'sgd':
                return torch.optim.SGD(self.parameters(), **vars(opt_config))
            case 'rmsprop':
                return torch.optim.RMSprop(self.parameters(), **vars(opt_config))
            case _:
                raise NotImplementedError(f'The variant {self.conf.optimiser}hasn\'t been implemented')

def train():
    params = configuration.get_params()

    model = ResNetMod(params.model)
    train_dl, val_dl, test_dl = data.get_loaders(params.data)

    accelerator = C.CPU
    if params.train.device == C.CUDA:
        accelerator = C.GPU
        
    early_stop = EarlyStopping(
        monitor="val_acc", 
        patience=20, 
        verbose=False, 
        mode="max")
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        filename='lc-challenge-epoch{epoch:02d}-val_acc{val_acc:.2f}',
        auto_insert_metric_name=False
    )
    progress_bar = RichProgressBar(
    theme=RichProgressBarTheme(
            description="purple",
            progress_bar="green1",
            progress_bar_finished="green1",
            progress_bar_pulse="#6206E0",
            batch_progress="red",
            time="purple",
            processing_speed="purple",
            metrics="purple",
        )
    )
    trainer = pl.Trainer(
        accelerator=accelerator, # use one GPU
        max_epochs=params.train.max_epochs, # set number of epochs
        callbacks=[progress_bar, 
                   #early_stop, 
                   checkpoint_callback],
    )
    trainer.fit(model,
                train_dataloaders=train_dl,
                val_dataloaders=val_dl,)

    
if __name__ == "__main__":
    torch.multiprocessing.set_start_method('forkserver', force=True)# https://github.com/pytorch/pytorch/issues/40403
    # For colab use: https://inside-machinelearning.com/en/how-to-install-use-conda-on-google-colab/
    train()