import torch
import torchmetrics as tm
import torchmetrics.classification as tmc
import pytorch_lightning as pl
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

from torch import nn
from torchvision import models
from pathlib import Path

from src import constants as C

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
            case C.RESNEXT50:
                self.model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1) 
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
            nn.Dropout(p=self.conf.dropout),
            nn.LeakyReLU(),
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

    def on_train_start(self):
        # Copy the parameters for this run.
        log_fp = Path(self.logger.log_dir)
        log_fp.mkdir(parents=True, exist_ok=True)
        from_param_fp = Path(C.PARAMS_FN)
        to_param_fp = log_fp / C.PARAMS_FN
        to_param_fp.touch()
        to_param_fp.write_text(from_param_fp.read_text())
        print(self.logger.log_dir)

  
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

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        test_acc = self.acc(logits, y)
        self.log('test_acc', test_acc, prog_bar=True,
            on_step=False,
            on_epoch=True)
        self.log('test_loss', loss, prog_bar=True,
            on_step=True,
            on_epoch=True)
        metrics = {
            "test_acc": test_acc,
            "test_loss": loss,
            "output": logits,
            "target": y,
        }

        return metrics
    
    @torch.no_grad()
    def test_epoch_end(self, outputs):
        # Outputs contain the outputs of all test steps. Can use this to
        # for example calculate metrics across entire set.
        _outputs = [
            {
                metric: metric_val
                for metric, metric_val in o.items()
            }
            for o in outputs
        ]
        logits = torch.concat([o["output"] for o in _outputs], axis=0)
        probs = torch.nn.Softmax(dim=1)(logits).to('cpu')

        # NOTE: We can obtain the actual class prediction by applying argmax to
        # the probabilities
        target = torch.concat([o["target"] for o in _outputs]).to('cpu')

        metrics = {
            'test_acc': tm.Accuracy(task='multiclass', num_classes=probs.shape[1])(probs, target),
            'test_f1_score': tmc.MulticlassF1Score(task='multiclass', num_classes=probs.shape[1])(probs, target)
        }
        # Plot the confusion matrix and save it as a figure.
        confmat = tmc.MulticlassConfusionMatrix(task='multiclass', num_classes=probs.shape[1])(probs, target)
        df_cm = pd.DataFrame(confmat, index =self.conf.labels,
                  columns=self.conf.labels)
        self.conf.labels
        plt.figure(figsize = (15,15))
        sn.heatmap(df_cm, annot=True)
        plt.savefig('confmat.png')
        # To get a print out after eval, need to log the metrics.
        for metric, metric_val in metrics.items():
            self.log(
                metric,
                metric_val
            )

        return metrics


  
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