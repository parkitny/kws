import logging
import sys
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import RichProgressBar, ModelCheckpoint
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from src import data
from src import constants as C
from src import configuration
from src.model import ResNetMod

file_handler = logging.FileHandler(filename='train.log')
stdout_handler = logging.StreamHandler(stream=sys.stdout)
handlers = [file_handler, stdout_handler]

logging.basicConfig(
    level=logging.DEBUG, 
    format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
    handlers=handlers
)

log = logging.getLogger(__file__)

def train():
    params = configuration.get_params()

    train_dl, val_dl, test_dl, labels = data.get_loaders(params.data)
    n_classes = len(labels)
    # Inject number of classes into the config, lets us train with other datasets if needed.
    params.model.n_classes = n_classes
    params.model.labels = labels # Used for confusion matrix.
    model = ResNetMod(params.model)

    accelerator = C.CPU
    if params.train.device == C.CUDA:
        accelerator = C.GPU
        
    early_stop = EarlyStopping(monitor="val_acc", min_delta=0.001, patience=10, verbose=False, mode="max")
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        filename='kws-res-epoch{epoch:02d}-val_acc{val_acc:.2f}',
        auto_insert_metric_name=False,
        save_top_k=3
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
        accelerator=accelerator, # use one GPU if available
        max_epochs=params.train.max_epochs,
        callbacks=[progress_bar, 
                   early_stop, 
                   checkpoint_callback],
    )
    trainer.fit(model,
                train_dataloaders=train_dl,
                val_dataloaders=val_dl,)
    trainer.test(model, test_dl)

    
if __name__ == "__main__":
    # Avoid issues with num_workers > 1 in dataloader:
    # https://github.com/pytorch/pytorch/issues/40403
    torch.multiprocessing.set_start_method('forkserver', force=True)
    train()
