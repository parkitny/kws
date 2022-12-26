import argparse
import logging
import sys
import pytorch_lightning as pl
from pathlib import Path
from src import data
from src import constants as C
from src import configuration
from src.model import ResNetMod

file_handler = logging.FileHandler(filename='eval.log')
stdout_handler = logging.StreamHandler(stream=sys.stdout)
handlers = [file_handler, stdout_handler]

logging.basicConfig(
    level=logging.INFO, 
    format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
    handlers=handlers
)



log = logging.getLogger(__file__)

def get_best_model(base_path):
    # Find the last version
    last_version = None
    last_version_path = None
    for subdir in Path(base_path).glob('*'):
        version = int(subdir.name.rsplit('_')[-1])
        if last_version is None or version > last_version:
            last_version = version
            last_version_path = subdir
            
    # Get the checkpoint
    best_acc = None
    best_path = None
    for ckpt_path in Path(last_version_path /"checkpoints").glob('*.ckpt'):
        val_acc = float(ckpt_path.stem.rsplit('val_acc')[-1])
        if best_acc is None or val_acc > best_acc:
            best_acc = val_acc
            best_path = ckpt_path
    return best_path



def evaluate(ckpt_path):
    params = configuration.get_params()

    _, _, test_dl, labels = data.get_loaders(params.data)
    n_classes = len(labels)
    # Inject number of classes into the config, lets us train with other datasets if needed.
    params.model.n_classes = n_classes
    params.model.labels = labels # Used for confusion matrix.
    # Re-create the model with updated params but weights from last run.
    model = ResNetMod(params.model)
    accelerator = C.CPU
    if params.train.device == C.CUDA:
        accelerator = C.GPU
    if ckpt_path is None:
        try:
            ckpt_path = get_best_model('lightning_logs')
        except AttributeError as e:
            log.error('Unable to find the latest checkpoint, ensure one exists for the last version in lightning logs.')
    trainer = pl.Trainer(
        accelerator=accelerator,
        logger=False
    )
    trainer.test(model, test_dl, ckpt_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--ckpt_path', nargs='?', type=str, default=None,
        help="Path to checkpoint for eval, if none - it tries to find latest for last experiment.")

    args = parser.parse_args()
    evaluate(args.ckpt_path)
