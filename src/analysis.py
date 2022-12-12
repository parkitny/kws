import yaml
import torch

from pathlib import Path
from tqdm import tqdm
from src import data
from src import configuration
from src.model import ResNetMod
import src.constants as C

# Train dataloader is used to set the model number of classes.
# Alternatively, this can be specified in the param.yaml under train.n_classes
def get_model(train_dataset, 
              params, 
              ckpt_path='lightning_logs/version_1/checkpoints/lc-challenge-epoch10-val_acc0.92.ckpt'):
    labels = sorted(list(set(datapoint[2] for datapoint in train_dataset)))
    n_classes = len(labels)
    params.model.n_classes = n_classes
    with open(Path(ckpt_path).parent.parent /'hparams.yaml') as f:
        args= yaml.safe_load(f)
    model = ResNetMod.load_from_checkpoint(checkpoint_path=ckpt_path, args=args, config=params.model).to('cuda')
    return model
    
    
def get_incorrect_predictions(expected, 
                              predicted, 
                              split='test', 
                              ckpt_path='lightning_logs/version_1/checkpoints/lc-challenge-epoch10-val_acc0.92.ckpt',
                              params_fn='params.yaml',
                              data_path=C.DATA_PATH):

    print('Loading data and checkpoint...')
    params = configuration.get_params(params_fn)
    train_dl, val_dl, test_dl, labels = data.get_loaders(data_params=params.data, data_path=data_path)
    match split:
        case 'training':
            dl = train_dl
        case 'validation':
            dl = val_dl
        case 'test':
            dl = test_dl
        case _:
            raise ValueError('Invalid set, choose from training, validation, test')


    labels_to_path = []
    model = get_model(train_dl.dataset, params, ckpt_path=ckpt_path)
    model.eval()
    print('Loaded!')
    with torch.no_grad():
        for batch_idx, (x, y) in tqdm(enumerate(dl), total=len(dl)):
            logits = model(x.to('cuda'))
            probs = torch.nn.Softmax(dim=1)(logits)
            preds = torch.argmax(probs, axis=1)
            for sample_idx in range(x.shape[0]):
                total_sample_idx = batch_idx * params.train.batch_size + sample_idx
                fp, _, l, *_ = dl.dataset.get_metadata(total_sample_idx)
                if (
                    labels.index(l) == labels.index(expected) and 
                    preds[sample_idx] == labels.index(predicted)
                    ):
                    labels_to_path.append(fp)

    return labels_to_path