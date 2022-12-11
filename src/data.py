from torchaudio.datasets import SPEECHCOMMANDS
import torchaudio.transforms as AT
from torchvision.transforms import transforms as VT
import os
import torch
import torchaudio
from functools import partial
from logging import getLogger
from tqdm import tqdm
from pathlib import Path

from src.configuration import get_params
import src.constants as C
log =  getLogger(__file__)

# TODO: Use the huggingface SC dataset.
class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str = None):
        super().__init__(C.DATA_PATH, download=True)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.normpath(os.path.join(self._path, line.strip())) for line in fileobj]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]


def augmentation_transform(img_size, mean, std):
    return VT.Compose(
        [
            VT.Resize([img_size, img_size], interpolation=VT.InterpolationMode.NEAREST),
            AT.TimeMasking(time_mask_param=int(img_size/10), p=0.5),
            AT.FrequencyMasking(freq_mask_param=int(img_size/10)),
            VT.Normalize([mean], [std]),
        ])
    
def resize_transform(img_size, mean, std):
    return VT.Compose(
        [
            VT.Resize([img_size, img_size], interpolation=VT.InterpolationMode.NEAREST),
            VT.Normalize([mean], [std]),
        ])
    
import numpy as np
# Noise injection: https://medium.com/@makcedward/data-augmentation-for-audio-76912b01fdf6
def add_noise(data, noise_factor):
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
    # Cast back to same data type
    augmented_data = augmented_data.astype(type(data[0]))
    return augmented_data
        
        
def preprocess_sample(waveform, sample_rate, n_fft, hop_length, n_mels, transform, global_min, global_max):
    # Transform 1D signal to spectrogram
    spec_transform = torchaudio.transforms.Spectrogram(n_fft=n_fft) #.to(device)
    torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=13,
        melkwargs={"n_fft": n_fft, "hop_length": hop_length, "n_mels": n_mels, "center": False})
    spec = spec_transform(waveform) # C X W X H
    # Apply the augmentation here so we ensure the samples are the same width
    # and height. Note that not all samples are the same length, so resizing should
    # be done per sample, not per batch - otherwise we have dimension size inconsistencies
    spec = transform(spec)
    if global_min is not None and global_max is not None:
        spec = (spec - global_min) / (global_max - global_min)
    return spec

    
def collate_fn(batch, labels, n_fft, hop_length, n_mels, transform, global_min=None, global_max=None, **kwargs):
    _spec_samples, targets = [], []
    # Generate spectrogram on the fly here. short samples so this is
    # OK for this use case. Generally I'd pre-process the data to a 
    # format on disk, then stream this during training for larger datasets.
    for waveform, sample_rate, label, *_ in batch:
        spec = preprocess_sample(waveform, sample_rate, n_fft, hop_length, n_mels, transform, global_min, global_max)
        _spec_samples += [spec]
        targets += [torch.tensor(labels.index(label))]
    # Group into tensor
    spec_samples = torch.stack(_spec_samples)
    targets = torch.stack(targets)

    return spec_samples, targets
   
def get_partial_collate_fn(data_params, labels, transform):
    return partial(collate_fn, 
            labels=labels,
            n_fft=data_params.n_fft, 
            hop_length=data_params.hop_length, 
            n_mels=data_params.n_mels, 
            transform=transform, 
            global_min=data_params.global_min, 
            global_max=data_params.global_max)
        
def get_loaders(data_params, **kwargs):
    # Create training and testing split of the data. We do not use validation in this tutorial.
    Path(C.DATA_PATH).mkdir(parents=True, exist_ok=True)
    train_set = SubsetSC("training")
    labels = sorted(list(set(datapoint[2] for datapoint in train_set)))
    val_set = SubsetSC("validation")
    test_set = SubsetSC("testing")
    augmentations = augmentation_transform(data_params.img_size, data_params.mean, data_params.std)
    resize = resize_transform(data_params.img_size, data_params.mean, data_params.std)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=data_params.batch_size,
        shuffle=True,
        collate_fn=get_partial_collate_fn(data_params, labels, transform=augmentations),
        num_workers=data_params.num_workers,
        pin_memory=data_params.pin_memory
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=data_params.batch_size,
        shuffle=False,
        collate_fn=get_partial_collate_fn(data_params, labels, transform=resize),
        num_workers=data_params.num_workers,
        pin_memory=data_params.pin_memory
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=data_params.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=get_partial_collate_fn(data_params, labels, transform=resize),
        num_workers=data_params.num_workers,
        pin_memory=data_params.pin_memory
    )
    return train_loader, val_loader, test_loader


def get_data_min_max(dataloader):
    data_min = None
    data_max = None
    for (batch, _) in tqdm(dataloader):
        _min = batch.min()
        if data_min is None or _min < data_min:
            data_min = _min
        _max = batch.max()
        if data_max is None or _max > data_max:
            data_max = _max
    print(f"DATA_MIN: {data_min} DATA_MAX: {data_max}")
    return data_min, data_max


if __name__ == "__main__":
    params = get_params()
    params.data.global_min=None
    params.data.global_max=None
    train_dl, val_dl, test_dl = get_loaders(params.data)
    get_data_min_max(train_dl)