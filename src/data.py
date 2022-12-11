from torchaudio.datasets import SPEECHCOMMANDS
import torchaudio.transforms as AT
from torchvision.transforms import transforms as VT
import os
import torch
import torchaudio
import numpy as np
from functools import partial
from tqdm import tqdm
from pathlib import Path

from src.configuration import get_params
import src.constants as C


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
        
        
def preprocess_sample(waveform, sample_rate, mfcc_kwargs, transform, global_min, global_max):
    # Transform 1D signal to MFCC's, re-sample to 8k to avoid empty filters.
    #spec_transform = torchaudio.transforms.Spectrogram(n_fft=n_fft) #.to(device)
    _waveform = AT.Resample(orig_freq=sample_rate,new_freq=C.DEFAULT_SR)(waveform)
    spec_transform = torchaudio.transforms.MFCC(
        sample_rate=C.DEFAULT_SR,
        **vars(mfcc_kwargs))
    spec = spec_transform(_waveform) # C X W X H
    # Apply the augmentation here so we ensure the samples are the same width
    # and height. Note that not all samples are the same length, so resizing should
    # be done per sample, not per batch - otherwise we have dimension size inconsistencies
    spec = transform(spec)
    # Scale by the training dataset min/max.
    # NOTE: This is applied to val and test set, so it is assumed they are of
    # the same distribution.
    if global_min is not None and global_max is not None:
        spec = (spec - global_min) / (global_max - global_min)
    return spec

    
def collate_fn(batch, labels, transform, mfcc_kwargs=None, global_min=None, global_max=None, **kwargs):
    _spec_samples, targets = [], []
    # Generate spectrogram on the fly here. short samples so this is
    # OK for this use case. Generally I'd pre-process the data to a 
    # format on disk, then stream this during training for larger datasets.
    for waveform, sample_rate, label, *_ in batch:
        spec = preprocess_sample(waveform, sample_rate, mfcc_kwargs, transform, global_min, global_max)
        _spec_samples += [spec]
        targets += [torch.tensor(labels.index(label))]
    # Group into tensor
    spec_samples = torch.stack(_spec_samples)
    targets = torch.stack(targets)

    return spec_samples, targets
   
def get_partial_collate_fn(data_params, labels, transform):
    return partial(collate_fn, 
            labels=labels,
            transform=transform, 
            global_min=data_params.global_min, 
            global_max=data_params.global_max,
            mfcc_kwargs=data_params.mfcc)
        
def get_loaders(data_params, **kwargs):
    # Create training and testing split of the data. We do not use validation in this tutorial.
    Path(C.DATA_PATH).mkdir(parents=True, exist_ok=True)
    train_set = SPEECHCOMMANDS(root=C.DATA_PATH, subset=C.TRAIN, download=True)
    val_set = SPEECHCOMMANDS(root=C.DATA_PATH, subset=C.VAL, download=True)
    test_set = SPEECHCOMMANDS(root=C.DATA_PATH, subset=C.VAL, download=True)

    labels = sorted(list(set(datapoint[2] for datapoint in train_set)))
    n_classes = len(np.unique(labels))
    augmentations = augmentation_transform(data_params.img_size, data_params.mean, data_params.std)
    resize = resize_transform(data_params.img_size, data_params.mean, data_params.std)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=data_params.batch_size,
        shuffle=True,
        collate_fn=get_partial_collate_fn(data_params, labels, transform=augmentations),
        num_workers=data_params.num_workers,
        pin_memory=data_params.pin_memory,
        prefetch_factor=data_params.prefetch_factor
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=data_params.batch_size,
        shuffle=False,
        collate_fn=get_partial_collate_fn(data_params, labels, transform=resize),
        num_workers=data_params.num_workers,
        pin_memory=data_params.pin_memory,
        prefetch_factor=data_params.prefetch_factor
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=data_params.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=get_partial_collate_fn(data_params, labels, transform=resize),
        num_workers=data_params.num_workers,
        pin_memory=data_params.pin_memory,
        prefetch_factor=data_params.prefetch_factor
    )
    return train_loader, val_loader, test_loader, n_classes


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
    train_dl, val_dl, test_dl, _ = get_loaders(params.data)
    get_data_min_max(train_dl)
