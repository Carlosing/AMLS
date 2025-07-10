import numpy as np
import random
import torch
from scipy.signal import resample
from scipy.ndimage import shift

def time_stretch(signal, stretch_factor=1.1):
    length = int(len(signal) * stretch_factor)
    return resample(signal, length)

def time_shift(self, signal, shift_max=0.1):
    shift_amt = int(len(signal) * random.uniform(-shift_max, shift_max))
    return shift(signal, shift_amt, cval=0)

def add_noise(signal, noise_level=0.01):
    noise = np.random.normal(0, noise_level, size=signal.shape)
    return signal + noise

def random_crop(signal, crop_ratio=0.9):
    crop_len = int(len(signal) * crop_ratio)
    start = random.randint(0, len(signal) - crop_len)
    return signal[start:start+crop_len]

def amplitude_scale(signal, scale_range=(0.8, 1.2)):
    factor = random.uniform(*scale_range)
    return signal * factor

def augment_signal(signal):
    if random.random() < 0.5:
        signal = time_stretch(signal, stretch_factor=random.uniform(0.8, 1.2))
    if random.random() < 0.5:
        signal = time_shift(signal, shift_max=0.1)
    if random.random() < 0.5:
        signal = add_noise(signal, noise_level=0.01)
    if random.random() < 0.5:
        signal = amplitude_scale(signal)
    if random.random() < 0.3:
        signal = random_crop(signal, crop_ratio=random.uniform(0.7, 0.95))
    return signal

def augment_batch(self, X_batch, lengths_batch):
        augmented = []
        new_lengths = []
        for i in range(len(X_batch)):
            sig = X_batch[i][:lengths_batch[i]].cpu().numpy()
            aug_sig = augment_signal(sig)
            aug_sig = torch.tensor(aug_sig, dtype=torch.float32)
            new_lengths.append(len(aug_sig))
            augmented.append(aug_sig)

        max_len = max(new_lengths)
        padded_batch = torch.zeros((len(augmented), max_len))
        for i, sig in enumerate(augmented):
            padded_batch[i, :len(sig)] = sig

        return padded_batch, torch.tensor(new_lengths, dtype=torch.int32)
