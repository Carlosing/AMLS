# ecg_augmentation.py

import numpy as np
import random
import torch
from scipy.signal import resample
from scipy.ndimage import shift
from scipy.fft import fft, ifft

import biosppy
import heartpy as hp

augmentation_config = {
    "time_stretch": {"enabled": True, "min": 0.9, "max": 1.1},
    "time_shift": {"enabled": True, "max_shift": 0.05},
    "add_noise": {"enabled": True, "noise_level": 0.02},
    "amplitude_scale": {"enabled": True, "range": (0.9, 1.1)},
    "random_crop": {"enabled": True, "min": 0.8, "max": 0.95},
    "ifft_transform": {"enabled": True, "drop_ratio": 0.1},
    "biosppy": {"enabled": True},
    "heartpy": {"enabled": True}
}

# ----------- Augmentations -----------

def time_stretch(signal, stretch_factor):
    length = int(len(signal) * stretch_factor)
    return resample(signal, length)

def time_shift(signal, shift_max):
    shift_amt = int(len(signal) * random.uniform(-shift_max, shift_max))
    return shift(signal, shift_amt, cval=0)

def add_noise(signal, noise_level):
    noise = np.random.normal(0, noise_level, size=signal.shape)
    return signal + noise

def amplitude_scale(signal, scale_range):
    factor = random.uniform(*scale_range)
    return signal * factor

def random_crop(signal, crop_ratio):
    crop_len = int(len(signal) * crop_ratio)
    start = random.randint(0, len(signal) - crop_len)
    return signal[start:start+crop_len]

def ifft_augment(signal, drop_ratio):
    spectrum = fft(signal)
    cutoff = int(len(spectrum) * (1 - drop_ratio))
    spectrum[cutoff:] = 0
    return np.real(ifft(spectrum))

# ----------- Feature Engineering -----------

def extract_biosppy_features(signal, sampling_rate=300):
    if biosppy is None:
        return {}
    out = biosppy.signals.ecg.ecg(signal=signal, sampling_rate=sampling_rate, show=False)
    return {
        "hr_mean": np.mean(out['heart_rate']),
        "hr_std": np.std(out['heart_rate']),
        "num_rpeaks": len(out['rpeaks']),
    }

def extract_heartpy_features(signal, sampling_rate=300):
    if hp is None:
        return {}
    try:
        wd, m = hp.process(signal, sample_rate=sampling_rate)
        return {
            "bpm": m.get('bpm', 0),
            "ibi": m.get('ibi', 0),
            "sdnn": m.get('sdnn', 0)
        }
    except Exception:
        return {"bpm": 0, "ibi": 0, "sdnn": 0}

# ----------- Pipeline -----------

def augment_signal(signal, config=augmentation_config):
    if config["time_stretch"]["enabled"]:
        factor = random.uniform(config["time_stretch"]["min"], config["time_stretch"]["max"])
        signal = time_stretch(signal, factor)
    if config["time_shift"]["enabled"]:
        signal = time_shift(signal, config["time_shift"]["max_shift"])
    if config["add_noise"]["enabled"]:
        signal = add_noise(signal, config["add_noise"]["noise_level"])
    if config["amplitude_scale"]["enabled"]:
        signal = amplitude_scale(signal, config["amplitude_scale"]["range"])
    if config["random_crop"]["enabled"]:
        crop_ratio = random.uniform(config["random_crop"]["min"], config["random_crop"]["max"])
        signal = random_crop(signal, crop_ratio)
    if config["ifft_transform"]["enabled"]:
        signal = ifft_augment(signal, config["ifft_transform"]["drop_ratio"])
    return signal

def augment_batch(X_batch, lengths, config=augmentation_config, sampling_rate=300):
    aug_batch = []
    new_lengths = []
    features_list = []

    for i in range(len(X_batch)):
        sig = X_batch[i][:lengths[i]].cpu().numpy()
        aug_sig = augment_signal(sig, config)
        features = {}

        if config["biosppy"]["enabled"]:
            features.update(extract_biosppy_features(aug_sig, sampling_rate))
        if config["heartpy"]["enabled"]:
            features.update(extract_heartpy_features(aug_sig, sampling_rate))

        aug_tensor = torch.tensor(aug_sig, dtype=torch.float32)
        aug_batch.append(aug_tensor)
        new_lengths.append(len(aug_tensor))
        features_list.append(features)

    max_len = max(new_lengths)
    padded_batch = torch.zeros((len(aug_batch), max_len))
    for i, sig in enumerate(aug_batch):
        padded_batch[i, :len(sig)] = sig

    return padded_batch, torch.tensor(new_lengths, dtype=torch.int32), features_list
