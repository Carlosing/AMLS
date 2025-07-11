import numpy as np
import random
import torch
from scipy.signal import resample
from scipy.ndimage import shift
from scipy.fft import fft, ifft

import biosppy
import heartpy as hp

# -------------------------------------------------
# Updated, more conservative augmentation config
# -------------------------------------------------
augmentation_config = {
    "time_stretch": {"enabled": True, "min": 0.98, "max": 1.02},
    "time_shift": {"enabled": True, "max_shift": 0.02},
    "add_noise": {"enabled": True, "noise_level": 0.005},
    "amplitude_scale": {"enabled": True, "range": (0.95, 1.05)},
    "random_crop": {"enabled": False},    # disabled by default (can harm ECG context)
    "ifft_transform": {"enabled": False}, # disabled by default (can distort morphology)
    "biosppy": {"enabled": True},
    "heartpy": {"enabled": False}
}

# -------------------------------------------------
# Augmentation functions
# -------------------------------------------------

def time_stretch(signal, stretch_factor):
    """
    Resample the signal to stretch/compress it in time.
    """
    length = int(len(signal) * stretch_factor)
    return resample(signal, length)

def time_shift(signal, shift_max):
    """
    Shift signal left/right by random fraction of total length.
    """
    shift_amt = int(len(signal) * random.uniform(-shift_max, shift_max))
    return shift(signal, shift_amt, cval=0)

def add_noise(signal, noise_level):
    """
    Add Gaussian noise.
    """
    noise = np.random.normal(0, noise_level, size=signal.shape)
    return signal + noise

def amplitude_scale(signal, scale_range):
    """
    Randomly scale signal amplitude.
    """
    factor = random.uniform(*scale_range)
    return signal * factor

def random_crop(signal, crop_ratio):
    """
    Randomly crop a fraction of the signal.
    """
    crop_len = int(len(signal) * crop_ratio)
    if crop_len >= len(signal):
        return signal
    start = random.randint(0, len(signal) - crop_len)
    return signal[start:start+crop_len]

def ifft_augment(signal, drop_ratio):
    """
    Zero out high frequencies in FFT.
    """
    spectrum = fft(signal)
    cutoff = int(len(spectrum) * (1 - drop_ratio))
    spectrum[cutoff:] = 0
    return np.real(ifft(spectrum))

# -------------------------------------------------
# Feature extraction functions
# -------------------------------------------------

def extract_biosppy_features(signal, sampling_rate=300):
    """
    Use biosppy to extract HR features.
    """
    try:
        out = biosppy.signals.ecg.ecg(signal=signal, sampling_rate=sampling_rate, show=False)
        return {
            "hr_mean": np.mean(out['heart_rate']) if len(out['heart_rate']) > 0 else 0,
            "hr_std": np.std(out['heart_rate']) if len(out['heart_rate']) > 0 else 0,
            "num_rpeaks": len(out['rpeaks'])
        }
    except Exception:
        return {"hr_mean": 0, "hr_std": 0, "num_rpeaks": 0}

def extract_heartpy_features(signal, sampling_rate=300):
    """
    Use heartpy to extract HRV features.
    """
    try:
        wd, m = hp.process(signal, sample_rate=sampling_rate)
        return {
            "bpm": m.get('bpm', 0),
            "ibi": m.get('ibi', 0),
            "sdnn": m.get('sdnn', 0)
        }
    except Exception:
        return {"bpm": 0, "ibi": 0, "sdnn": 0}

# -------------------------------------------------
# Augmentation pipeline
# -------------------------------------------------

def augment_signal(signal, config=augmentation_config):
    """
    Apply all enabled augmentations sequentially.
    """
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
        crop_ratio = random.uniform(config["random_crop"].get("min", 0.95), config["random_crop"].get("max", 1.0))
        signal = random_crop(signal, crop_ratio)

    if config["ifft_transform"]["enabled"]:
        signal = ifft_augment(signal, config["ifft_transform"]["drop_ratio"])

    return signal

def augment_batch(X_batch, lengths, config=augmentation_config, sampling_rate=300):
    """
    Apply augmentation to a batch of signals and extract optional features.
    """
    aug_batch = []
    new_lengths = []
    features_list = []

    for i in range(len(X_batch)):
        # Cut to real length and convert to numpy
        sig = X_batch[i][:lengths[i]].cpu().numpy()
        
        # Augment
        aug_sig = augment_signal(sig, config)
        
        # Extract features if needed
        features = {}
        if config["biosppy"]["enabled"]:
            features.update(extract_biosppy_features(aug_sig, sampling_rate))
        if config["heartpy"]["enabled"]:
            features.update(extract_heartpy_features(aug_sig, sampling_rate))

        # Convert back to tensor
        aug_tensor = torch.tensor(aug_sig, dtype=torch.float32)
        aug_batch.append(aug_tensor)
        new_lengths.append(len(aug_tensor))
        features_list.append(features)

    # Pad sequences to same length
    max_len = max(new_lengths)
    padded_batch = torch.zeros((len(aug_batch), max_len))
    for i, sig in enumerate(aug_batch):
        padded_batch[i, :len(sig)] = sig

    return padded_batch, torch.tensor(new_lengths, dtype=torch.int32), features_list
