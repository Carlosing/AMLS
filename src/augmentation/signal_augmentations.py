import torch
import torch.nn.functional as F

# -------------------------------------------------
# GPU-only augmentation config (all CPU features disabled)
# -------------------------------------------------
augmentation_config = {
    "time_stretch": {"enabled": True, "min": 0.98, "max": 1.02},
    "time_shift": {"enabled": True, "max_shift": 0.02},
    "add_noise": {"enabled": False, "noise_level": 0.005},
    "amplitude_scale": {"enabled": True, "range": (0.95, 1.05)},
    "random_crop": {"enabled": False},
    "ifft_transform": {"enabled": False}
}

# -------------------------------------------------
# Pure GPU Augmentation functions
# -------------------------------------------------

def time_stretch_gpu(signal, stretch_factor):
    """Resample the signal to stretch/compress it in time using GPU interpolation"""
    orig_len = signal.size(0)
    new_len = max(1, int(orig_len * stretch_factor))
    
    signal = signal.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
    signal = F.interpolate(signal, size=new_len, mode='linear', align_corners=False)
    return signal.squeeze(0).squeeze(0)

def time_shift_gpu(signal, shift_max):
    """Shift signal left/right with zero padding on GPU"""
    n = signal.size(0)
    shift_amt = int(n * torch.rand(1, device=signal.device).item() * (2 * shift_max) - shift_max)
    
    if shift_amt == 0:
        return signal
    
    result = torch.zeros_like(signal)
    if shift_amt > 0:
        result[shift_amt:] = signal[:-shift_amt]
    else:
        result[:shift_amt] = signal[-shift_amt:]
    return result

def add_noise_gpu(signal, noise_level):
    """Add Gaussian noise directly on GPU"""
    noise = torch.randn_like(signal) * noise_level * signal.std()
    return signal + noise

def amplitude_scale_gpu(signal, scale_range):
    """Randomly scale signal amplitude on GPU"""
    factor = torch.rand(1, device=signal.device).item() * (scale_range[1] - scale_range[0]) + scale_range[0]
    return signal * factor

def random_crop_gpu(signal, crop_ratio):
    """Randomly crop a fraction of the signal on GPU"""
    orig_len = signal.size(0)
    crop_len = max(1, int(orig_len * crop_ratio))
    start = int(torch.rand(1, device=signal.device).item() * (orig_len - crop_len))
    return signal[start:start+crop_len]

def augment_signal_gpu(signal, config=augmentation_config):
    """Pure GPU augmentation pipeline"""
    signal = signal.clone()
    
    if config["time_stretch"]["enabled"]:
        factor = torch.rand(1, device=signal.device).item() * (config["time_stretch"]["max"] - config["time_stretch"]["min"]) + config["time_stretch"]["min"]
        signal = time_stretch_gpu(signal, factor)

    if config["time_shift"]["enabled"]:
        signal = time_shift_gpu(signal, config["time_shift"]["max_shift"])

    if config["add_noise"]["enabled"]:
        signal = add_noise_gpu(signal, config["add_noise"]["noise_level"])

    if config["amplitude_scale"]["enabled"]:
        signal = amplitude_scale_gpu(signal, config["amplitude_scale"]["range"])

    if config["random_crop"]["enabled"]:
        crop_ratio = torch.rand(1, device=signal.device).item() * (config["random_crop"].get("max", 1.0) - config["random_crop"].get("min", 0.95)) + config["random_crop"].get("min", 0.95)
        signal = random_crop_gpu(signal, crop_ratio)

    return signal

def augment_batch(X_batch, lengths, config=augmentation_config, device="cuda"):
    """
    Pure GPU batch augmentation - no CPU transfers
    Returns:
        padded_batch: (batch_size, max_len) tensor on GPU
        new_lengths: (batch_size,) tensor of lengths on GPU
    """
    aug_batch = []
    new_lengths = []

    for i in range(X_batch.size(0)):
        sig = X_batch[i, :lengths[i]]
        aug_sig = augment_signal_gpu(sig, config)
        aug_batch.append(aug_sig)
        new_lengths.append(aug_sig.size(0))

    # Pad sequences on GPU
    max_len = max(new_lengths) if new_lengths else 0
    padded_batch = torch.zeros((len(aug_batch), max_len), dtype=torch.float32, device=device)
    for i, sig in enumerate(aug_batch):
        padded_batch[i, :sig.size(0)] = sig

    return padded_batch, torch.tensor(new_lengths, dtype=torch.int32, device=device)