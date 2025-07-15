"""
rdp is a third-party library for Ramer-Douglas-Peucker line simplification.
Run pip install rdp
"""

import struct
import zipfile
import numpy as np
import os
import json
from typing import List, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---------------------- CONFIG ----------------------
CONFIG = {
    "techniques": ["pca", "quantization", "fft"],  # Choose from: "pca", "pla", "quantization", "fft"
    "pca_segments": 10,
    "pla_segments": 10,
    "quantization_levels": 32,
    "fft_keep_ratio": 0.1,
    "compression_output": "compressed_data.bin",
    "config_output": "compression_meta.json"
}

# ---------------------- DATA PARSING ----------------------
def read_zip_binary(path: str) -> List[List[int]]:
    ragged_array = []
    with zipfile.ZipFile(path, 'r') as zf:
        inner_path = path.split("/")[-1].split(".")[0]
        with zf.open(f'{inner_path}.bin', 'r') as r:
            read_binary_from(ragged_array, r)
    return ragged_array

def read_binary_from(ragged_array, r):
    while True:
        size_bytes = r.read(4)
        if not size_bytes:
            break
        sub_array_size = struct.unpack('i', size_bytes)[0]
        sub_array = list(struct.unpack(f'{sub_array_size}h', r.read(sub_array_size * 2)))
        ragged_array.append(sub_array)

# ---------------------- LOSSY METHODS ----------------------
def piecewise_constant_approx(ts: List[int], segments: int) -> List[int]:
    n = len(ts)
    approx = []
    for i in range(segments):
        start = i * n // segments
        end = (i + 1) * n // segments
        mean_val = int(np.mean(ts[start:end]))
        approx.extend([mean_val] * (end - start))
    return approx

def piecewise_linear_approx(ts: List[int], segments: int) -> List[int]:
    n = len(ts)
    approx = np.zeros(n)
    for i in range(segments):
        start = i * n // segments
        end = (i + 1) * n // segments
        if end - start <= 1:
            approx[start:end] = ts[start:end]
            continue
        x = np.linspace(0, 1, end - start)
        y = np.interp(x, [0, 1], [ts[start], ts[end - 1]])
        approx[start:end] = y
    return approx.astype(int).tolist()

def quantize(ts: List[int], levels: int) -> List[int]:
    ts = np.array(ts)
    min_val, max_val = ts.min(), ts.max()
    q = np.round((ts - min_val) / (max_val - min_val) * (levels - 1)).astype(int)
    return (q * (max_val - min_val) / (levels - 1) + min_val).astype(int).tolist()

def fft_threshold(ts: List[int], keep_ratio: float) -> List[int]:
    fft = np.fft.rfft(ts)
    n_keep = int(len(fft) * keep_ratio)
    indices = np.argsort(np.abs(fft))[:-n_keep]
    fft[indices] = 0
    return np.fft.irfft(fft, n=len(ts)).astype(int).tolist()

# ---------------------- COMPRESSION PIPELINE ----------------------
def apply_compression(ts: List[int], config: dict) -> List[int]:
    compressed = ts.copy()
    for method in config["techniques"]:
        if method == "pca":
            compressed = piecewise_constant_approx(compressed, config["pca_segments"])
        elif method == "pla":
            compressed = piecewise_linear_approx(compressed, config["pla_segments"])
        elif method == "quantization":
            compressed = quantize(compressed, config["quantization_levels"])
        elif method == "fft":
            compressed = fft_threshold(compressed, config["fft_keep_ratio"])
    return compressed

# ---------------------- SAVE/LOAD ----------------------
def compress_and_save(input_data: List[List[int]], config: dict):
    with open(config["compression_output"], "wb") as f:
        metadata = []
        for ts in input_data:
            compressed = apply_compression(ts, config)
            metadata.append({
                "orig_len": len(ts),
                "mae": mean_absolute_error(ts, compressed),
                "mse": mean_squared_error(ts, compressed)
            })
            f.write(struct.pack("i", len(compressed)))
            f.write(struct.pack(f"{len(compressed)}h", *compressed))
        with open(config["config_output"], "w") as meta_f:
            json.dump(metadata, meta_f, indent=2)

# ---------------------- READ FOR TRAINING ----------------------
def read_custom_compressed(path: str) -> List[List[int]]:
    ragged_array = []
    with open(path, "rb") as r:
        while True:
            size_bytes = r.read(4)
            if not size_bytes:
                break
            sub_array_size = struct.unpack("i", size_bytes)[0]
            sub_array = list(struct.unpack(f"{sub_array_size}h", r.read(sub_array_size * 2)))
            ragged_array.append(sub_array)
    return ragged_array

# ---------------------- ENTRY ----------------------
if __name__ == "__main__":
    input_path = "training.zip"  # Change this to your actual file
    ecg_data = read_zip_binary(input_path)
    compress_and_save(ecg_data, CONFIG)
    print("Compression complete. Custom data and metadata saved.")
