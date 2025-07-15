import struct
import zipfile
import numpy as np
import os
import json
from typing import List
from sklearn.metrics import mean_absolute_error, mean_squared_error
import sys

# ---------------------- CONFIG ----------------------

PROJECT_ROOT = os.path.abspath("..")  # move up one level from notebooks/
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

compressed_path = os.path.join(PROJECT_ROOT, "data", "processed", "compressed_data.bin")
output_path = os.path.join(PROJECT_ROOT, "data", "processed", "compression_meta.json")

CONFIG = {
    "techniques": ["quantization"],  # Choose one from: "quantization", "pla", "fft"
    "pca_segments": 10,
    "pla_segments": 10,
    "quantization_levels": 32,
    "fft_keep_ratio": 0.1,
    "compression_output": compressed_path,
    "config_output": output_path
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
def quantize_compress(ts: List[int], levels: int):
    ts = np.array(ts)
    min_val, max_val = ts.min(), ts.max()
    scale = (max_val - min_val) / (levels - 1) if max_val != min_val else 1.0
    indices = np.round((ts - min_val) / scale).astype(np.uint8)
    return indices, min_val, scale

def quantize_decompress(indices: np.ndarray, min_val, scale):
    return (indices.astype(float) * scale + min_val).astype(int).tolist()

def pla_compress(ts: List[int], segments: int):
    n = len(ts)
    breakpoints = np.linspace(0, n, segments + 1, dtype=int)
    segs = []
    for i in range(segments):
        start, end = breakpoints[i], breakpoints[i+1]
        slope = (ts[end - 1] - ts[start]) / (end - start) if end - start > 1 else 0
        segs.append((ts[start], slope, end - start))
    return segs

def pla_decompress(segments):
    ts = []
    for start_val, slope, length in segments:
        seg = [int(start_val + i * slope) for i in range(length)]
        ts.extend(seg)
    return ts

def fft_compress(ts: List[int], keep_ratio: float):
    fft = np.fft.rfft(ts)
    n_keep = int(len(fft) * keep_ratio)
    indices = np.argsort(np.abs(fft))[-n_keep:]
    kept = fft[indices]
    return indices.astype(np.uint16), kept.astype(np.complex64), len(ts)

def fft_decompress(indices, kept_vals, orig_len):
    fft_full = np.zeros(orig_len // 2 + 1, dtype=np.complex64)
    fft_full[indices] = kept_vals
    return np.fft.irfft(fft_full, n=orig_len).astype(int).tolist()

# ---------------------- COMPRESSION ----------------------
def compress_and_save(input_data: List[List[int]], config: dict):
    with open(config["compression_output"], "wb") as f:
        metadata = []
        for ts in input_data:
            original_len = len(ts)

            if "quantization" in config["techniques"]:
                indices, min_val, scale = quantize_compress(ts, config["quantization_levels"])
                f.write(b'Q')
                f.write(struct.pack('<Hff', original_len, min_val, scale))
                f.write(indices.tobytes())
                reconstructed = quantize_decompress(indices, min_val, scale)

            elif "pla" in config["techniques"]:
                segs = pla_compress(ts, config["pla_segments"])
                f.write(b'P')
                f.write(struct.pack('<H', len(segs)))
                for start, slope, length in segs:
                    f.write(struct.pack('<hff', start, slope, length))
                reconstructed = pla_decompress(segs)

            elif "fft" in config["techniques"]:
                indices, coeffs, orig_len = fft_compress(ts, config["fft_keep_ratio"])
                f.write(b'F')
                f.write(struct.pack('<H', len(indices)))
                f.write(struct.pack('<H', orig_len))
                f.write(indices.tobytes())
                f.write(coeffs.tobytes())
                reconstructed = fft_decompress(indices, coeffs, orig_len)

            else:
                raise ValueError("Unsupported technique.")

            metadata.append({
                "orig_len": original_len,
                "mae": mean_absolute_error(ts, reconstructed),
                "mse": mean_squared_error(ts, reconstructed)
            })

    with open(config["config_output"], "w") as meta_f:
        json.dump(metadata, meta_f, indent=2)
        
# ---------------------- READ FOR TRAINING ----------------------
def read_custom_compressed(path: str) -> List[List[int]]:
    ragged_array = []
    with open(path, "rb") as r:
        while True:
            tag = r.read(1)
            if not tag:
                break

            tag = tag.decode()

            if tag == 'Q':
                original_len, min_val, scale = struct.unpack('<Hff', r.read(10))
                indices = np.frombuffer(r.read(original_len), dtype=np.uint8)
                ts = quantize_decompress(indices, min_val, scale)
                ragged_array.append(ts)

            elif tag == 'P':
                num_segs = struct.unpack('<H', r.read(2))[0]
                segments = []
                for _ in range(num_segs):
                    start, slope, length = struct.unpack('<hff', r.read(10))
                    segments.append((start, slope, length))
                ts = pla_decompress(segments)
                ragged_array.append(ts)

            elif tag == 'F':
                n_indices = struct.unpack('<H', r.read(2))[0]
                orig_len = struct.unpack('<H', r.read(2))[0]
                indices = np.frombuffer(r.read(n_indices * 2), dtype=np.uint16)
                coeffs = np.frombuffer(r.read(n_indices * 8), dtype=np.complex64)
                ts = fft_decompress(indices, coeffs, orig_len)
                ragged_array.append(ts)

            else:
                raise ValueError(f"Unknown compression tag: {tag}")

    return ragged_array


# ---------------------- ENTRY ----------------------
if __name__ == "__main__":
    input_path = "training.zip"  # Change this to your actual file
    ecg_data = read_zip_binary(input_path)
    compress_and_save(ecg_data, CONFIG)
    print("Compression complete. Custom data and metadata saved.")