import os
import pandas as pd
import struct
import numpy as np
from scipy.fft import fft, ifft
from sklearn.model_selection import train_test_split

# Detect project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def read_binary(path):
    ragged_array = []
    with open(path, "rb") as f:
        while True:
            size_bytes = f.read(4)
            if not size_bytes:
                break
            (sub_array_size,) = struct.unpack('i', size_bytes)
            sub_array_bytes = f.read(sub_array_size * 2)
            sub_array = list(struct.unpack(f'{sub_array_size}h', sub_array_bytes))
            ragged_array.append(sub_array)
    return ragged_array

# === Old style loading ===
def load_train_data():
    x_path = os.path.join(PROJECT_ROOT, "data", "raw", "X_train", "X_train.bin")
    y_path = os.path.join(PROJECT_ROOT, "data", "raw", "y_train.csv")

    X_train = read_binary(x_path)
    y_train = pd.read_csv(y_path, header=None)

    print(f"[✓] Loaded X_train with {len(X_train)} sequences")
    print(f"[✓] Loaded y_train with shape {y_train.shape}")
    return X_train, y_train

def load_test_data():
    x_path = os.path.join(PROJECT_ROOT, "data", "raw", "X_test", "X_test.bin")
    X_test = read_binary(x_path)
    print(f"[✓] Loaded X_test with {len(X_test)} sequences")
    return X_test

# === New style loading and compression ===
def lossy_fft_compress(signal, keep_ratio=0.1):
    spectrum = fft(signal)
    n_keep = int(len(spectrum) * keep_ratio)
    idx = np.argsort(np.abs(spectrum))[-n_keep:]
    compressed = np.zeros_like(spectrum, dtype=np.complex64)
    compressed[idx] = spectrum[idx]
    reconstructed = np.real(ifft(compressed))
    return reconstructed

def reduce_dataset_and_save(X, y, size_ratio=0.25, keep_ratio=0.1, x_save_path="X_train_reduced.bin", y_save_path="y_train_reduced.csv"):
    print(f"\n[🔧] Reducing dataset to {int(size_ratio*100)}% and compressing with keep_ratio={keep_ratio}")
    print(f"[ℹ] Original dataset size: {len(X)} samples")

    _, X_reduced, _, y_reduced = train_test_split(
        X, y, test_size=size_ratio, stratify=y, random_state=42
    )
    print(f"[ℹ] Reduced dataset size: {len(X_reduced)} samples")

    X_compressed = []

    for i, signal in enumerate(X_reduced):
        compressed_signal = lossy_fft_compress(signal, keep_ratio=keep_ratio)
        X_compressed.append(compressed_signal.astype(np.float32))

    save_compressed_to_binary(X_compressed, x_save_path)
    save_labels_csv(y_reduced, y_save_path)
    print(f"[✅] Saved reduced binary data to {x_save_path} and labels to {y_save_path}")

def save_compressed_to_binary(X_compressed, path):
    with open(path, "wb") as f:
        for signal in X_compressed:
            scaled = np.clip(signal * 1000, -32768, 32767).astype(np.int16)
            f.write(struct.pack("i", len(scaled)))                      # Length
            f.write(struct.pack(f"{len(scaled)}h", *scaled))           # Data

def save_labels_csv(y, filename):
    pd.DataFrame(y).to_csv(filename, header=False, index=False)


def load_reduced_data():
    x_path = os.path.join(PROJECT_ROOT, "data", "X_train_reduced.bin")
    y_path = os.path.join(PROJECT_ROOT, "data", "y_train_reduced.csv")
    X_train, y_train = load_train_data()
    reduce_dataset_and_save(
            X_train,
            y_train.values.ravel(),
            size_ratio=0.25,
            keep_ratio=0.1,
            x_save_path=x_path,
            y_save_path=y_path
        )
    X_train = read_binary(x_path)
    y_train = pd.read_csv(y_path, header=None)

    print(f"Loaded reduced X_train with {len(X_train)} sequences")
    print(f"Loaded reduced y_train with shape {y_train.shape}")
    return X_train, y_train


# === Main ===
if __name__ == "__main__":
    use_new_method = True  # Change this flag to switch between old and new loading
    X_train, y_train = load_train_data()
    if use_new_method:
        reduce_dataset_and_save(
            X_train,
            y_train.values.ravel(),
            size_ratio=0.25,
            keep_ratio=0.1,
            x_save_path=os.path.join(PROJECT_ROOT, "data", "X_train_reduced.bin"),
            y_save_path=os.path.join(PROJECT_ROOT, "data", "y_train_reduced.csv")
        )
    
    X_test = load_test_data()
