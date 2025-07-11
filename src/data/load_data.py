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

def reduce_dataset(X, y, size_ratio=0.25, keep_ratio=0.1, save_path="reduced_dataset.npz"):
    print(f"\n[🔧] Reducing dataset to {int(size_ratio*100)}% and compressing with keep_ratio={keep_ratio}")
    print(f"[ℹ] Original dataset size: {len(X)} samples")

    _, X_reduced, _, y_reduced = train_test_split(
        X, y, test_size=size_ratio, stratify=y, random_state=42
    )
    print(f"[ℹ] Reduced dataset size: {len(X_reduced)} samples")

    X_compressed = []
    maes, mses = [], []

    for i, signal in enumerate(X_reduced):
        compressed_signal = lossy_fft_compress(signal, keep_ratio=keep_ratio)
        mae = np.mean(np.abs(signal - compressed_signal))
        mse = np.mean((signal - compressed_signal)**2)
        maes.append(mae)
        mses.append(mse)
        X_compressed.append(compressed_signal.astype(np.float32))

        if (i + 1) % 10 == 0 or (i + 1) == len(X_reduced):
            print(f"[⏳] Processed {i+1}/{len(X_reduced)} signals — Current MAE: {mae:.4f}, MSE: {mse:.4f}")

    print(f"[📊] Mean MAE: {np.mean(maes):.4f}, Mean MSE: {np.mean(mses):.4f}")

    np.savez_compressed(
        save_path,
        X=np.array(X_compressed, dtype=object),
        y=np.array(y_reduced)
    )
    print(f"[✅] Saved reduced dataset to {save_path}")

def load_reduced_dataset(path):
    data = np.load(path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    print(f"[📦] Loaded reduced dataset: {len(X)} samples")
    return X, y

def load_train_data_reduced(size_ratio=0.25, keep_ratio=0.1):
    X_train, y_train = load_train_data()
    save_path = os.path.join(PROJECT_ROOT, f"data/reduced_{int(size_ratio*100)}.npz")
    reduce_dataset(X_train, y_train.values.ravel(), size_ratio=size_ratio, keep_ratio=keep_ratio, save_path=save_path)
    return load_reduced_dataset(save_path)

# === Main ===
if __name__ == "__main__":
    use_new_method = False  # Change this flag to switch between old and new loading

    if use_new_method:
        # Use new reduced & compressed dataset loading
        X_train, y_train = load_train_data_reduced(size_ratio=0.25, keep_ratio=0.1)
        # If you want, you can add a similar function for test data in new style
    else:
        # Use old style loading
        X_train, y_train = load_train_data()
        X_test = load_test_data()
