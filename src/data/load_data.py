import os
import pandas as pd
import struct

# NEW: detects the project root from the current file's location
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def read_binary(path):
    print("load_data_project_root",path)
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

if __name__ == "__main__":
    X_train, y_train = load_train_data()
    X_test = load_test_data()
