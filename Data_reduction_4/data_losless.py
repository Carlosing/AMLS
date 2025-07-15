import os
import sys
import numpy as np
import torch as torch
from torch.nn.utils.rnn import pad_sequence
from collections import Counter


# Add project root path if not already present
PROJECT_ROOT = os.path.abspath("..")  # move up one level from notebooks/
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
    # Visual confirmation
print("[✓] Project located at:", PROJECT_ROOT)
    
from src.data.lossless_compression import write_compressed_file, read_compressed_file
from src.data.lossy_compression import read_custom_compressed, compress_and_save, CONFIG



from src.data.load_data import load_train_data, load_test_data, EDGCDataset
from src.data.stratified_split import stratified_split_pad_torch

def print_class_stats(y, name="Dataset"):
    """Print basic class statistics"""
    # Convert to list of scalar values
    if isinstance(y, np.ndarray):
        y = y.flatten().tolist()
    elif torch.is_tensor(y):
        y = y.flatten().tolist()
    elif isinstance(y, list):
        y = [item.item() if torch.is_tensor(item) else item for item in y]
    
    class_counts = Counter(y)
    total = len(y)
    
    print(f"\nClass statistics - {name}:")
    print(f"Total number of records: {total}")
    print(f"Number of classes: {len(class_counts)}")
    
    print("\nRecords per class:")
    for class_label, count in sorted(class_counts.items()):
        print(f"  Class {class_label}: {count} records ({count/total:.1%})")


def main():

    print("Loading data...")
    X_train, y_train = load_train_data()
    
    use_compressed = False
    
    if use_compressed:
        write_compressed_file("compressed_data.bin", X_train)

        X_train_compressed = read_compressed_file("compressed_data.bin")
        X_tensor_list = [torch.tensor(x, dtype=torch.float32) for x in X_train_compressed]
        X_train_pad = pad_sequence(X_tensor_list, batch_first=True)
        lengths_train_final = torch.tensor([len(x) for x in X_train_compressed])
        y_train_tensor = torch.tensor(y_train.iloc[:, 0].values, dtype=torch.long)

    durations = np.array([len(x) / 300 for x in X_train])

    print("Splitting and padding...")

    X_train, X_val, lengths_train, lengths_val, y_train, y_val = (
        stratified_split_pad_torch(X_train, y_train)
    )     

    print_class_stats(y_train, "Training set")
    print_class_stats(y_val, "Validation set")
    
    train_dataset = EDGCDataset(X_train_pad, lengths_train_final, y_train_tensor) if use_compressed else EDGCDataset(X_train, lengths_train, y_train)
    val_dataset = EDGCDataset(X_val, lengths_val, y_val)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)

    for X_batch, lengths_batch, y_batch in train_loader:
        print(
            f"Batch shapes → X: {X_batch.shape}, lengths: {lengths_batch.shape}, y: {y_batch.shape}"
        )
        break

    print("Data preparation completed.")

    return train_loader, val_loader, X_train.shape[1]


if __name__ == "__main__":
    main()
