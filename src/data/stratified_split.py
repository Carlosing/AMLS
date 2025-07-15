import torch
import numpy as np
import pandas as pd

def stratified_split_pad_torch(X, y, val_ratio=0.1, padding_value=0):
    """
    Performs a stratified split and returns PyTorch tensors:
    X_train_padded, X_val_padded, lengths_train, lengths_val, y_train_split, y_val_split
    """
    X = np.array(X, dtype=object)
    y = np.array(y)

    train_indices = []
    val_indices = []

    for cls in np.unique(y):
        cls_indices = np.where(y == cls)[0]
        np.random.shuffle(cls_indices)
        split_point = int(len(cls_indices) * val_ratio)
        val_indices.extend(cls_indices[:split_point])
        train_indices.extend(cls_indices[split_point:])

    X_train_split = [X[i] for i in train_indices]
    X_val_split = [X[i] for i in val_indices]
    y_train_split = y[train_indices]
    y_val_split = y[val_indices]

    def pad_sequences(sequences, padding_value=0.0):
        # Convert and normalize
        norm_seqs = []
        lengths = torch.zeros(len(sequences), dtype=torch.int32)

        for i, seq in enumerate(sequences):
            seq = torch.tensor(seq, dtype=torch.float32)
            mean = seq.mean()
            std = seq.std()
            norm_seq = (seq - mean) / (std + 1e-8)
            norm_seqs.append(norm_seq)
            lengths[i] = len(seq)

        maxlen = max(lengths).item()
        padded = torch.full((len(sequences), maxlen), padding_value, dtype=torch.float32)

        for i, seq in enumerate(norm_seqs):
            padded[i, :len(seq)] = seq

        return padded, lengths

    X_train_padded, lengths_train = pad_sequences(X_train_split, padding_value)
    X_val_padded, lengths_val = pad_sequences(X_val_split, padding_value)

    # convert labels to tensors (int64 for classification)
    y_train_split = torch.tensor(y_train_split, dtype=torch.int64)
    y_val_split = torch.tensor(y_val_split, dtype=torch.int64)

    return X_train_padded, X_val_padded, lengths_train, lengths_val, y_train_split, y_val_split

def pad_test_torch(X, padding_value=0.0):
    """
    Preprocesses the test set by applying normalization and padding.
    Returns PyTorch tensors: X_test_padded, lengths_test
    """
    X = np.array(X, dtype=object)

    norm_seqs = []
    lengths = torch.zeros(len(X), dtype=torch.int32)

    for i, seq in enumerate(X):
        seq = torch.tensor(seq, dtype=torch.float32)
        mean = seq.mean()
        std = seq.std()
        norm_seq = (seq - mean) / (std + 1e-8)
        norm_seqs.append(norm_seq)
        lengths[i] = len(seq)

    maxlen = max(lengths).item()
    padded = torch.full((len(X), maxlen), padding_value, dtype=torch.float32)

    for i, seq in enumerate(norm_seqs):
        padded[i, :len(seq)] = seq

    return padded, lengths
