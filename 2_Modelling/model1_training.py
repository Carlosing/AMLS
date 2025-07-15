import numpy as np
import os
import sys
import torch as torch
from src.models.hyperparamter_tunning import hyperparameter_search
from src.models.model_1 import ECGNet


# Add the project root path if not already present
PROJECT_ROOT = os.path.abspath("..")  # move up one level from notebooks/
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)







def main(train_loader, val_loader, signal_size):

    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    param_grid = {
        "lst_hidden_size": [32, 64, 128],
        
        "learning_rate": [.01, 0.001, 0.0005],
        
        "dropout": [0.1, 0.2, .5],
    }
    fixed = {
        "num_classes": 4,
        "signal_length": signal_size,
        "n_fft": 512,
        "hop_length": 256,
        "conv1_padding": 1,
        "conv2_padding": 1,
        "conv1_kernel": 3,
        "conv2_kernel": 3,
        "lstm_num_layers": 1,
        "conv1_channels": 32,
        "conv2_channels": 32
    }

    results = hyperparameter_search(
        ECGNet,
        param_grid,
        fixed,
        device=device,
        epochs=7,
        train_loader=train_loader,
        val_loader=val_loader,
    )
    
    return results


if __name__ == "__main__":
    main()