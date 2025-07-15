import numpy as np
import os
import sys
import torch as torch
from src.models.hyperparamter_tunning import hyperparameter_search
from src.models.model_2 import TCN_STFT_Classifier


# Add the project root path if not already present
PROJECT_ROOT = os.path.abspath("..")  # move up one level from notebooks/
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)







def main(train_loader, val_loader):

    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    param_grid = {
    # Configuraciones donde len(hidden_channels) == num_levels
    
        'hidden_channels': [[64,128,128,128],[128,128,128,128]],
    'dropout': [0.1, 0.2, 0.3],
    'kernel_size': [3, 5],
    'num_levels': [3,4]
    }


    fixed = {
        "num_classes": 4,
        "n_fft": 256,
        "hop_length": 128,
        "kernel_size": 3,
        "learning_rate" : .001,
    }

    results = hyperparameter_search(
        TCN_STFT_Classifier,
        param_grid,
        fixed,
        device=device,
        epochs=5,
        train_loader=train_loader,
        val_loader=val_loader,
    )
        
    return results




if __name__ == "__main__":
    main()