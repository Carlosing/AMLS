import numpy as np
import os
import sys
import torch as torch
from src.data.load_data import load_test_data, EDGCTestDataset
from src.data.stratified_split import pad_test_torch
from src.models.model_trainer import Trainer
import pandas as pd
from src.data.load_data import load_train_data
from src.models.model_2 import TCN_STFT_Classifier






# Add the project root path if not already present
PROJECT_ROOT = os.path.abspath("..")  # move up one level from notebooks/
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)







def main(train_loader, val_loader, signal_size):
    
    X_train, y_train = load_train_data()
    
    X_test = load_test_data()

    X_test, lengths_test = pad_test_torch(X_test)

    test_dataset = EDGCTestDataset(X_test, lengths_test)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TCN_STFT_Classifier(
    num_classes=4,
    hop_length = 128,
    n_fft = 256,
    kernel_size = 5, 
    hidden_channels=  [128, 128, 128, 128],
    dropout = 0.1,
    num_levels = 3,
    device=device,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    criterion = torch.nn.CrossEntropyLoss()

    trainer = Trainer(model, optimizer, criterion, augment_data=True, device=device)

    history = trainer.fit(train_loader, val_loader, epochs=5)

    cm, report = trainer.detailed_metrics(val_loader, class_names=["class_0", "class_1", "class_2", "class_3"])
    
    print(report)

    model.eval()  # Modo evaluación

    all_preds = []

    with torch.no_grad():
        for X_batch, lengths_batch in test_loader:
            X_batch = X_batch.to(device)
            lengths_batch = lengths_batch.to(device)
            
            outputs = model(X_batch, lengths_batch)
            preds = torch.argmax(outputs, dim=1)  # clase con mayor probabilidad
            all_preds.extend(preds.cpu().numpy())
    
    df = pd.DataFrame({'predicted_label': all_preds})
    
    df.to_csv('base.csv', index=False)
    
    


if __name__ == "__main__":
    main()