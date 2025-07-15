import numpy as np
import os
import sys
import torch as torch
from src.data.load_data import load_test_data, EDGCTestDataset
from src.data.stratified_split import pad_test_torch
from src.models.model_1 import ECGNet
from src.models.model_trainer import Trainer
import pandas as pd
from src.data.load_data import load_train_data, EDGCDataset
from src.data.stratified_split import stratified_split_pad_torch






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

    model = ECGNet(
        num_classes=4,
        n_fft=512,
        hop_length=256,
        conv1_padding=1,
        conv2_padding=1,
        conv1_kernel=3,
        conv2_kernel=3,
        lstm_num_layers=1,
        conv1_channels=32,
        conv2_channels=32,
        lst_hidden_size=128,
        dropout=0.1,
        signal_length=signal_size,
        device=device,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    criterion = torch.nn.CrossEntropyLoss()

    trainer = Trainer(model, optimizer, criterion, augment_data=False, device=device)

    history = trainer.fit(train_loader, val_loader, epochs=50)


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