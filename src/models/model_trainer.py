import torch
from torch import nn

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from src.augmentation.signal_augmentations import augment_batch


class Trainer:
    
    def __init__(self, model, optimizer, criterion, augment_data: bool, device="cuda"):
        
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.augment_data = augment_data
        
    def train_epoch(self, dataloader):
                
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for X_batch, lengths_batch, y_batch in dataloader:
            
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            lengths_batch = lengths_batch.to(self.device)
        
            # Fix: flatten y_batch to 1D
            y_batch = y_batch.view(-1)
            
            self.optimizer.zero_grad()
            
            # Inside your training loop
            if self.augment_data:
                # Move raw batch to device
                X_batch = X_batch.to(self.device)
                lengths_batch = lengths_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                # Apply augmentation on GPU (augment_batch already handles device)
                X_batch_aug, lengths_aug, features_list = augment_batch(X_batch, lengths_batch, device=self.device)
                
                # Forward pass on GPU
                outputs = self.model(X_batch_aug, lengths_aug)

            else:
                outputs = self.model(X_batch, lengths_batch)

            loss = self.criterion(outputs, y_batch)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * X_batch.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
            
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def evaluate(self, dataloader):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for X_batch, lengths_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device)
                lengths_batch = lengths_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                # Fix: flatten y_batch to 1D
                y_batch = y_batch.view(-1)

                outputs = self.model(X_batch, lengths_batch)
                loss = self.criterion(outputs, y_batch)

                running_loss += loss.item() * X_batch.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == y_batch).sum().item()
                total += y_batch.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc
    
    def fit(self, train_loader, val_loader, epochs):
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
        }

        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.evaluate(val_loader)

            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} - "
                  f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)

        return history