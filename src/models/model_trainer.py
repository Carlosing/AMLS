import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from src.augmentation.signal_augmentations import augment_batch

class Trainer:
    
    def __init__(self, model, optimizer, criterion, augment_data: bool, num_classes=4, device="cuda"):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.augment_data = augment_data
        self.num_classes = num_classes
        
    def _calculate_f1(self, y_true, y_pred):
        """
        Calculate F1 score for multi-class classification
        Args:
            y_true: Ground truth labels (1D tensor)
            y_pred: Predicted labels (1D tensor)
        Returns:
            f1: Macro F1 score
        """
        # Create confusion matrix
        confusion_matrix = torch.zeros((self.num_classes, self.num_classes), 
                                     dtype=torch.int64, device=self.device)
        
        for t, p in zip(y_true.view(-1), y_pred.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1
        
        # Calculate precision and recall per class
        true_positives = torch.diag(confusion_matrix)
        false_positives = confusion_matrix.sum(0) - true_positives
        false_negatives = confusion_matrix.sum(1) - true_positives
        
        # Avoid division by zero
        precision = true_positives / (true_positives + false_positives + 1e-9)
        recall = true_positives / (true_positives + false_negatives + 1e-9)
        
        # Calculate F1 score per class
        f1_per_class = 2 * (precision * recall) / (precision + recall + 1e-9)
        
        # Macro F1 (average of all classes)
        macro_f1 = f1_per_class.mean()
        return macro_f1.item()
    
    def train_epoch(self, dataloader):
        self.model.train()
        running_loss = 0.0
        total_samples = 0
        all_preds = []
        all_targets = []
        
        for X_batch, lengths_batch, y_batch in dataloader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device).view(-1)
            lengths_batch = lengths_batch.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.augment_data:
                X_batch_aug, lengths_aug = augment_batch(X_batch, lengths_batch, device=self.device)
                outputs = self.model(X_batch_aug, lengths_aug)
            else:
                outputs = self.model(X_batch, lengths_batch)
            
            loss = self.criterion(outputs, y_batch)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * X_batch.size(0)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.append(predicted)
            all_targets.append(y_batch)
            total_samples += y_batch.size(0)
        
        # Concatenate all predictions and targets
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        
        epoch_loss = running_loss / total_samples
        epoch_f1 = self._calculate_f1(all_targets, all_preds)
        
        return epoch_loss, epoch_f1
    
    def evaluate(self, dataloader):
        self.model.eval()
        running_loss = 0.0
        total_samples = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, lengths_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device)
                lengths_batch = lengths_batch.to(self.device)
                y_batch = y_batch.to(self.device).view(-1)
                
                outputs = self.model(X_batch, lengths_batch)
                loss = self.criterion(outputs, y_batch)
                
                running_loss += loss.item() * X_batch.size(0)
                _, predicted = torch.max(outputs, 1)
                
                all_preds.append(predicted)
                all_targets.append(y_batch)
                total_samples += y_batch.size(0)
        
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        
        epoch_loss = running_loss / total_samples
        epoch_f1 = self._calculate_f1(all_targets, all_preds)
        
        return epoch_loss, epoch_f1
    
    def fit(self, train_loader, val_loader, epochs):
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_f1': [],
            'val_f1': [],
        }

        for epoch in range(epochs):
            train_loss, train_f1 = self.train_epoch(train_loader)
            val_loss, val_f1 = self.evaluate(val_loader)

            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss:.4f} - Train F1: {train_f1:.4f} - "
                  f"Val Loss: {val_loss:.4f} - Val F1: {val_f1:.4f}")

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_f1'].append(train_f1)
            history['val_f1'].append(val_f1)

        return history