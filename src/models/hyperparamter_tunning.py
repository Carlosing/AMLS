import itertools
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
from src.models.model_trainer import Trainer
    
    

def hyperparameter_search(
    model_class,
    param_grid,
    fixed_params=None,
    device="cuda",
    epochs=10,
    train_loader=None,
    val_loader=None,
    augmented_data = False
):
    

    if fixed_params is None:
        fixed_params = {}

    combinations = list(itertools.product(*param_grid.values()))
    param_names = list(param_grid.keys())
    results = []

    for hp_vals in combinations:
        hyperparams = dict(zip(param_names, hp_vals))
        config = {**fixed_params, **hyperparams}

        # Remove keys not in model_class.__init__ signature
        config = {**fixed_params, **hyperparams}
        model_init_keys = model_class.__init__.__code__.co_varnames
        model_config = {k: v for k, v in config.items() if k in model_init_keys}

        print(f"\n🔧 Training with config: {config}")

        # Modelo
        model = model_class(**model_config).to(device)

        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=config.get("learning_rate", 1e-3))
        criterion = nn.CrossEntropyLoss()
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            augment_data=augmented_data,
            device=device
        )

        # Entrenamiento
        best_val_f1 = 0
        best_train_f1 = 0
        best_train_loss = None
        best_val_loss = None

        for epoch in range(epochs):
            train_loss, train_f1 = trainer.train_epoch(train_loader)
            val_loss, val_f1 = trainer.evaluate(val_loader)

            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_train_f1 = train_f1
                best_train_loss = train_loss
                best_val_loss = val_loss

        results.append({
            'params': deepcopy(config),
            'train_f1': best_train_f1,
            'val_f1': best_val_f1,
            'train_loss': best_train_loss,
            'val_loss': best_val_loss
        })

    results.sort(key=lambda x: x['val_f1'], reverse=True)
    print("\n🏆 Top 3 configurations:")
    for i, res in enumerate(results[:3]):
        print(f"\n#{i+1} - Val F1: {res['val_f1']:.4f}")
        print("Hyperparameters:")
        for k, v in res['params'].items():
            print(f"  {k}: {v}")
        print(f"Train F1: {res['train_f1']:.4f} | Val Loss: {res['val_loss']:.4f}")

    return results
