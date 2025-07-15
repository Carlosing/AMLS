# main.py
from pathlib import Path
import argparse



def main():
    parser = argparse.ArgumentParser(description="ECG Classification Pipeline")
    parser.add_argument('--prepare', action='store_true', help='Run data preparation')
    parser.add_argument('--hyperparameter_model1', action='store_true', help='Train model')
    parser.add_argument('--hyperparameter_model2', action='store_true', help='Train model')
    parser.add_argument('--evaluate_model1', action='store_true', help='Evaluate model')
    parser.add_argument('--evaluate_model2', action='store_true', help='Evaluate model')
    parser.add_argument('--augmented_model1', action='store_true', help='Evaluate model')
    parser.add_argument('--augmented_model2', action='store_true', help='Evaluate model')
    
    args = parser.parse_args()

    if args.prepare:
        from preparation_1.data_preparation import main as prepare_main
        prepare_main()
    if args.hyperparameter_model1:
        from Modelling_2.model1_training import main as hyper
        from preparation_1.data_preparation import main as data
        
        train_loader, val_loader,signal_size = data()
        hyper(train_loader, val_loader, signal_size)
    if args.hyperparameter_model2:
        from Modelling_2.model2_training import main as hyper
        from preparation_1.data_preparation import main as data
        
        train_loader, val_loader,signal_size = data()
        hyper(train_loader, val_loader)
        
    if args.evaluate_model1:
        from Modelling_2.model1_evaluate import main as eval
        from preparation_1.data_preparation import main as data
        
        train_loader, val_loader,signal_size = data()
        eval(train_loader, val_loader, signal_size)
        
    if args.evaluate_model2:
        from Modelling_2.model2_evaluate import main as eval
        from preparation_1.data_preparation import main as data
        
        train_loader, val_loader,signal_size = data()
        eval(train_loader, val_loader, signal_size)
        
    if args.augmented_model1:
        from Data_augmentation_3.model1_augmented import main as eval
        from preparation_1.data_preparation import main as data
        
        train_loader, val_loader,signal_size = data()
        eval(train_loader, val_loader, signal_size)
        
    if args.augmented_model2:
        from Data_augmentation_3.model2_augmented import main as eval
        from preparation_1.data_preparation import main as data
        
        train_loader, val_loader,signal_size = data()
        eval(train_loader, val_loader, signal_size)

if __name__ == "__main__":
    main()
