# ECG classification project

## Project Structure

```


AMLS/
├── main.py                 # Main controller script
├── requirements.txt       
├── base.csv        
├── preparation_1/          
│   ├── data_preparation.py
│   └── Exploring.ipynb
├── Modelling_2/            
│   ├── model1_training.py
│   ├── model1_evaluate.py
│   ├── model2_training.py
│   ├── model2_evaluate.py
│  └── Modelling.ipynb
├── Data_augmentation_3/    
│   ├── model1_augmented.py
│   ├── model2_augmented.py
│   └── Data_augmentation.ipynb
├── Data_reduction_4/       
│   ├── data_lossless.py
│   └── Reduction.ipynb
├── data/                   
│   ├── raw/                 
│   │   ├── X_train.bin     
│   │   ├── y_train.csv     
│   │   └── X_test.bin      
│   └── processed/          
│       └── compressed.bin     
├── src/                    
│   ├── augmentation/       
│   │   └── signal_augmentations.py  
│   ├── data/               
│   │   ├── load_data.py           
│   │   ├── lossless_compression.py 
│   │   ├── lossy_compression.py    
│   │   └── stratified_split.py     
│   └── models/              
│       ├── model_1/        
│       │   ├── architecture.py
│       │   └── config.yaml
│       ├── model_2/        
│       │   ├── architecture.py
│       │   └── config.yaml
│       ├── hyperparameter_tunning/ 
│       │   ├── grid_search.py
│       │   └── bayesian_opt.py
│       └── model_trainer.py    


````

## Description

- `read_binary(path)`: Reads a ragged array from a binary file where each sequence is preceded by its size (int32), followed by 16-bit signed integer data.
- `load_train_data()`: Loads training sequences (`X_train.bin`) and their labels (`y_train.csv`).
- `load_test_data()`: Loads test sequences (`X_test.bin`).

The root directory of the project is automatically detected relative to the location of the script.

## Usage

```python
from src.data.load_data import load_train_data, load_test_data

X_train, y_train = load_train_data()
X_test = load_test_data()
````

## Notes

* Make sure the data files are placed in the `data/raw/` folder inside the project root as shown above.
* Training labels CSV file does not have a header.
