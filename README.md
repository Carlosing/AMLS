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
