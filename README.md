# ECG Signal Classification using Deep Learning

## 1. Introduction

Electrocardiogram (ECG) signal classification is a critical task in healthcare, enabling the detection of arrhythmias and other cardiac abnormalities. This project focuses on classifying **univariate ECG time-series signals** into four clinically relevant categories:
- **Normal rhythm**
- **Atrial Fibrillation (AF)**
- **Other arrhythmias**
- **Noisy signals**

The dataset consists of **variable-length ECG recordings sampled at 300 Hz**, which introduces several challenges such as class imbalance, heterogeneous signal lengths, and noise. To address these issues, we designed an end-to-end deep learning pipeline covering:
- Exploratory data analysis
- Data preprocessing and normalization
- Model design and training
- Hyperparameter tuning
- Data augmentation
- Data reduction (compression)
- Final evaluation and test prediction generation

### Models

Two distinct deep learning architectures were implemented and evaluated:
1. **CNN–LSTM Hybrid Model**
   - Uses **Short-Time Fourier Transform (STFT)** for time–frequency representation.
   - Combines convolutional feature extraction with temporal modeling.
2. **Temporal Convolutional Network (TCN)**
   - Employs dilated causal convolutions for long-range temporal dependencies.

Model evaluation was performed using **stratified train/validation splits**, and **F1-score** was chosen as the primary metric due to class imbalance. Hyperparameters were optimized via grid search.

### Additional Experiments

Beyond model design, the project also investigates:
- **Data augmentation** (e.g., time stretching, noise injection) to improve generalization
- **Data reduction** through lossless and lossy compression to reduce storage requirements

### Key Findings

- The **CNN–LSTM model outperformed the TCN**, achieving an F1-score of **0.748** compared to **0.702**.
- Data augmentation showed **marginal improvement** on validation performance.
- **50% lossless compression** preserved classification performance with minimal F1-score degradation.

### Deliverables

The repository includes:
- Modular and reproducible code
- Detailed model evaluation and analysis
- Test-set predictions for three scenarios:
  - `base.csv` — baseline model
  - `augment.csv` — trained with augmented data
  - `reduced.csv` — trained with compressed data

Although the achieved performance is not yet optimal, the results highlight promising directions for future work, such as **ensemble learning** or **transformer-based architectures**.

---

## 2. How to Run the Code

### 2.1 Folder Structure

```
AMLS/
├── main.py                     # Main controller script
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
│   └── Modelling.ipynb
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
└── src/
    ├── augmentation/
    │   └── signal_augmentations.py
    ├── data/
    │   ├── load_data.py
    │   ├── lossless_compression.py
    │   ├── lossy_compression.py
    │   └── stratified_split.py
    ├── models/
    │   ├── model_1/
    │   │   ├── architecture.py
    │   │   └── config.yaml
    │   └── model_2/
    │       ├── architecture.py
    │       └── config.yaml
    └── hyperparameter_tunning/
        ├── grid_search.py
        ├── bayesian_opt.py
        └── model_trainer.py
```

---

### 2.2 Comprehensive Execution Guide

2.2.1 Environment Setup

**1. Create a virtual environment**

```bash
python -m venv ecg-env
```

Activate it:

```bash
# Linux / macOS
source ecg-env/bin/activate

# Windows
ecg-env\Scripts\activate
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

Key dependencies include **PyTorch**, **scikit-learn**, **pandas**, and **Jupyter**.

**3. Data preparation**

Download the dataset from **TU-Cloud** and place the files in:

```
data/raw/
├── X_train.bin   # Training signals
├── y_train.csv   # Training labels
└── X_test.bin    # Test signals (no labels)
```

---

### 2.2.2 Data Loading and Preparation

```bash
python main.py --prepare
```

**What it does:**
- Parses binary ECG signals
- Applies Z-score normalization per signal
- Performs a **90% / 10% stratified train–validation split**
- Pads shorter signals with zeros
- Creates PyTorch-compatible dataloaders

---

### 2.2.3 Hyperparameter Tuning

```bash
python main.py --hyperparameter_model1
python main.py --hyperparameter_model2
```

**What it does:**
- Runs grid search for the selected model
- Prints F1-score per epoch and parameter combination
- Returns the **top 3 hyperparameter configurations**
- Outputs a confusion matrix for the best model

---

### 2.2.4 Model Training and Evaluation

```bash
python main.py --evaluate_model1
python main.py --evaluate_model2
```

**What it does:**
- Trains the selected model for **50 epochs** using optimal hyperparameters
- Prints epoch-wise F1-scores
- Outputs:
  - Confusion matrix
  - Accuracy, precision, recall, and F1-score per class
- Saves test predictions to:

```
base.csv
```

---

### 2.2.5 Training with Augmented Data

```bash
python main.py --model1_augmented
python main.py --model2_augmented
```

**What it does:**
- Applies signal-level augmentations (e.g., noise injection, time stretching)
- Retrains models using the same hyperparameters
- Outputs evaluation metrics and confusion matrix
- Saves test predictions to:

```
augment.csv
```

---

### 2.2.6 Training with Compressed Data

```bash
python main.py --lossless_model1
python main.py --lossless_model2
```

**What it does:**
- Applies **50% lossless compression** to the training dataset
- Retrains models on compressed signals
- Evaluates performance and reports metrics
- Saves test predictions to:

```
reduced.csv
```

---

📌 **Note:** All experiments are controlled via `main.py`, ensuring reproducibility and consistency across training, augmentation, and compression scenarios.

---
