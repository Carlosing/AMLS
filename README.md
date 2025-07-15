# ECG classification project

## Project Structure

```


project\_root/
│
├── data/
│   └── raw/
│       ├── X\_train/
│       │   └── X\_train.bin
│       ├── y\_train.csv
│       └── X\_test/
│           └── X\_test.bin
│
├── src/
│   └── data/
│       └── load\_data.py   # This script
│
└── README.md


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
