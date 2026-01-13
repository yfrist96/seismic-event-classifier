import h5py
import numpy as np
import os

def load_split(file_path):
    """
    Loads X, y, and event_id from a single HDF5 file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find file: {file_path}")

    with h5py.File(file_path, 'r') as f:
        # Load data into memory
        X = f['X'][:]
        y = f['y'][:]

        # Load event_ids if available (good for debugging leakage)
        if 'event_id' in f:
            event_ids = f['event_id'][:]
        else:
            event_ids = None

    print(f"Loaded {file_path}: X shape={X.shape}, y shape={y.shape}")
    return X, y, event_ids


def load_data(data_dir="./seismic_datasets"):
    """
    Loads the pre-split Train, Validation, and Test datasets.
    """
    print(f"Loading datasets from {data_dir}...")

    # Define paths based on your folder structure
    train_path = os.path.join(data_dir, "dataset_train.h5")
    val_path = os.path.join(data_dir, "dataset_val.h5")
    test_path = os.path.join(data_dir, "dataset_test.h5")

    # Load each split
    X_train, y_train, ids_train = load_split(train_path)
    X_val, y_val, ids_val = load_split(val_path)
    X_test, y_test, ids_test = load_split(test_path)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)