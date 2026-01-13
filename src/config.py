import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

CONFIG = {
    # HYPERPARAMETERS
    # ----------------
    # k=30 got 89%. k=60 captures more fine-grained frequency details.
    "dimensions": [70],

    # Euclidean is mandatory for FFT (Magnitude) data
    "distances": ["euclidean"],

    # Aggressive Grid Search
    # We added C=10000 and C=50000 to allow tighter fits.
    "svm_param_grid": {
        'C': [1000, 5000, 10000, 20000, 50000],
        'gamma': ['scale', 0.001, 0.0001],
        'kernel': ['rbf']
    },

    # PATHS
    # -----
    "output_dir": os.path.join(project_root, "output"),
    "data_dir": os.path.join(project_root, "data")
}