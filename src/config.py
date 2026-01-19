import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

CONFIG = {
    # HYPERPARAMETERS
    # ----------------
    # k=30 got 89%. k=60 captures more fine-grained frequency details.
    "dimensions": [80],

    # Available distance functions from distances.py:
    # euclidean, cosine, ncc, wasserstein_fourier_distance, likelihood_ratio_distance,
    # kulczynski, soergel, lorentzian, canberra
    "distances": [
        "euclidean", 
        #"wasserstein_fourier_distance", 
        #"likelihood_ratio_distance",
        #"kulczynski",
        #"soergel",
        "lorentzian",
        "canberra"
    ],

    # Grid Search
    # We added C=10000 and C=50000 to allow tighter fits.
    "svm_param_grid": [
        {
            'kernel': ['rbf'],
            #OPTIONS: 1000, 5000, 10000, 20000
            'C': [1000, 5000, 10000, 20000],
            # OPTIONS: 'scale', 0.01, 0.001, 0.0001
            'gamma': [0.01],
            #OPTIONS: 'balanced', None
            'class_weight': ['balanced', None]
        },

        {
            'kernel': ['poly'],
            # OPTIONS:100, 1000, 5000 
            'C': [1000],
            #OPTIONS: 'scale', 0.01
            'gamma': [0.01],
            'degree': [2, 3],
            'coef0': [0, 1],
            'class_weight': ['balanced'],
        }
    ],

    # PATHS
    # -----
    "output_dir": os.path.join(project_root, "output"),
    "data_dir": os.path.join(project_root, "data")
}