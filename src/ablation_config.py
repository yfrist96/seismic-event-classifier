import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

ABLATION_CONFIG = {
    # --- Axes of ablation ---

    # 1. Distance metrics to compare
    "distances": [
        "euclidean",
        "lorentzian",
        "canberra",
    ],

    # 2. FastMap dimensions to sweep
    "dimensions": [2, 10, 30, 60, 80, 120],

    # 3. Feature domain: with and without FFT
    "use_fft": [True, False],

    # 4. Ensemble vs single model
    "use_ensemble": [True, False],
    "ensemble_n_models": 5,

    # NumPy seed for FastMap's random pivot selection (set once at the start of the
    # sweep; each model and each ensemble voter still draws its own pivots). The
    # published results came from an unseeded run, so a seeded rerun is repeatable
    # but differs slightly from them. Set to None for unseeded behavior.
    "seed": 42,

    # SVM param grid (same for all — tuned on val)
    "svm_param_grid": [
        {
            'kernel': ['rbf'],
            'C': [1000, 5000, 10000, 20000],
            'gamma': [0.01],
            'class_weight': ['balanced', None],
        },
        {
            'kernel': ['poly'],
            'C': [1000],
            'gamma': [0.01],
            'degree': [2, 3],
            'coef0': [0, 1],
            'class_weight': ['balanced'],
        },
    ],

    # Paths
    "output_dir": os.path.join(project_root, "output", "ablation"),
    "data_dir": os.path.join(project_root, "data"),
}
