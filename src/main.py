import os
import json
import numpy as np
from datetime import datetime
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from src.dataloader import load_data
from src.classifier import FastMapSVMClassifier, BaselineClassifier
from src.config import CONFIG  # <--- Imported from your new file


def to_frequency_domain(X):
    """
    Converts raw time-series seismic data to Frequency Domain (FFT).
    X shape: (N, 600, 3) -> (N, Flattened_FFT_Features)
    """
    # 1. Apply FFT along time axis
    X_fft = np.fft.rfft(X, axis=1)

    # 2. Take Magnitude (Energy)
    X_mag = np.abs(X_fft)

    # 3. Log scale (Decibels) - handles massive energy differences
    X_mag = np.log10(X_mag + 1e-6)

    # 4. Flatten for FastMap
    return X_mag.reshape(X.shape[0], -1)


def run_experiments():
    # 1. Setup Directories
    os.makedirs(f"{CONFIG['output_dir']}/models", exist_ok=True)
    os.makedirs(f"{CONFIG['output_dir']}/results", exist_ok=True)
    os.makedirs(f"{CONFIG['output_dir']}/embeddings", exist_ok=True)

    # 2. Load Data
    print(f"Loading Data from: {CONFIG['data_dir']}...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data(CONFIG['data_dir'])

    # 3. Run Baseline (Random Forest)
    # Must run on RAW DATA before FFT conversion
    print("\n=== Running Baseline Model ===")
    try:
        baseline = BaselineClassifier()
        baseline.fit(X_train, y_train)
        y_pred_base = baseline.predict(X_test)
        base_report = classification_report(y_test, y_pred_base, output_dict=True)
        print(f"Baseline Accuracy: {base_report['accuracy']:.4f}")

        with open(f"{CONFIG['output_dir']}/results/baseline_report.json", "w") as f:
            json.dump(base_report, f, indent=4)
    except Exception as e:
        print(f"Baseline failed (likely due to data format mismatch if debugging): {e}")

    # --- NEW STEP: Transform to Frequency Domain ---
    print("\nConverting data to Frequency Domain (FFT)...")
    X_train = to_frequency_domain(X_train)
    X_val = to_frequency_domain(X_val)
    X_test = to_frequency_domain(X_test)
    print(f"New Feature Shape: {X_train.shape}")
    # -----------------------------------------------

    # 4. Run FastMap + SVM Grid Search
    for dist in CONFIG['distances']:
        for k in CONFIG['dimensions']:
            exp_name = f"fastmap_FFT_{dist}_k{k}_tuned"
            print(f"\n=== Running {exp_name} (with Grid Search) ===")

            # A. Generate Embeddings first
            temp_model = FastMapSVMClassifier(k=k, dist_func=dist)

            print(f"  [FastMap] Generating embeddings for k={k}...")
            start_time = datetime.now()
            X_train_emb = temp_model.fastmap.fit_transform(X_train)
            X_test_emb = temp_model.fastmap.transform(X_test)
            duration = (datetime.now() - start_time).total_seconds()
            print(f"  [FastMap] Done in {duration:.1f}s")

            # Save Embeddings
            np.save(f"{CONFIG['output_dir']}/embeddings/{exp_name}_train.npy", X_train_emb)

            # B. Standardize (Crucial for SVM)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_emb)
            X_test_scaled = scaler.transform(X_test_emb)

            # C. Grid Search for SVM
            print("  [GridSearch] Tuning SVM parameters...")
            # Using the param_grid from CONFIG
            grid = GridSearchCV(SVC(), CONFIG['svm_param_grid'], refit=True, verbose=1, cv=3, n_jobs=-1)
            grid.fit(X_train_scaled, y_train)

            print(f"  [GridSearch] Best Params: {grid.best_params_}")

            # D. Evaluate Best Model
            best_model = grid.best_estimator_
            y_pred = best_model.predict(X_test_scaled)

            report = classification_report(y_test, y_pred, output_dict=True)
            report['best_params'] = grid.best_params_

            print(f"  [RESULT] Tuned Accuracy: {report['accuracy']:.4f}")

            # Save Results
            with open(f"{CONFIG['output_dir']}/results/{exp_name}_report.json", "w") as f:
                json.dump(report, f, indent=4)


if __name__ == "__main__":
    run_experiments()