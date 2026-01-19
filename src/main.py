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
from sklearn.preprocessing import normalize
from scipy.stats import mode


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

    X_flat = X_mag.reshape(X.shape[0], -1)

    X_norm = normalize(X_flat, axis=1, norm='l2')

    # 4. Flatten for FastMap
    #return X_mag.reshape(X.shape[0], -1)
    return X_norm


def run_experiments():
    # 1. Setup Directories
    os.makedirs(f"{CONFIG['output_dir']}/models", exist_ok=True)
    os.makedirs(f"{CONFIG['output_dir']}/results", exist_ok=True)
    os.makedirs(f"{CONFIG['output_dir']}/embeddings", exist_ok=True)

    # 2. Load Data
    print(f"Loading Data from: {CONFIG['data_dir']}...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data(CONFIG['data_dir'])

    unique, counts = np.unique(y_train, return_counts=True)
    print(f"\nClass Distribution in Training Set:")
    for label, count in zip(unique, counts):
        print(f"  Class {label}: {count} samples ({count/len(y_train)*100:.1f}%)")

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

            # Temp Added for esemble method 
            best_params = grid.best_params_ #
            print(f"  [GridSearch] Best Params: {grid.best_params_}")



            # D. Evaluate Best Model
            best_model = grid.best_estimator_
            y_pred_single = best_model.predict(X_test_scaled)

            report_single = classification_report(y_test, y_pred_single, output_dict=True)
            report_single['best_params'] = grid.best_params_

            print(f"  [RESULT] Tuned Accuracy: {report_single['accuracy']:.4f}")

            # Save Results
            with open(f"{CONFIG['output_dir']}/results/{exp_name}_report.json", "w") as f:
                json.dump(report_single, f, indent=4)
            
            # --- PHASE B: Ensemble (The Upgrade) ---

            print(f"  [Ensemble] Running Ensemble for {exp_name}...")
            
            y_pred_ensemble = train_and_evaluate_ensemble(
                X_train, y_train, X_test, y_test, 
                k=k, dist=dist, 
                best_params=best_params, # Reuse the optimized C/gamma
                n_models=5
            )
            
            # Evaluate Ensemble
            report_ensemble = classification_report(y_test, y_pred_ensemble, output_dict=True)
            print(f"  [RESULT] Single: {report_single['accuracy']:.4f} | Ensemble: {report_ensemble['accuracy']:.4f}")

            # Save Ensemble Results
            with open(f"{CONFIG['output_dir']}/results/{exp_name}_ensemble_report.json", "w") as f:
                json.dump(report_ensemble, f, indent=4)

def train_and_evaluate_ensemble(X_train, y_train, X_test, y_test, k, dist, best_params, n_models=5):
    """
    Trains multiple FastMap models using the BEST parameters found by GridSearch,
    then aggregates their predictions via Majority Vote.
    """
    print(f"  [Ensemble] Training {n_models} voters using params: {best_params}...")
    
    # Matrix to store predictions: (n_models, n_test_samples)
    all_preds = np.zeros((n_models, len(y_test)))
    
    for i in range(n_models):
        # 1. New Random FastMap Projection
        # We re-initialize to get DIFFERENT random pivots
        model_wrapper = FastMapSVMClassifier(k=k, dist_func=dist) 
        
        # Embed
        X_train_emb = model_wrapper.fastmap.fit_transform(X_train)
        X_test_emb = model_wrapper.fastmap.transform(X_test)
        
        # 2. Standardize (Each projection has different scale)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_emb)
        X_test_scaled = scaler.transform(X_test_emb)
        
        # 3. Train SVM with the BEST parameters found earlier
        clf = SVC(**best_params) # Unpack dict (e.g., C=1000, gamma='scale')
        clf.fit(X_train_scaled, y_train)
        
        # 4. Predict
        all_preds[i] = clf.predict(X_test_scaled)
        
    # 5. Majority Vote (Mode)
    # mode returns [[most_frequent_val], [count]]
    final_preds, _ = mode(all_preds, axis=0, keepdims=False)
    
    # Flatten final_preds (it comes out as 2D row vector)
    final_preds = final_preds.ravel().astype(int)
    
    return final_preds

if __name__ == "__main__":
    run_experiments()