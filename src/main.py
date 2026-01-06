import os
import json
import pickle
import numpy as np
from datetime import datetime
from sklearn.metrics import classification_report

from src.dataloader import load_data
from src.classifier import FastMapSVMClassifier, BaselineClassifier

# GET PROJECT ROOT AUTOMATICALLY
# This ensures it works no matter where you run it from
current_dir = os.path.dirname(os.path.abspath(__file__)) # .../src
project_root = os.path.dirname(current_dir)              # .../seismic_project

# CONFIGURATION
CONFIG = {
    "dimensions": [2, 5, 10, 20],
    "distances": ["euclidean", "correlation"],
    "output_dir": os.path.join(project_root, "output"),
    "data_dir": os.path.join(project_root, "data")       # <--- FIXED: Points to 'data' folder
}

def run_experiments():
    # 1. Setup
    os.makedirs(f"{CONFIG['output_dir']}/models", exist_ok=True)
    os.makedirs(f"{CONFIG['output_dir']}/results", exist_ok=True)
    os.makedirs(f"{CONFIG['output_dir']}/embeddings", exist_ok=True)

    # 2. Load Data
    # Important: This expects dataset_train.h5, dataset_val.h5, dataset_test.h5 inside 'data/'
    print(f"Loading Data from: {CONFIG['data_dir']}...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data(CONFIG['data_dir'])

    # =======================================================
    # REMOVE BEFORE RUNNING FULL EXPERIMENTS
    # =======================================================
    # =======================================================
    # REMOVE BEFORE RUNNING FULL EXPERIMENTS
    # =======================================================
    # --- DEBUG START: Run on just 100 samples to test pipeline ---
    print("!!! DEBUG MODE: Shuffling and truncating data to 100 samples !!!")

    # 1. Shuffle Train Data (to ensure we get both classes)
    idx_train = np.random.permutation(len(X_train))
    X_train = X_train[idx_train][:300]
    y_train = y_train[idx_train][:300]

    # 2. Shuffle Test Data
    idx_test = np.random.permutation(len(X_test))
    X_test = X_test[idx_test][:300]
    y_test = y_test[idx_test][:300]

    # Verify we actually have 2 classes
    print(f"Debug Class Balance: {np.unique(y_train, return_counts=True)}")
    # --- DEBUG END ---
    # =======================================================
    # =======================================================

    # 3. Run Baseline (Random Forest)
    print("\n=== Running Baseline Model ===")
    baseline = BaselineClassifier()
    baseline.fit(X_train, y_train)
    y_pred_base = baseline.predict(X_test)

    base_report = classification_report(y_test, y_pred_base, output_dict=True)
    with open(f"{CONFIG['output_dir']}/results/baseline_report.json", "w") as f:
        json.dump(base_report, f, indent=4)
    print("Baseline Accuracy:", base_report['accuracy'])

    # 4. Run FastMap Experiments
    for dist in CONFIG['distances']:
        for k in CONFIG['dimensions']:
            exp_name = f"fastmap_{dist}_k{k}"
            print(f"\n=== Running {exp_name} ===")

            # Train
            model = FastMapSVMClassifier(k=k, dist_func=dist)
            start_time = datetime.now()
            model.fit(X_train, y_train)
            duration = (datetime.now() - start_time).total_seconds()

            # Predict
            y_pred = model.predict(X_test)

            # Metrics
            report = classification_report(y_test, y_pred, output_dict=True)
            report['training_time_sec'] = duration

            print(f"   Accuracy: {report['accuracy']:.4f}")

            # SAVE ARTIFACTS
            # A. Config/Metrics
            with open(f"{CONFIG['output_dir']}/results/{exp_name}_report.json", "w") as f:
                json.dump(report, f, indent=4)

            # B. Model
            with open(f"{CONFIG['output_dir']}/models/{exp_name}.pkl", "wb") as f:
                pickle.dump(model, f)

            # C. Embeddings
            train_emb = model.fastmap.fit_transform(X_train)
            np.save(f"{CONFIG['output_dir']}/embeddings/{exp_name}_train.npy", train_emb)

if __name__ == "__main__":
    run_experiments()