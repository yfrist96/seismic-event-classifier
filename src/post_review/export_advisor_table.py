"""
Export a combined val + test table for the advisor.

Validation rows contain only metadata and true labels.
Test rows also include ensemble-with-calibration predictions.

Calibration: fits a BayesThresholdCalibrator on the single model's
validation decision scores, then applies that threshold to the
ensemble mean decision scores from the existing test_predictions.csv.

Usage: python -m src.post_review.export_advisor_table
"""

import os
import numpy as np
import pandas as pd
import h5py
import joblib
from sklearn.preprocessing import normalize

from src.classifier import BayesThresholdCalibrator

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "output", "ablation", "models")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

LABEL_MAP = {0: "earthquake", 1: "explosion"}


def to_frequency_domain(X):
    X_fft = np.fft.rfft(X, axis=1)
    X_mag = np.abs(X_fft)
    X_mag = np.log10(X_mag + 1e-6)
    X_flat = X_mag.reshape(X.shape[0], -1)
    return normalize(X_flat, axis=1, norm="l2")


def load_metadata(split_name):
    """Load event_id, station, and y from an HDF5 split file."""
    path = os.path.join(DATA_DIR, f"dataset_{split_name}.h5")
    with h5py.File(path, "r") as f:
        event_ids = f["event_id"][:]
        stations = np.array([s.decode() for s in f["station"][:]])
        y = f["y"][:]
    return event_ids, stations, y


def fit_calibrator_on_val():
    """Load the saved single model and fit the calibrator on val decision scores."""
    model_path = os.path.join(MODEL_DIR, "ablation_FFT_euclidean_k120_single.joblib")
    model_dict = joblib.load(model_path)

    # Load val waveforms and compute FFT features
    val_path = os.path.join(DATA_DIR, "dataset_val.h5")
    with h5py.File(val_path, "r") as f:
        X_val_raw = f["X"][:]
        y_val = f["y"][:]

    X_val_fft = to_frequency_domain(X_val_raw)
    X_val_emb = model_dict["fastmap"].transform(X_val_fft)
    X_val_scaled = model_dict["scaler"].transform(X_val_emb)
    val_decision = model_dict["svm"].decision_function(X_val_scaled)

    calibrator = BayesThresholdCalibrator()
    calibrator.fit(val_decision, y_val)
    print(f"Bayes-optimal threshold: {calibrator.threshold:.4f}")
    return calibrator


if __name__ == "__main__":
    # --- Validation rows (metadata + label only) ---
    print("Loading validation metadata...")
    val_event_ids, val_stations, val_y = load_metadata("val")

    df_val = pd.DataFrame({
        "event_id": val_event_ids,
        "datasplit": "validation",
        "station": val_stations,
        "label": [LABEL_MAP[y] for y in val_y],
        "ensemble_calibrated_prediction": "",
    })

    # --- Test rows (metadata + label + ensemble w/ calibration) ---
    print("Loading test predictions...")
    test_preds_path = os.path.join(OUTPUT_DIR, "test_predictions.csv")
    df_test_raw = pd.read_csv(test_preds_path)

    print("Fitting calibrator on validation set...")
    calibrator = fit_calibrator_on_val()

    ensemble_decision = df_test_raw["ensemble_mean_decision_score"].values
    ensemble_cal_pred = calibrator.predict(ensemble_decision)

    df_test = pd.DataFrame({
        "event_id": df_test_raw["event_id"],
        "datasplit": "test",
        "station": df_test_raw["station"],
        "label": df_test_raw["true_class"],
        "ensemble_calibrated_prediction": [LABEL_MAP[y] for y in ensemble_cal_pred],
    })

    # --- Combine and save ---
    df = pd.concat([df_val, df_test], ignore_index=True)

    output_path = os.path.join(OUTPUT_DIR, "advisor_table.csv")
    df.to_csv(output_path, index=False)

    print(f"\nSaved {len(df)} rows ({len(df_val)} val + {len(df_test)} test) to: {output_path}")

    # Summary stats for test predictions
    test_labels = df_test_raw["true_label"].values
    ens_cal_acc = np.mean(ensemble_cal_pred == test_labels)
    print(f"Ensemble + calibration test accuracy: {ens_cal_acc:.4f}")
