"""
Export test set predictions with metadata for Ittai.
Uses the best single model (FFT + Euclidean + k=120) and also
trains a 5-voter ensemble to provide ensemble predictions + confidence.

Usage: python -m src.export_test_predictions
"""

import os
import numpy as np
import pandas as pd
import h5py
import joblib
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.svm import SVC
from scipy.stats import mode

from src.dataloader import load_data
from src.classifier import FastMapSVMClassifier

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "output", "ablation", "models")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

LABEL_MAP = {0: 'earthquake', 1: 'explosion'}
N_ENSEMBLE = 5


def to_frequency_domain(X):
    X_fft = np.fft.rfft(X, axis=1)
    X_mag = np.abs(X_fft)
    X_mag = np.log10(X_mag + 1e-6)
    X_flat = X_mag.reshape(X.shape[0], -1)
    return normalize(X_flat, axis=1, norm='l2')


def load_test_metadata():
    """Load all metadata from the test HDF5 file."""
    with h5py.File(os.path.join(DATA_DIR, "dataset_test.h5"), 'r') as f:
        event_ids = f['event_id'][:]
        stations = np.array([s.decode() for s in f['station'][:]])
        y = f['y'][:]
    return event_ids, stations, y


def predict_single_model(X_test_fft):
    """Load saved single model and predict with decision function scores."""
    model_path = os.path.join(MODEL_DIR, "ablation_FFT_euclidean_k120_single.joblib")
    model_dict = joblib.load(model_path)

    X_emb = model_dict['fastmap'].transform(X_test_fft)
    X_scaled = model_dict['scaler'].transform(X_emb)

    y_pred = model_dict['svm'].predict(X_scaled)
    decision_vals = model_dict['svm'].decision_function(X_scaled)

    return y_pred, decision_vals


def predict_ensemble(X_train_fft, y_train, X_val_fft, y_val, X_test_fft, best_params):
    """Train a fresh 5-voter ensemble and predict with vote counts."""
    X_trainval = np.vstack([X_train_fft, X_val_fft])
    y_trainval = np.concatenate([y_train, y_val])

    all_preds = np.zeros((N_ENSEMBLE, len(X_test_fft)))
    all_decision = np.zeros((N_ENSEMBLE, len(X_test_fft)))

    print(f"  Training {N_ENSEMBLE} ensemble voters...")
    for i in range(N_ENSEMBLE):
        model = FastMapSVMClassifier(k=120, dist_func='euclidean')
        X_trainval_emb = model.fastmap.fit_transform(X_trainval)
        X_test_emb = model.fastmap.transform(X_test_fft)

        scaler = StandardScaler()
        X_trainval_scaled = scaler.fit_transform(X_trainval_emb)
        X_test_scaled = scaler.transform(X_test_emb)

        clf = SVC(**best_params)
        clf.fit(X_trainval_scaled, y_trainval)
        all_preds[i] = clf.predict(X_test_scaled)
        all_decision[i] = clf.decision_function(X_test_scaled)
        print(f"    Voter {i+1}/{N_ENSEMBLE} done")

    # Majority vote
    final_preds, _ = mode(all_preds, axis=0, keepdims=False)
    final_preds = final_preds.ravel().astype(int)

    # Confidence: fraction of voters agreeing with the majority
    vote_counts = np.sum(all_preds == final_preds[np.newaxis, :], axis=0)
    vote_confidence = vote_counts / N_ENSEMBLE

    # Mean decision function across voters
    mean_decision = np.mean(all_decision, axis=0)

    return final_preds, vote_confidence, mean_decision


if __name__ == "__main__":
    print("Loading data...")
    (X_train_raw, y_train), (X_val_raw, y_val), (X_test_raw, y_test) = load_data(DATA_DIR)
    event_ids, stations, y_true = load_test_metadata()

    print("Computing FFT features...")
    X_train_fft = to_frequency_domain(X_train_raw)
    X_val_fft = to_frequency_domain(X_val_raw)
    X_test_fft = to_frequency_domain(X_test_raw)

    # Single model predictions
    print("\nSingle model predictions...")
    single_pred, single_decision = predict_single_model(X_test_fft)

    # Load best params from saved model for ensemble
    model_dict = joblib.load(os.path.join(MODEL_DIR, "ablation_FFT_euclidean_k120_single.joblib"))
    best_params = model_dict['params']

    # Ensemble predictions
    print("\nEnsemble predictions...")
    ens_pred, ens_vote_conf, ens_mean_decision = predict_ensemble(
        X_train_fft, y_train, X_val_fft, y_val, X_test_fft, best_params
    )

    # Build output DataFrame
    df = pd.DataFrame({
        'window_index': np.arange(len(y_true)),
        'event_id': event_ids,
        'station': stations,
        'true_label': y_true,
        'true_class': [LABEL_MAP[y] for y in y_true],
        'single_pred_label': single_pred,
        'single_pred_class': [LABEL_MAP[y] for y in single_pred],
        'single_decision_score': np.round(single_decision, 4),
        'ensemble_pred_label': ens_pred,
        'ensemble_pred_class': [LABEL_MAP[y] for y in ens_pred],
        'ensemble_vote_confidence': np.round(ens_vote_conf, 2),
        'ensemble_mean_decision_score': np.round(ens_mean_decision, 4),
    })

    output_path = os.path.join(OUTPUT_DIR, "test_predictions.csv")
    df.to_csv(output_path, index=False)

    # Print summary
    single_acc = np.mean(single_pred == y_true)
    ens_acc = np.mean(ens_pred == y_true)
    print(f"\nSingle model accuracy: {single_acc:.4f}")
    print(f"Ensemble accuracy:     {ens_acc:.4f}")
    print(f"\nSaved {len(df)} rows to: {output_path}")
