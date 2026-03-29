"""
Ablation study for the seismic event classifier.
Sweeps over: distance metrics, FastMap dimensions, FFT vs time domain, single vs ensemble.
All experiments use the same data split: train for training, val for tuning, test for final eval.
"""

import os
import json
import numpy as np
import joblib
from datetime import datetime
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.model_selection import ParameterGrid
from scipy.stats import mode
from src.dataloader import load_data
from src.classifier import FastMapSVMClassifier, BaselineClassifier
from src.ablation_config import ABLATION_CONFIG as CONFIG


def to_frequency_domain(X):
    X_fft = np.fft.rfft(X, axis=1)
    X_mag = np.abs(X_fft)
    X_mag = np.log10(X_mag + 1e-6)
    X_flat = X_mag.reshape(X.shape[0], -1)
    return normalize(X_flat, axis=1, norm='l2')


def flatten_time_domain(X):
    return X.reshape(X.shape[0], -1)


def run_single_experiment(X_train, y_train, X_val, y_val, X_test, y_test,
                          dist, k, exp_name, exp_idx):
    """Run a single FastMap+SVM experiment: tune on val, retrain on train+val, eval on test."""
    temp_model = FastMapSVMClassifier(k=k, dist_func=dist)

    print(f"  [FastMap] Generating embeddings for k={k}...")
    start_time = datetime.now()
    X_train_emb = temp_model.fastmap.fit_transform(X_train)
    X_val_emb = temp_model.fastmap.transform(X_val)
    X_test_emb = temp_model.fastmap.transform(X_test)
    duration = (datetime.now() - start_time).total_seconds()
    print(f"  [FastMap] Done in {duration:.1f}s")

    # Save embeddings (train + test) for visualization later
    emb_dir = os.path.join(CONFIG['output_dir'], "embeddings")
    np.save(os.path.join(emb_dir, f"{exp_name}_train.npy"), X_train_emb)
    np.save(os.path.join(emb_dir, f"{exp_name}_test.npy"), X_test_emb)

    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_emb)
    X_val_scaled = scaler.transform(X_val_emb)

    # Tune on validation set
    print(f"  [GridSearch] Tuning SVM on validation set...")
    param_combinations = list(ParameterGrid(CONFIG['svm_param_grid']))
    best_score = -1
    best_params = None

    for i, params in enumerate(param_combinations):
        clf = SVC(**params)
        clf.fit(X_train_scaled, y_train)
        val_score = accuracy_score(y_val, clf.predict(X_val_scaled))
        if val_score > best_score:
            best_score = val_score
            best_params = params

    print(f"  [GridSearch] Best Params: {best_params}")
    print(f"  [GridSearch] Best Val Accuracy: {best_score:.4f}")

    # Retrain on train+val
    X_trainval_emb = np.vstack([X_train_emb, X_val_emb])
    y_trainval = np.concatenate([y_train, y_val])

    scaler_final = StandardScaler()
    X_trainval_scaled = scaler_final.fit_transform(X_trainval_emb)
    X_test_scaled = scaler_final.transform(X_test_emb)

    final_model = SVC(**best_params)
    final_model.fit(X_trainval_scaled, y_trainval)

    y_pred = final_model.predict(X_test_scaled)
    report = classification_report(y_test, y_pred, output_dict=True)
    test_acc = report['accuracy']
    print(f"  [RESULT] Test Accuracy: {test_acc:.4f}")

    # Save model
    model_path = os.path.join(CONFIG['output_dir'], "models", f"{exp_name}.joblib")
    joblib.dump({
        'fastmap': temp_model.fastmap,
        'scaler': scaler_final,
        'svm': final_model,
        'params': best_params,
        'val_accuracy': best_score,
        'test_accuracy': test_acc,
    }, model_path)

    return report, best_params, best_score


def run_ensemble(X_train, y_train, X_val, y_val, X_test, y_test,
                 dist, k, best_params, n_models, exp_name, exp_idx):
    """Run ensemble of n_models FastMap+SVM voters with majority vote."""
    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])

    print(f"  [Ensemble] Training {n_models} voters...")
    all_preds = np.zeros((n_models, len(y_test)))

    for i in range(n_models):
        model_wrapper = FastMapSVMClassifier(k=k, dist_func=dist)
        X_trainval_emb = model_wrapper.fastmap.fit_transform(X_trainval)
        X_test_emb = model_wrapper.fastmap.transform(X_test)

        scaler = StandardScaler()
        X_trainval_scaled = scaler.fit_transform(X_trainval_emb)
        X_test_scaled = scaler.transform(X_test_emb)

        clf = SVC(**best_params)
        clf.fit(X_trainval_scaled, y_trainval)
        all_preds[i] = clf.predict(X_test_scaled)

    final_preds, _ = mode(all_preds, axis=0, keepdims=False)
    final_preds = final_preds.ravel().astype(int)

    report = classification_report(y_test, final_preds, output_dict=True)
    ensemble_acc = report['accuracy']
    print(f"  [Ensemble RESULT] Test Accuracy: {ensemble_acc:.4f}")

    return report


def run_ablation():
    np.random.seed(42)
    # Setup directories
    for subdir in ["models", "results", "embeddings"]:
        os.makedirs(os.path.join(CONFIG['output_dir'], subdir), exist_ok=True)

    # Load data
    print(f"Loading Data from: {CONFIG['data_dir']}...")
    (X_train_raw, y_train), (X_val_raw, y_val), (X_test_raw, y_test) = load_data(CONFIG['data_dir'])

    print(f"Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")

    # Precompute both feature domains
    print("Precomputing FFT features...")
    X_train_fft = to_frequency_domain(X_train_raw)
    X_val_fft = to_frequency_domain(X_val_raw)
    X_test_fft = to_frequency_domain(X_test_raw)

    print("Precomputing time-domain features...")
    X_train_time = flatten_time_domain(X_train_raw)
    X_val_time = flatten_time_domain(X_val_raw)
    X_test_time = flatten_time_domain(X_test_raw)

    # --- Baseline ---
    print("\n=== Baseline (Random Forest) ===")
    baseline = BaselineClassifier()
    baseline.fit(X_train_raw, y_train)
    y_pred_val = baseline.predict(X_val_raw)
    y_pred_test = baseline.predict(X_test_raw)
    base_val_acc = accuracy_score(y_val, y_pred_val)
    base_report = classification_report(y_test, y_pred_test, output_dict=True)
    print(f"  Val: {base_val_acc:.4f} | Test: {base_report['accuracy']:.4f}")

    with open(os.path.join(CONFIG['output_dir'], "results", "baseline_report.json"), "w") as f:
        json.dump(base_report, f, indent=4)

    # --- Summary table ---
    summary = []
    summary.append({
        "experiment": "baseline_RF",
        "domain": "time_stats",
        "distance": "N/A",
        "k": "N/A",
        "model": "single",
        "val_accuracy": round(base_val_acc, 4),
        "test_accuracy": round(base_report['accuracy'], 4),
        "test_f1_eq": round(base_report['0']['f1-score'], 4),
        "test_f1_ex": round(base_report['1']['f1-score'], 4),
    })

    # --- Main ablation loop ---
    exp_idx = 0
    total = len(CONFIG['distances']) * len(CONFIG['dimensions']) * len(CONFIG['use_fft'])
    exp_times = []
    ablation_start = datetime.now()
    print(f"\nStarting ablation: {total} base experiments (x2 with ensemble)\n")

    for use_fft in CONFIG['use_fft']:
        domain = "FFT" if use_fft else "time"
        X_train = X_train_fft if use_fft else X_train_time
        X_val = X_val_fft if use_fft else X_val_time
        X_test = X_test_fft if use_fft else X_test_time

        for dist in CONFIG['distances']:
            for k in CONFIG['dimensions']:
                exp_start = datetime.now()

                # --- Single model ---
                exp_name = f"ablation_{domain}_{dist}_k{k}_single"
                print(f"\n=== [{exp_idx+1}/{total}] {exp_name} ===")

                report, best_params, val_acc = run_single_experiment(
                    X_train, y_train, X_val, y_val, X_test, y_test,
                    dist, k, exp_name, exp_idx,
                )

                with open(os.path.join(CONFIG['output_dir'], "results", f"{exp_name}_report.json"), "w") as f:
                    json.dump(report, f, indent=4)

                summary.append({
                    "experiment": exp_name,
                    "domain": domain,
                    "distance": dist,
                    "k": k,
                    "model": "single",
                    "val_accuracy": round(val_acc, 4),
                    "test_accuracy": round(report['accuracy'], 4),
                    "test_f1_eq": round(report['0']['f1-score'], 4),
                    "test_f1_ex": round(report['1']['f1-score'], 4),
                    "best_params": {pk: str(pv) for pk, pv in best_params.items()},
                })

                # --- Ensemble ---
                ens_name = f"ablation_{domain}_{dist}_k{k}_ensemble"
                print(f"  --- {ens_name} ---")

                ens_report = run_ensemble(
                    X_train, y_train, X_val, y_val, X_test, y_test,
                    dist, k, best_params, CONFIG['ensemble_n_models'],
                    ens_name, exp_idx,
                )

                with open(os.path.join(CONFIG['output_dir'], "results", f"{ens_name}_report.json"), "w") as f:
                    json.dump(ens_report, f, indent=4)

                summary.append({
                    "experiment": ens_name,
                    "domain": domain,
                    "distance": dist,
                    "k": k,
                    "model": "ensemble",
                    "val_accuracy": round(val_acc, 4),
                    "test_accuracy": round(ens_report['accuracy'], 4),
                    "test_f1_eq": round(ens_report['0']['f1-score'], 4),
                    "test_f1_ex": round(ens_report['1']['f1-score'], 4),
                    "best_params": {pk: str(pv) for pk, pv in best_params.items()},
                })

                # --- Progress tracking ---
                exp_elapsed = (datetime.now() - exp_start).total_seconds()
                exp_times.append(exp_elapsed)
                exp_idx += 1

                avg_time = sum(exp_times) / len(exp_times)
                remaining = total - exp_idx
                eta_seconds = avg_time * remaining
                eta_min = eta_seconds / 60
                total_elapsed = (datetime.now() - ablation_start).total_seconds()

                print(f"\n  [PROGRESS] {exp_idx}/{total} done | "
                      f"This: {exp_elapsed:.0f}s | Avg: {avg_time:.0f}s | "
                      f"Elapsed: {total_elapsed/60:.1f}min | ETA: {eta_min:.1f}min remaining")

                # Save partial summary after each experiment (resume-safe)
                partial_path = os.path.join(CONFIG['output_dir'], "results", "ablation_summary_partial.json")
                with open(partial_path, "w") as f:
                    json.dump(summary, f, indent=4)

    # --- Save summary ---
    summary_path = os.path.join(CONFIG['output_dir'], "results", "ablation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    # Print summary table
    print("\n" + "=" * 90)
    print("ABLATION STUDY RESULTS")
    print("=" * 90)
    print(f"{'Experiment':<45} {'Domain':<6} {'k':<5} {'Val':>7} {'Test':>7} {'F1-EQ':>7} {'F1-EX':>7}")
    print("-" * 90)
    for row in sorted(summary, key=lambda x: x['test_accuracy'], reverse=True):
        print(f"{row['experiment']:<45} {row['domain']:<6} {str(row['k']):<5} "
              f"{row['val_accuracy']:>7.4f} {row['test_accuracy']:>7.4f} "
              f"{row['test_f1_eq']:>7.4f} {row['test_f1_ex']:>7.4f}")

    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    run_ablation()
