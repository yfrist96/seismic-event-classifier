import os
import json
import numpy as np
import joblib
from datetime import datetime
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from torch.utils.tensorboard import SummaryWriter

from src.dataloader import load_data
from src.classifier import FastMapSVMClassifier, BaselineClassifier
from src.config import CONFIG
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

    return X_norm


def run_experiments():
    # 1. Setup Directories
    os.makedirs(f"{CONFIG['output_dir']}/models", exist_ok=True)
    os.makedirs(f"{CONFIG['output_dir']}/results", exist_ok=True)
    os.makedirs(f"{CONFIG['output_dir']}/embeddings", exist_ok=True)

    # Setup TensorBoard
    log_dir = os.path.join(CONFIG['output_dir'], "tb_logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs: {log_dir}")

    # 2. Load Data
    print(f"Loading Data from: {CONFIG['data_dir']}...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data(CONFIG['data_dir'])

    unique, counts = np.unique(y_train, return_counts=True)
    print(f"\nClass Distribution in Training Set:")
    for label, count in zip(unique, counts):
        print(f"  Class {label}: {count} samples ({count/len(y_train)*100:.1f}%)")

    # 3. Run Baseline (Random Forest)
    print("\n=== Running Baseline Model ===")
    try:
        baseline = BaselineClassifier()
        baseline.fit(X_train, y_train)

        # Evaluate on validation
        y_pred_val = baseline.predict(X_val)
        val_acc = accuracy_score(y_val, y_pred_val)
        print(f"Baseline Val Accuracy: {val_acc:.4f}")

        # Final evaluation on test
        y_pred_test = baseline.predict(X_test)
        test_report = classification_report(y_test, y_pred_test, output_dict=True)
        print(f"Baseline Test Accuracy: {test_report['accuracy']:.4f}")

        # Log baseline to TensorBoard
        writer.add_scalar("baseline/val_accuracy", val_acc, 0)
        writer.add_scalar("baseline/test_accuracy", test_report['accuracy'], 0)

        with open(f"{CONFIG['output_dir']}/results/baseline_report.json", "w") as f:
            json.dump(test_report, f, indent=4)
    except Exception as e:
        print(f"Baseline failed: {e}")

    # --- Transform to Frequency Domain ---
    print("\nConverting data to Frequency Domain (FFT)...")
    X_train_fft = to_frequency_domain(X_train)
    X_val_fft = to_frequency_domain(X_val)
    X_test_fft = to_frequency_domain(X_test)
    print(f"New Feature Shape: {X_train_fft.shape}")

    # 4. Run FastMap + SVM experiments
    exp_idx = 0
    for dist in CONFIG['distances']:
        for k in CONFIG['dimensions']:
            exp_name = f"fastmap_FFT_{dist}_k{k}_tuned"
            print(f"\n=== Running {exp_name} ===")

            # A. Generate Embeddings
            temp_model = FastMapSVMClassifier(k=k, dist_func=dist)

            print(f"  [FastMap] Generating embeddings for k={k}...")
            start_time = datetime.now()
            X_train_emb = temp_model.fastmap.fit_transform(X_train_fft)
            X_val_emb = temp_model.fastmap.transform(X_val_fft)
            X_test_emb = temp_model.fastmap.transform(X_test_fft)

            duration = (datetime.now() - start_time).total_seconds()
            print(f"  [FastMap] Done in {duration:.1f}s")

            # Save Embeddings
            np.save(f"{CONFIG['output_dir']}/embeddings/{exp_name}_train.npy", X_train_emb)

            # B. Standardize
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_emb)
            X_val_scaled = scaler.transform(X_val_emb)
            X_test_scaled = scaler.transform(X_test_emb)

            # C. Grid Search — tune on VALIDATION set
            print("  [GridSearch] Tuning SVM on validation set...")
            best_score = -1
            best_params = None
            best_model = None

            # Flatten param grid combinations manually to evaluate on val
            from sklearn.model_selection import ParameterGrid
            param_combinations = list(ParameterGrid(CONFIG['svm_param_grid']))
            print(f"  [GridSearch] Evaluating {len(param_combinations)} param combinations...")

            for i, params in enumerate(param_combinations):
                clf = SVC(**params)
                clf.fit(X_train_scaled, y_train)
                val_score = accuracy_score(y_val, clf.predict(X_val_scaled))

                # Log each param combo to TensorBoard
                writer.add_scalar(f"{exp_name}/grid_search_val_acc", val_score, i)

                if val_score > best_score:
                    best_score = val_score
                    best_params = params
                    best_model = clf

            print(f"  [GridSearch] Best Params: {best_params}")
            print(f"  [GridSearch] Best Val Accuracy: {best_score:.4f}")

            # D. Retrain on train+val with best params, evaluate on test
            print("  [Retrain] Training on train+val with best params...")
            X_trainval_emb = np.vstack([X_train_emb, X_val_emb])
            y_trainval = np.concatenate([y_train, y_val])

            scaler_final = StandardScaler()
            X_trainval_scaled = scaler_final.fit_transform(X_trainval_emb)
            X_test_scaled_final = scaler_final.transform(X_test_emb)

            final_model = SVC(**best_params)
            final_model.fit(X_trainval_scaled, y_trainval)

            y_pred_test = final_model.predict(X_test_scaled_final)
            report_single = classification_report(y_test, y_pred_test, output_dict=True)
            report_single['best_params'] = {k: str(v) for k, v in best_params.items()}

            test_acc = report_single['accuracy']
            print(f"  [RESULT] Test Accuracy: {test_acc:.4f}")

            # Log to TensorBoard
            writer.add_scalar(f"{exp_name}/val_accuracy", best_score, exp_idx)
            writer.add_scalar(f"{exp_name}/test_accuracy", test_acc, exp_idx)
            writer.add_scalar(f"{exp_name}/test_f1_eq", report_single['0']['f1-score'], exp_idx)
            writer.add_scalar(f"{exp_name}/test_f1_ex", report_single['1']['f1-score'], exp_idx)
            writer.add_hparams(
                {f"{exp_name}/k": k, f"{exp_name}/dist": dist,
                 **{f"{exp_name}/{pk}": str(pv) for pk, pv in best_params.items()}},
                {f"{exp_name}/hparam/val_acc": best_score,
                 f"{exp_name}/hparam/test_acc": test_acc}
            )

            # Save Results
            with open(f"{CONFIG['output_dir']}/results/{exp_name}_report.json", "w") as f:
                json.dump(report_single, f, indent=4)

            # Save Model (FastMap + Scaler + SVM)
            model_path = f"{CONFIG['output_dir']}/models/{exp_name}.joblib"
            joblib.dump({
                'fastmap': temp_model.fastmap,
                'scaler': scaler_final,
                'svm': final_model,
                'params': best_params,
                'val_accuracy': best_score,
                'test_accuracy': test_acc
            }, model_path)
            print(f"  [Saved] Model -> {model_path}")

            # --- Ensemble ---
            print(f"  [Ensemble] Running Ensemble for {exp_name}...")

            y_pred_ensemble = train_and_evaluate_ensemble(
                X_train_fft, y_train, X_val_fft, y_val, X_test_fft, y_test,
                k=k, dist=dist,
                best_params=best_params,
                n_models=5
            )

            # Evaluate Ensemble
            report_ensemble = classification_report(y_test, y_pred_ensemble, output_dict=True)
            ensemble_acc = report_ensemble['accuracy']
            print(f"  [RESULT] Single: {test_acc:.4f} | Ensemble: {ensemble_acc:.4f}")

            # Log ensemble to TensorBoard
            writer.add_scalar(f"{exp_name}_ensemble/test_accuracy", ensemble_acc, exp_idx)
            writer.add_scalar(f"{exp_name}_ensemble/test_f1_eq", report_ensemble['0']['f1-score'], exp_idx)
            writer.add_scalar(f"{exp_name}_ensemble/test_f1_ex", report_ensemble['1']['f1-score'], exp_idx)

            # Save Ensemble Results
            with open(f"{CONFIG['output_dir']}/results/{exp_name}_ensemble_report.json", "w") as f:
                json.dump(report_ensemble, f, indent=4)

            exp_idx += 1

    writer.close()
    print(f"\nDone! TensorBoard logs saved to: {log_dir}")
    print(f"Run: tensorboard --logdir {CONFIG['output_dir']}/tb_logs")


def train_and_evaluate_ensemble(X_train, y_train, X_val, y_val, X_test, y_test,
                                 k, dist, best_params, n_models=5):
    """
    Trains multiple FastMap models using the BEST parameters found by GridSearch,
    then aggregates their predictions via Majority Vote.
    Now trains each voter on train+val combined.
    """
    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])

    print(f"  [Ensemble] Training {n_models} voters on train+val using params: {best_params}...")

    all_preds = np.zeros((n_models, len(y_test)))

    for i in range(n_models):
        # 1. New Random FastMap Projection
        model_wrapper = FastMapSVMClassifier(k=k, dist_func=dist)

        # Embed
        X_trainval_emb = model_wrapper.fastmap.fit_transform(X_trainval)
        X_test_emb = model_wrapper.fastmap.transform(X_test)

        # 2. Standardize
        scaler = StandardScaler()
        X_trainval_scaled = scaler.fit_transform(X_trainval_emb)
        X_test_scaled = scaler.transform(X_test_emb)

        # 3. Train SVM with best parameters
        clf = SVC(**best_params)
        clf.fit(X_trainval_scaled, y_trainval)

        # 4. Predict
        all_preds[i] = clf.predict(X_test_scaled)

    # 5. Majority Vote
    final_preds, _ = mode(all_preds, axis=0, keepdims=False)
    final_preds = final_preds.ravel().astype(int)

    return final_preds


if __name__ == "__main__":
    run_experiments()
