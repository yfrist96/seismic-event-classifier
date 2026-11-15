"""
Quickstart: train and evaluate one FFT + Euclidean FastMap-SVM model, then
recalibrate its decision threshold for a deployment class prior.

Steps
  1. Load the train / validation / test HDF5 splits.
  2. Compute the log-magnitude FFT features used throughout the pipeline
     (src.ablation.to_frequency_domain).
  3. Fit FastMap (k=120, Euclidean) on the training split and embed all splits.
  4. Standardize the embedding and train SVC(C=1000, gamma=0.01, kernel='rbf',
     class_weight='balanced') on the training split.
  5. Report validation and test accuracy at the default threshold (0).
  6. Fit BayesThresholdCalibrator on the validation decision values and, when
     --prior-ex is given, recompute the threshold for that explosion prior.

The protocol (train on the train split only, calibrate on validation, evaluate
on test, NumPy seed 42 before FastMap) is the one used by
src/plot_gmm_analysis.py and src/plot_threshold_calibration.py.

Usage (from the repository root):
    python -m examples.quickstart --data-dir data --out-dir /path/to/new_dir --prior-ex 0.95

Everything the script writes goes into --out-dir (default: a new temporary
directory). It takes about 40 s on a laptop CPU, most of it in FastMap.
"""

import argparse
import contextlib
import csv
import io
import json
import os
import sys
import tempfile
import time

# Do not write .pyc files into the repository when importing src/.
sys.dont_write_bytecode = True

# Allow "python examples/quickstart.py" as well as "python -m examples.quickstart".
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.ablation import to_frequency_domain      # FFT feature function of the pipeline
from src.classifier import BayesThresholdCalibrator
from src.dataloader import load_split
from src.fastmap import FastMap

K = 120                     # FastMap dimensions of the selected model
DISTANCE = "euclidean"      # distance used by FastMap on the FFT features
SVM_PARAMS = dict(C=1000, gamma=0.01, kernel="rbf", class_weight="balanced")
CLASS_NAMES = {0: "Earthquake", 1: "Explosion"}


def parse_args():
    p = argparse.ArgumentParser(description="FastMap-SVM quickstart (single model + threshold calibration).")
    p.add_argument("--data-dir", default="data",
                   help="Directory with dataset_train.h5, dataset_val.h5, dataset_test.h5 (default: data).")
    p.add_argument("--out-dir", default=None,
                   help="Directory for all outputs (created if missing). "
                        "Default: a new directory under the system temp dir.")
    p.add_argument("--prior-ex", type=float, default=None,
                   help="Deployment explosion prior P(EX) in (0, 1), e.g. 0.95. "
                        "Default: none (use the validation class mix only).")
    p.add_argument("--seed", type=int, default=42,
                   help="NumPy seed set before FastMap pivot selection (default: 42).")
    p.add_argument("--verbose", action="store_true",
                   help="Show FastMap's per-dimension progress messages.")
    args = p.parse_args()
    if args.prior_ex is not None and not 0.0 < args.prior_ex < 1.0:
        p.error("--prior-ex must be strictly between 0 and 1")
    return args


def quiet(enabled):
    """Context manager that hides stdout (FastMap prints one line per dimension)."""
    return contextlib.nullcontext() if enabled else contextlib.redirect_stdout(io.StringIO())


def recalls(decision_vals, y, threshold):
    """Earthquake and explosion recall when predicting EX for decision > threshold."""
    pred = (decision_vals > threshold).astype(int)
    return float(np.mean(pred[y == 0] == 0)), float(np.mean(pred[y == 1] == 1))


def main():
    args = parse_args()
    t_start = time.time()

    out_dir = args.out_dir or tempfile.mkdtemp(prefix="seismic_quickstart_")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output directory: {out_dir}\n")

    # 1. Load data. Each X is (N, 600, 3) float32: 3 s at 200 Hz, channels Z, N, E.
    #    y is 0 = Earthquake, 1 = Explosion.
    print("[1] Loading data")
    X_train, y_train, _ = load_split(os.path.join(args.data_dir, "dataset_train.h5"))
    X_val, y_val, _ = load_split(os.path.join(args.data_dir, "dataset_val.h5"))
    X_test, y_test, eid_test = load_split(os.path.join(args.data_dir, "dataset_test.h5"))
    for name, y in (("train", y_train), ("val", y_val), ("test", y_test)):
        print(f"    {name:<5}: {len(y):5d} windows  (EQ {np.sum(y == 0)}, EX {np.sum(y == 1)})")

    # 2. FFT features: rfft along time -> |.| -> log10(. + 1e-6) -> flatten -> L2-normalize.
    #    600 samples give 301 frequency bins x 3 channels = 903 features per window.
    print("\n[2] Computing FFT features")
    F_train, F_val, F_test = (to_frequency_domain(X) for X in (X_train, X_val, X_test))
    print(f"    feature matrix (train): {F_train.shape}")

    # 3. FastMap. Pivot selection draws from NumPy's global RNG, so seed it first.
    print(f"\n[3] Fitting FastMap (k={K}, {DISTANCE}) on the training split")
    np.random.seed(args.seed)
    fastmap = FastMap(n_components=K, dist_func=DISTANCE)
    t0 = time.time()
    with quiet(args.verbose):
        E_train = fastmap.fit_transform(F_train)
    t_fit = time.time() - t0
    t0 = time.time()
    with quiet(args.verbose):
        E_val = fastmap.transform(F_val)
        E_test = fastmap.transform(F_test)
    t_transform = time.time() - t0
    print(f"    fit_transform(train): {t_fit:.1f} s; transform(val + test): {t_transform:.1f} s")

    # 4. Standardize (statistics from the training split only) and train the SVM.
    print(f"\n[4] Training SVC{SVM_PARAMS}")
    scaler = StandardScaler().fit(E_train)
    S_train, S_val, S_test = (scaler.transform(E) for E in (E_train, E_val, E_test))
    t0 = time.time()
    svm = SVC(**SVM_PARAMS).fit(S_train, y_train)
    print(f"    trained in {time.time() - t0:.1f} s; {svm.n_support_.sum()} support vectors")

    # 5. Evaluate at the default threshold (decision_function > 0 -> Explosion).
    d_val = svm.decision_function(S_val)
    d_test = svm.decision_function(S_test)
    val_acc = accuracy_score(y_val, svm.predict(S_val))
    test_acc = accuracy_score(y_test, svm.predict(S_test))
    print("\n[5] Default threshold (t = 0)")
    print(f"    validation accuracy: {val_acc:.4f}")
    print(f"    test accuracy:       {test_acc:.4f}")
    cm = confusion_matrix(y_test, (d_test > 0).astype(int))
    print(f"    test confusion matrix [rows true EQ/EX, cols predicted EQ/EX]: {cm.tolist()}")

    # 6a. Bayes-optimal threshold at the validation class mix (empirical priors).
    print("\n[6] Bayes threshold calibration on validation decision values")
    cal_val = BayesThresholdCalibrator().fit(d_val, y_val)
    print(f"    Gaussian fits: EQ N({cal_val.mu_eq:+.3f}, {cal_val.sigma_eq:.3f}), "
          f"EX N({cal_val.mu_ex:+.3f}, {cal_val.sigma_ex:.3f})")
    print(f"    validation priors P(EX) = {cal_val.priors[1]:.3f} -> threshold {cal_val.threshold:+.4f}")
    print(f"    test accuracy at this threshold: {accuracy_score(y_test, cal_val.predict(d_test)):.4f}")

    # 6b. Optional: deployment prior supplied by the user. Same Gaussians, new priors.
    summary_prior = None
    pred_prior = None
    if args.prior_ex is not None:
        p_ex = args.prior_ex
        cal_dep = BayesThresholdCalibrator().fit(d_val, y_val, priors={0: 1.0 - p_ex, 1: p_ex})
        pred_prior = cal_dep.predict(d_test)
        eq0, ex0 = recalls(d_test, y_test, 0.0)
        eqt, ext = recalls(d_test, y_test, cal_dep.threshold)
        # Prior-weighted accuracy: the accuracy expected if the deployment class mix were P(EX).
        acc0 = (1 - p_ex) * eq0 + p_ex * ex0
        acct = (1 - p_ex) * eqt + p_ex * ext
        print(f"\n    Deployment prior P(EX) = {p_ex:.3f} -> threshold {cal_dep.threshold:+.4f}")
        print(f"      {'':<22}{'EQ recall':>10}{'EX recall':>11}{'prior-weighted acc':>20}")
        print(f"      {'default (t = 0)':<22}{eq0:>10.4f}{ex0:>11.4f}{acc0:>20.4f}")
        print(f"      {'calibrated':<22}{eqt:>10.4f}{ext:>11.4f}{acct:>20.4f}")
        print(f"      gain: {100 * (acct - acc0):+.2f} percentage points; "
              f"test windows predicted EX: {int(np.sum(d_test > 0))} at t = 0, "
              f"{int(pred_prior.sum())} calibrated")
        summary_prior = dict(prior_ex=p_ex, threshold=float(cal_dep.threshold),
                             eq_recall_default=eq0, ex_recall_default=ex0,
                             eq_recall_calibrated=eqt, ex_recall_calibrated=ext,
                             weighted_acc_default=acc0, weighted_acc_calibrated=acct)

    # 7. Save everything into out_dir.
    model_path = os.path.join(out_dir, "model.joblib")
    joblib.dump({"fastmap": fastmap, "scaler": scaler, "svm": svm, "params": SVM_PARAMS,
                 "val_accuracy": val_acc, "test_accuracy": test_acc,
                 "calibrator": cal_val}, model_path)
    np.savez(os.path.join(out_dir, "decision_values.npz"),
             val=d_val, y_val=y_val, test=d_test, y_test=y_test)
    with open(os.path.join(out_dir, "test_predictions.csv"), "w", newline="") as f:
        w = csv.writer(f)
        header = ["window_index", "event_id", "true_label", "decision_value",
                  "pred_default", "pred_val_prior"]
        if pred_prior is not None:
            header.append("pred_deploy_prior")
        w.writerow(header)
        pred_valprior = cal_val.predict(d_test)
        for i in range(len(y_test)):
            row = [i, int(eid_test[i]), int(y_test[i]), f"{d_test[i]:.4f}",
                   int(d_test[i] > 0), int(pred_valprior[i])]
            if pred_prior is not None:
                row.append(int(pred_prior[i]))
            w.writerow(row)
    total = time.time() - t_start
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"k": K, "distance": DISTANCE, "svm_params": SVM_PARAMS, "seed": args.seed,
                   "val_accuracy": val_acc, "test_accuracy": test_acc,
                   "val_prior_threshold": float(cal_val.threshold),
                   "deployment_prior": summary_prior,
                   "timing_s": {"fastmap_fit": t_fit, "fastmap_transform": t_transform,
                                "total": total}}, f, indent=2)

    print(f"\n[7] Wrote model.joblib, decision_values.npz, test_predictions.csv, summary.json")
    print(f"    to {out_dir}")
    print(f"Total wall-clock time: {total:.1f} s")


if __name__ == "__main__":
    main()
