"""
Threshold calibration analysis under varying class prior distributions.

Simulates operational deployment scenarios where the earthquake/explosion
ratio differs from the test set, and shows the accuracy gain from
Bayes-optimal threshold calibration at each prior.

Usage: python -m src.plot_threshold_calibration
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import accuracy_score

from src.dataloader import load_data
from src.classifier import FastMapSVMClassifier

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PLOT_DIR = os.path.join(PROJECT_ROOT, "output", "decision_boundary_plots")


def save_figure(fig, path_without_ext):
    """Save figure as both PNG (300 DPI) and PDF (vector)."""
    fig.savefig(path_without_ext + ".png", dpi=300, bbox_inches='tight')
    fig.savefig(path_without_ext + ".pdf", bbox_inches='tight')


def to_frequency_domain(X):
    X_fft = np.fft.rfft(X, axis=1)
    X_mag = np.abs(X_fft)
    X_mag = np.log10(X_mag + 1e-6)
    X_flat = X_mag.reshape(X.shape[0], -1)
    return normalize(X_flat, axis=1, norm='l2')


def run_calibration_analysis(decision_vals, y, save_dir):
    """Simulate Bayes-optimal threshold calibration at varying class priors.

    Args:
        decision_vals: SVM decision function values, shape (n_samples,).
        y: true labels (0=Earthquake, 1=Explosion).
        save_dir: directory to save outputs.
    """
    eq_vals = decision_vals[y == 0]
    ex_vals = decision_vals[y == 1]

    # Fit per-class Gaussians
    mu_eq, sigma_eq = norm.fit(eq_vals)
    mu_ex, sigma_ex = norm.fit(ex_vals)

    # Scenarios to evaluate
    scenarios = [
        ("Balanced", 0.50),
        ("Test set (57/43)", 0.574),
        ("70% EX", 0.70),
        ("80% EX", 0.80),
        ("90% EX", 0.90),
        ("95% EX", 0.95),
        ("99% EX", 0.99),
    ]

    prior_exs = []
    thresholds = []
    accs_default = []
    accs_calibrated = []
    gains = []

    print("  === Threshold Calibration Under Varying Priors ===\n")
    print(f"  {'Scenario':<20} {'π_EX':>6} {'τ*':>8} {'Acc(t=0)':>9} {'Acc(τ*)':>9} {'Gain':>8}")
    print("  " + "-" * 60)

    for name, prior_ex in scenarios:
        prior_eq = 1.0 - prior_ex

        def posterior_diff(x):
            return (prior_eq * norm.pdf(x, mu_eq, sigma_eq)
                    - prior_ex * norm.pdf(x, mu_ex, sigma_ex))

        try:
            tau = brentq(posterior_diff, -4, 4)
        except ValueError:
            x_grid = np.linspace(-4, 4, 10000)
            diffs = np.abs([posterior_diff(x) for x in x_grid])
            tau = x_grid[np.argmin(diffs)]

        # Weighted accuracy at default threshold (t=0)
        eq_correct_0 = np.mean(eq_vals < 0)
        ex_correct_0 = np.mean(ex_vals > 0)
        acc_0 = prior_eq * eq_correct_0 + prior_ex * ex_correct_0

        # Weighted accuracy at Bayes-optimal threshold
        eq_correct_t = np.mean(eq_vals < tau)
        ex_correct_t = np.mean(ex_vals > tau)
        acc_t = prior_eq * eq_correct_t + prior_ex * ex_correct_t

        gain_pp = (acc_t - acc_0) * 100

        prior_exs.append(prior_ex)
        thresholds.append(tau)
        accs_default.append(acc_0)
        accs_calibrated.append(acc_t)
        gains.append(gain_pp)

        print(f"  {name:<20} {prior_ex:>5.0%} {tau:>+8.3f} {acc_0:>8.2%} {acc_t:>9.2%} {gain_pp:>+7.2f}pp")

    # --- Save results as JSON ---
    results = {
        "gaussian_params": {
            "mu_eq": round(mu_eq, 4), "sigma_eq": round(sigma_eq, 4),
            "mu_ex": round(mu_ex, 4), "sigma_ex": round(sigma_ex, 4),
        },
        "scenarios": [
            {
                "name": name,
                "prior_ex": round(prior_ex, 3),
                "threshold": round(tau, 4),
                "acc_default": round(acc_0, 4),
                "acc_calibrated": round(acc_t, 4),
                "gain_pp": round(g, 2),
            }
            for (name, prior_ex), tau, acc_0, acc_t, g
            in zip(scenarios, thresholds, accs_default, accs_calibrated, gains)
        ],
    }
    json_path = os.path.join(save_dir, "threshold_calibration_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\n  Saved: {json_path}")

    # --- Plot: two panels ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Dense sweep for smooth curves
    prior_sweep = np.linspace(0.5, 0.99, 200)
    tau_sweep = []
    acc0_sweep = []
    acct_sweep = []
    gain_sweep = []

    for pe in prior_sweep:
        peq = 1.0 - pe

        def pd(x, peq=peq, pe=pe):
            return peq * norm.pdf(x, mu_eq, sigma_eq) - pe * norm.pdf(x, mu_ex, sigma_ex)

        try:
            t = brentq(pd, -4, 4)
        except ValueError:
            t = 0
        tau_sweep.append(t)

        eq0 = np.mean(eq_vals < 0)
        ex0 = np.mean(ex_vals > 0)
        a0 = peq * eq0 + pe * ex0

        eqt = np.mean(eq_vals < t)
        ext = np.mean(ex_vals > t)
        at = peq * eqt + pe * ext

        acc0_sweep.append(a0)
        acct_sweep.append(at)
        gain_sweep.append((at - a0) * 100)

    # Left panel: accuracy curves
    ax1.plot(prior_sweep, [a * 100 for a in acc0_sweep], color='#e74c3c',
             linewidth=2, label='Default (t=0)')
    ax1.plot(prior_sweep, [a * 100 for a in acct_sweep], color='#2ecc71',
             linewidth=2, label='Bayes-optimal (t=τ*)')
    ax1.fill_between(prior_sweep,
                      [a * 100 for a in acc0_sweep],
                      [a * 100 for a in acct_sweep],
                      alpha=0.15, color='#2ecc71')
    ax1.axvline(x=0.574, color='gray', linestyle='--', alpha=0.5, label='Test set prior')
    ax1.set_xlabel('Explosion Prior (π_EX)', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Classification Accuracy vs Class Prior', fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.set_xlim(0.5, 0.99)

    # Right panel: gain and threshold
    color_gain = '#2980b9'
    color_tau = '#e67e22'

    ax2.plot(prior_sweep, gain_sweep, color=color_gain, linewidth=2, label='Accuracy gain (pp)')
    ax2.set_xlabel('Explosion Prior (π_EX)', fontsize=12)
    ax2.set_ylabel('Accuracy Gain (pp)', fontsize=12, color=color_gain)
    ax2.tick_params(axis='y', labelcolor=color_gain)
    ax2.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
    ax2.axvline(x=0.574, color='gray', linestyle='--', alpha=0.5)
    ax2.grid(alpha=0.3)
    ax2.set_xlim(0.5, 0.99)

    ax2b = ax2.twinx()
    ax2b.plot(prior_sweep, tau_sweep, color=color_tau, linewidth=2, linestyle='--',
              label='Threshold τ*')
    ax2b.set_ylabel('Bayes-optimal Threshold (τ*)', fontsize=12, color=color_tau)
    ax2b.tick_params(axis='y', labelcolor=color_tau)

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc='upper left')
    ax2.set_title('Calibration Gain and Threshold Shift', fontweight='bold')

    plt.suptitle('Bayes-Optimal Threshold Calibration Under Deployment Imbalance',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    save_path = os.path.join(save_dir, "threshold_calibration")
    save_figure(fig, save_path)
    plt.close()
    print(f"  Saved: {save_path}.png/pdf")


if __name__ == "__main__":
    os.makedirs(PLOT_DIR, exist_ok=True)

    # Load data
    print("Loading data...")
    (X_train_raw, y_train), (X_val_raw, y_val), (X_test_raw, y_test) = load_data(DATA_DIR)

    # Combine train+val
    X_trainval_raw = np.vstack([X_train_raw, X_val_raw])
    y_trainval = np.concatenate([y_train, y_val])

    # FFT features
    print("Computing FFT features...")
    X_trainval_fft = to_frequency_domain(X_trainval_raw)
    X_test_fft = to_frequency_domain(X_test_raw)

    # Train k=120 FastMap model (same seed as GMM analysis)
    print("Training k=120 FastMap model...")
    np.random.seed(42)
    model = FastMapSVMClassifier(k=120, dist_func='euclidean')
    X_trainval_emb = model.fastmap.fit_transform(X_trainval_fft)
    X_test_emb = model.fastmap.transform(X_test_fft)

    scaler = StandardScaler()
    X_trainval_scaled = scaler.fit_transform(X_trainval_emb)
    X_test_scaled = scaler.transform(X_test_emb)

    clf = SVC(C=1000, gamma=0.01, kernel='rbf', class_weight='balanced')
    clf.fit(X_trainval_scaled, y_trainval)

    decision_vals_test = clf.decision_function(X_test_scaled)
    test_acc = accuracy_score(y_test, clf.predict(X_test_scaled))
    print(f"Single model test accuracy: {test_acc:.4f}\n")

    # Run calibration analysis
    run_calibration_analysis(decision_vals_test, y_test, PLOT_DIR)
