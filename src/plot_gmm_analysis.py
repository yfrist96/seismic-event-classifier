"""
Deep statistical analysis of SVM decision function distribution.

Fits Gaussian mixture models, runs normality tests, computes d-prime,
finds the Bayes-optimal threshold, and quantifies distributional overlap.

Usage: python -m src.plot_gmm_analysis
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import accuracy_score
from sklearn.mixture import GaussianMixture
from scipy.stats import shapiro, kstest, norm, probplot
from scipy.optimize import brentq

from src.dataloader import load_data
from src.classifier import FastMapSVMClassifier

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PLOT_DIR = os.path.join(PROJECT_ROOT, "output", "decision_boundary_plots")


def to_frequency_domain(X):
    X_fft = np.fft.rfft(X, axis=1)
    X_mag = np.abs(X_fft)
    X_mag = np.log10(X_mag + 1e-6)
    X_flat = X_mag.reshape(X.shape[0], -1)
    return normalize(X_flat, axis=1, norm='l2')


def plot_decision_function_analysis(decision_vals, y, save_dir):
    """Deep statistical analysis of SVM decision function distribution.

    Fits Gaussian models per class, a 2-component GMM on unlabeled data,
    runs normality tests, computes d-prime, finds the Bayes-optimal threshold,
    and quantifies the overlap between class distributions.

    Args:
        decision_vals: SVM decision function values, shape (n_samples,).
        y: true labels (0=Earthquake, 1=Explosion), shape (n_samples,).
        save_dir: directory to save the output figure.
    """
    print("  === Decision Function Statistical Analysis ===\n")

    eq_vals = decision_vals[y == 0]
    ex_vals = decision_vals[y == 1]

    # ── Per-class Gaussian fits ──
    mu_eq, sigma_eq = norm.fit(eq_vals)
    mu_ex, sigma_ex = norm.fit(ex_vals)
    n_eq, n_ex = len(eq_vals), len(ex_vals)
    n_total = n_eq + n_ex

    print(f"  Per-class Gaussian fits:")
    print(f"    Earthquake:  mu = {mu_eq:+.4f},  sigma = {sigma_eq:.4f},  n = {n_eq}")
    print(f"    Explosion:   mu = {mu_ex:+.4f},  sigma = {sigma_ex:.4f},  n = {n_ex}")

    # ── d-prime (sensitivity index) ──
    d_prime = abs(mu_ex - mu_eq) / np.sqrt(0.5 * (sigma_eq**2 + sigma_ex**2))
    print(f"\n  Separation metrics:")
    print(f"    d-prime (sensitivity index): {d_prime:.4f}")

    # ── Normality tests ──
    # Shapiro-Wilk has a sample limit of 5000
    def safe_shapiro(vals, name):
        if len(vals) > 5000:
            rng = np.random.RandomState(42)
            subset = rng.choice(vals, 5000, replace=False)
            w, p = shapiro(subset)
            return w, p, True
        w, p = shapiro(vals)
        return w, p, False

    w_eq, p_eq, sub_eq = safe_shapiro(eq_vals, "Earthquake")
    w_ex, p_ex, sub_ex = safe_shapiro(ex_vals, "Explosion")

    ks_eq, ksp_eq = kstest(eq_vals, 'norm', args=(mu_eq, sigma_eq))
    ks_ex, ksp_ex = kstest(ex_vals, 'norm', args=(mu_ex, sigma_ex))

    print(f"\n  Normality tests:")
    sub_note_eq = " (subsampled to 5000)" if sub_eq else ""
    sub_note_ex = " (subsampled to 5000)" if sub_ex else ""
    print(f"    Earthquake:  Shapiro W={w_eq:.4f} (p={p_eq:.4e}){sub_note_eq}  |  KS D={ks_eq:.4f} (p={ksp_eq:.4e})")
    print(f"    Explosion:   Shapiro W={w_ex:.4f} (p={p_ex:.4e}){sub_note_ex}  |  KS D={ks_ex:.4f} (p={ksp_ex:.4e})")

    # ── GMM model comparison (unsupervised) ──
    X_gmm = decision_vals.reshape(-1, 1)
    gmm1 = GaussianMixture(n_components=1, random_state=42, n_init=5).fit(X_gmm)
    gmm2 = GaussianMixture(n_components=2, random_state=42, n_init=5).fit(X_gmm)

    aic1, bic1 = gmm1.aic(X_gmm), gmm1.bic(X_gmm)
    aic2, bic2 = gmm2.aic(X_gmm), gmm2.bic(X_gmm)

    # Sort GMM components by mean
    order = np.argsort(gmm2.means_.ravel())
    gmm_means = gmm2.means_.ravel()[order]
    gmm_stds = np.sqrt(gmm2.covariances_.ravel()[order])
    gmm_weights = gmm2.weights_[order]

    print(f"\n  GMM model comparison (unsupervised on all decision values):")
    print(f"    1-component:  AIC = {aic1:.1f},  BIC = {bic1:.1f}")
    print(f"    2-component:  AIC = {aic2:.1f},  BIC = {bic2:.1f}")
    preferred = "2-component" if bic2 < bic1 else "1-component"
    print(f"    Preferred: {preferred} (delta_BIC = {abs(bic1 - bic2):.1f})")

    print(f"\n  GMM 2-component parameters (unsupervised):")
    print(f"    Component 1: mu={gmm_means[0]:+.4f}, sigma={gmm_stds[0]:.4f}, weight={gmm_weights[0]:.4f}")
    print(f"    Component 2: mu={gmm_means[1]:+.4f}, sigma={gmm_stds[1]:.4f}, weight={gmm_weights[1]:.4f}")

    # ── Bayes-optimal threshold ──
    prior_eq = n_eq / n_total
    prior_ex = n_ex / n_total

    def posterior_diff(x):
        """prior_eq * pdf_eq(x) - prior_ex * pdf_ex(x)"""
        return prior_eq * norm.pdf(x, mu_eq, sigma_eq) - prior_ex * norm.pdf(x, mu_ex, sigma_ex)

    try:
        optimal_threshold = brentq(posterior_diff, mu_eq, mu_ex)
    except ValueError:
        # Fallback: grid search
        x_grid = np.linspace(decision_vals.min(), decision_vals.max(), 10000)
        diffs = np.abs([posterior_diff(x) for x in x_grid])
        optimal_threshold = x_grid[np.argmin(diffs)]

    # ── Threshold sweep for empirical best ──
    thresholds = np.linspace(decision_vals.min(), decision_vals.max(), 2000)
    accuracies = np.array([
        accuracy_score(y, (decision_vals > t).astype(int)) for t in thresholds
    ])
    best_idx = np.argmax(accuracies)
    empirical_best_threshold = thresholds[best_idx]
    empirical_best_acc = accuracies[best_idx]

    acc_svm = accuracy_score(y, (decision_vals > 0).astype(int))
    acc_bayes = accuracy_score(y, (decision_vals > optimal_threshold).astype(int))

    print(f"\n  Threshold analysis:")
    print(f"    SVM boundary (t=0.000):             accuracy = {acc_svm:.2%}")
    print(f"    Bayes-optimal (t={optimal_threshold:+.3f}):       accuracy = {acc_bayes:.2%}")
    print(f"    Empirical best (t={empirical_best_threshold:+.3f}):      accuracy = {empirical_best_acc:.2%}")
    improvement = (empirical_best_acc - acc_svm) * 100
    print(f"    Improvement from optimal shift: {improvement:+.2f}pp")

    # ── Overlap coefficient ──
    x_grid = np.linspace(decision_vals.min() - 1, decision_vals.max() + 1, 5000)
    pdf_eq = prior_eq * norm.pdf(x_grid, mu_eq, sigma_eq)
    pdf_ex = prior_ex * norm.pdf(x_grid, mu_ex, sigma_ex)
    overlap = np.trapezoid(np.minimum(pdf_eq, pdf_ex), x_grid)

    misclass_eq = np.mean(eq_vals > 0)
    misclass_ex = np.mean(ex_vals < 0)

    print(f"\n  Overlap analysis:")
    print(f"    Overlap coefficient: {overlap:.4f} ({overlap:.2%} of total density)")
    print(f"    EQ beyond SVM boundary: {misclass_eq:.1%}")
    print(f"    EX beyond SVM boundary: {misclass_ex:.1%}")

    # ═══════════════════════════════════════════════════════════
    # Figure: 3x2 grid
    # ═══════════════════════════════════════════════════════════
    print(f"\n  Generating analysis figure...")

    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle('Statistical Analysis of SVM Decision Function',
                 fontsize=16, fontweight='bold', y=0.98)

    # ── Panel (0,0): Histogram + fitted PDFs ──
    ax = axes[0, 0]
    x_plot = np.linspace(decision_vals.min() - 0.5, decision_vals.max() + 0.5, 500)
    bins = np.linspace(decision_vals.min(), decision_vals.max(), 60)

    ax.hist(eq_vals, bins=bins, alpha=0.4, color='#3498db', label='Earthquake (data)', density=True)
    ax.hist(ex_vals, bins=bins, alpha=0.4, color='#e74c3c', label='Explosion (data)', density=True)

    # Per-class fitted Gaussians
    ax.plot(x_plot, norm.pdf(x_plot, mu_eq, sigma_eq) * prior_eq,
            color='#2980b9', linewidth=2.5, linestyle='-',
            label=f'EQ fit: N({mu_eq:.2f}, {sigma_eq:.2f})')
    ax.plot(x_plot, norm.pdf(x_plot, mu_ex, sigma_ex) * prior_ex,
            color='#c0392b', linewidth=2.5, linestyle='-',
            label=f'EX fit: N({mu_ex:.2f}, {sigma_ex:.2f})')

    # GMM fit (unsupervised)
    gmm_pdf = np.zeros_like(x_plot)
    for k in range(2):
        gmm_pdf += gmm_weights[k] * norm.pdf(x_plot, gmm_means[k], gmm_stds[k])
    ax.plot(x_plot, gmm_pdf, color='green', linewidth=2, linestyle='--',
            label='GMM (unsupervised)', alpha=0.8)

    ax.axvline(x=0, color='black', linewidth=1.5, linestyle='--', alpha=0.7, label='SVM boundary (t=0)')
    ax.axvline(x=optimal_threshold, color='purple', linewidth=1.5, linestyle=':',
               label=f'Bayes-optimal (t={optimal_threshold:.3f})')

    ax.set_xlabel('Decision Function Value')
    ax.set_ylabel('Density')
    ax.set_title('Histogram with Fitted Gaussian PDFs', fontweight='bold')
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(alpha=0.2)

    # ── Panel (0,1): Q-Q plots ──
    ax = axes[0, 1]
    for vals, name, color in [(eq_vals, 'Earthquake', '#3498db'), (ex_vals, 'Explosion', '#e74c3c')]:
        osm, osr = probplot(vals, dist="norm", fit=False)
        # osm = theoretical quantiles, osr = ordered sample values
        ax.scatter(osm, osr, color=color, s=8, alpha=0.5, label=name, edgecolors='none')

    # Reference line
    all_quantiles = np.concatenate([probplot(eq_vals, dist="norm", fit=False)[0],
                                     probplot(ex_vals, dist="norm", fit=False)[0]])
    q_min, q_max = all_quantiles.min(), all_quantiles.max()

    # Fit line for each class
    for vals, color in [(eq_vals, '#2980b9'), (ex_vals, '#c0392b')]:
        osm, osr = probplot(vals, dist="norm", fit=False)
        slope, intercept = np.polyfit(osm, osr, 1)
        ax.plot([q_min, q_max], [slope * q_min + intercept, slope * q_max + intercept],
                color=color, linewidth=1.5, linestyle='--', alpha=0.7)

    ax.set_xlabel('Theoretical Quantiles')
    ax.set_ylabel('Sample Quantiles')
    ax.set_title('Q-Q Plots (Normal Distribution)', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # ── Panel (1,0): Empirical vs fitted CDFs ──
    ax = axes[1, 0]
    for vals, mu, sigma, name, color in [
        (eq_vals, mu_eq, sigma_eq, 'Earthquake', '#3498db'),
        (ex_vals, mu_ex, sigma_ex, 'Explosion', '#e74c3c')
    ]:
        sorted_vals = np.sort(vals)
        ecdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        ax.step(sorted_vals, ecdf, color=color, linewidth=1.5, alpha=0.7,
                label=f'{name} (empirical)')
        ax.plot(x_plot, norm.cdf(x_plot, mu, sigma), color=color,
                linewidth=1.5, linestyle='--', alpha=0.7,
                label=f'{name} (fitted)')

    ax.axvline(x=0, color='black', linewidth=1.5, linestyle='--', alpha=0.5)
    ax.axvline(x=optimal_threshold, color='purple', linewidth=1.5, linestyle=':', alpha=0.7)

    # Shade overlap region
    overlap_left = max(eq_vals.min(), ex_vals.min())
    overlap_right = min(eq_vals.max(), ex_vals.max())
    ax.axvspan(overlap_left, overlap_right, alpha=0.08, color='gray', label='Overlap region')

    ax.set_xlabel('Decision Function Value')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Empirical vs Fitted CDFs', fontweight='bold')
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(alpha=0.3)

    # ── Panel (1,1): Accuracy vs threshold sweep ──
    ax = axes[1, 1]
    ax.plot(thresholds, accuracies, color='#2c3e50', linewidth=1.5)

    ax.axvline(x=0, color='black', linewidth=1.5, linestyle='--',
               label=f'SVM (t=0, acc={acc_svm:.2%})')
    ax.axvline(x=optimal_threshold, color='purple', linewidth=1.5, linestyle=':',
               label=f'Bayes (t={optimal_threshold:.3f}, acc={acc_bayes:.2%})')
    ax.axvline(x=empirical_best_threshold, color='green', linewidth=1.5, linestyle='-.',
               label=f'Best (t={empirical_best_threshold:.3f}, acc={empirical_best_acc:.2%})')

    ax.scatter([0], [acc_svm], color='black', s=60, zorder=5)
    ax.scatter([optimal_threshold], [acc_bayes], color='purple', s=60, zorder=5)
    ax.scatter([empirical_best_threshold], [empirical_best_acc], color='green', s=60, zorder=5)

    ax.set_xlabel('Decision Threshold')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy vs Decision Threshold', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ── Panel (2,0): Overlap detail (zoomed) ──
    ax = axes[2, 0]
    zoom_left = optimal_threshold - 2 * max(sigma_eq, sigma_ex)
    zoom_right = optimal_threshold + 2 * max(sigma_eq, sigma_ex)
    x_zoom = np.linspace(zoom_left, zoom_right, 500)

    pdf_eq_zoom = prior_eq * norm.pdf(x_zoom, mu_eq, sigma_eq)
    pdf_ex_zoom = prior_ex * norm.pdf(x_zoom, mu_ex, sigma_ex)

    ax.plot(x_zoom, pdf_eq_zoom, color='#3498db', linewidth=2, label='Earthquake PDF')
    ax.plot(x_zoom, pdf_ex_zoom, color='#e74c3c', linewidth=2, label='Explosion PDF')

    # Shade classification error regions
    # EQ error: area under EQ pdf to the right of threshold
    mask_eq_error = x_zoom > optimal_threshold
    ax.fill_between(x_zoom[mask_eq_error], 0, pdf_eq_zoom[mask_eq_error],
                     color='#3498db', alpha=0.3, label='EQ misclassified')
    # EX error: area under EX pdf to the left of threshold
    mask_ex_error = x_zoom < optimal_threshold
    ax.fill_between(x_zoom[mask_ex_error], 0, pdf_ex_zoom[mask_ex_error],
                     color='#e74c3c', alpha=0.3, label='EX misclassified')

    ax.axvline(x=optimal_threshold, color='purple', linewidth=2, linestyle=':',
               label=f'Bayes threshold')
    ax.axvline(x=0, color='black', linewidth=1.5, linestyle='--', alpha=0.5,
               label='SVM boundary')

    ax.set_xlabel('Decision Function Value')
    ax.set_ylabel('Weighted Density')
    ax.set_title(f'Overlap Region Detail (Overlap Coeff = {overlap:.4f})', fontweight='bold')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(alpha=0.3)

    # ── Panel (2,1): Statistics summary ──
    ax = axes[2, 1]
    ax.axis('off')

    shapiro_note_eq = "*" if sub_eq else ""
    shapiro_note_ex = "*" if sub_ex else ""

    stats_text = (
        "GAUSSIAN FIT PARAMETERS\n"
        "─────────────────────────────────────\n"
        f"  Earthquake:  mu = {mu_eq:+.4f},  sigma = {sigma_eq:.4f}  (n={n_eq})\n"
        f"  Explosion:   mu = {mu_ex:+.4f},  sigma = {sigma_ex:.4f}  (n={n_ex})\n"
        "\n"
        "SEPARATION METRIC\n"
        "─────────────────────────────────────\n"
        f"  d' (sensitivity index) = {d_prime:.4f}\n"
        "\n"
        "NORMALITY TESTS\n"
        "─────────────────────────────────────\n"
        f"  Earthquake:  Shapiro W={w_eq:.4f} (p={p_eq:.2e}){shapiro_note_eq}\n"
        f"               KS D={ks_eq:.4f} (p={ksp_eq:.2e})\n"
        f"  Explosion:   Shapiro W={w_ex:.4f} (p={p_ex:.2e}){shapiro_note_ex}\n"
        f"               KS D={ks_ex:.4f} (p={ksp_ex:.2e})\n"
        "\n"
        "GMM MODEL COMPARISON\n"
        "─────────────────────────────────────\n"
        f"  1-component:  AIC={aic1:.1f}  BIC={bic1:.1f}\n"
        f"  2-component:  AIC={aic2:.1f}  BIC={bic2:.1f}\n"
        f"  Preferred: {preferred} (delta_BIC={abs(bic1-bic2):.1f})\n"
        "\n"
        "GMM 2-COMPONENT (UNSUPERVISED)\n"
        "─────────────────────────────────────\n"
        f"  Comp 1: mu={gmm_means[0]:+.4f}, sigma={gmm_stds[0]:.4f}, w={gmm_weights[0]:.3f}\n"
        f"  Comp 2: mu={gmm_means[1]:+.4f}, sigma={gmm_stds[1]:.4f}, w={gmm_weights[1]:.3f}\n"
        "\n"
        "THRESHOLD ANALYSIS\n"
        "─────────────────────────────────────\n"
        f"  SVM boundary (t= 0.000):    acc = {acc_svm:.2%}\n"
        f"  Bayes-optimal (t={optimal_threshold:+.3f}):  acc = {acc_bayes:.2%}\n"
        f"  Empirical best (t={empirical_best_threshold:+.3f}): acc = {empirical_best_acc:.2%}\n"
        f"  Improvement: {improvement:+.2f}pp\n"
        "\n"
        "OVERLAP ANALYSIS\n"
        "─────────────────────────────────────\n"
        f"  Overlap coefficient: {overlap:.4f} ({overlap:.2%})\n"
        f"  EQ beyond SVM boundary: {misclass_eq:.1%}\n"
        f"  EX beyond SVM boundary: {misclass_ex:.1%}\n"
    )
    if sub_eq or sub_ex:
        stats_text += "\n  * Shapiro-Wilk subsampled to 5000"

    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            fontsize=9.5, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', edgecolor='#dee2e6'))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(save_dir, "decision_function_analysis.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n  Saved: {save_path}")


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

    # Train k=120 FastMap model
    print("Training k=120 FastMap model...")
    model = FastMapSVMClassifier(k=120, dist_func='euclidean')
    X_trainval_emb = model.fastmap.fit_transform(X_trainval_fft)
    X_test_emb = model.fastmap.transform(X_test_fft)

    scaler = StandardScaler()
    X_trainval_scaled = scaler.fit_transform(X_trainval_emb)
    X_test_scaled = scaler.transform(X_test_emb)

    clf = SVC(C=1000, gamma=0.01, kernel='rbf', class_weight='balanced')
    clf.fit(X_trainval_scaled, y_trainval)

    y_pred_test = clf.predict(X_test_scaled)
    decision_vals_test = clf.decision_function(X_test_scaled)
    test_acc = accuracy_score(y_test, y_pred_test)
    print(f"Full model test accuracy: {test_acc:.4f}\n")

    # Run the analysis
    plot_decision_function_analysis(decision_vals_test, y_test, PLOT_DIR)
