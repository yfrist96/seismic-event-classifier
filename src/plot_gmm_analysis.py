"""
Deep statistical analysis of SVM decision function distribution.

Out-of-sample calibration: the class-conditional Gaussians, the 2-component GMM,
the normality tests, d-prime, and the Bayes-optimal threshold are all estimated
on the VALIDATION set, which is held out from model training (the model is trained
on the TRAINING split only). Classification accuracy at the default (tau=0) and
Bayes-optimal thresholds is then reported on the disjoint TEST set. This keeps the
threshold calibration and its evaluation on separate data, so the Gaussian
structure and the recalibration benefit are a genuine out-of-sample result rather
than an in-sample artifact.

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
from scipy.integrate import trapezoid  # stable across NumPy 1.x/2.x (np.trapz removed in NumPy 2.0)

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


def plot_decision_function_analysis(val_vals, y_val, test_vals, y_test, save_dir):
    """Out-of-sample statistical analysis of the SVM decision function.

    The Gaussian model (per-class fits, GMM, normality tests, d', Bayes-optimal
    threshold) is estimated on the VALIDATION decision values, which are held out
    from model training. Classification accuracy is then reported on the disjoint
    TEST set, applying the validation-derived threshold. This isolates the
    calibration claim from the data used to evaluate it.

    Args:
        val_vals:  SVM decision-function values on the validation set (calibration).
        y_val:     validation labels (0=Earthquake, 1=Explosion).
        test_vals: SVM decision-function values on the test set (evaluation).
        y_test:    test labels.
        save_dir:  directory to save the output figures.
    """
    print("  === Decision Function Statistical Analysis (out-of-sample) ===\n")

    # ── Calibration set (validation): fit the Gaussian model ──
    eq_vals = val_vals[y_val == 0]
    ex_vals = val_vals[y_val == 1]

    mu_eq, sigma_eq = norm.fit(eq_vals)
    mu_ex, sigma_ex = norm.fit(ex_vals)
    n_eq, n_ex = len(eq_vals), len(ex_vals)
    n_total = n_eq + n_ex

    print(f"  Per-class Gaussian fits (VALIDATION / calibration set):")
    print(f"    Earthquake:  mu = {mu_eq:+.4f},  sigma = {sigma_eq:.4f},  n = {n_eq}")
    print(f"    Explosion:   mu = {mu_ex:+.4f},  sigma = {sigma_ex:.4f},  n = {n_ex}")

    # ── Test set: used only for evaluation and the generalization figure ──
    test_eq_vals = test_vals[y_test == 0]
    test_ex_vals = test_vals[y_test == 1]
    mu_eq_t, sigma_eq_t = norm.fit(test_eq_vals)
    mu_ex_t, sigma_ex_t = norm.fit(test_ex_vals)

    # ── d-prime (sensitivity index) on both splits ──
    d_prime = abs(mu_ex - mu_eq) / np.sqrt(0.5 * (sigma_eq**2 + sigma_ex**2))
    d_prime_test = abs(mu_ex_t - mu_eq_t) / np.sqrt(0.5 * (sigma_eq_t**2 + sigma_ex_t**2))
    print(f"\n  Separation metrics:")
    print(f"    d-prime (validation, calibration): {d_prime:.4f}")
    print(f"    d-prime (test, out-of-sample):     {d_prime_test:.4f}")

    # ── Normality tests (validation set) ──
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

    print(f"\n  Normality tests (validation set):")
    sub_note_eq = " (subsampled to 5000)" if sub_eq else ""
    sub_note_ex = " (subsampled to 5000)" if sub_ex else ""
    print(f"    Earthquake:  Shapiro W={w_eq:.4f} (p={p_eq:.4e}){sub_note_eq}  |  KS D={ks_eq:.4f} (p={ksp_eq:.4e})")
    print(f"    Explosion:   Shapiro W={w_ex:.4f} (p={p_ex:.4e}){sub_note_ex}  |  KS D={ks_ex:.4f} (p={ksp_ex:.4e})")

    # ── GMM model comparison (unsupervised, validation set) ──
    X_gmm = val_vals.reshape(-1, 1)
    gmm1 = GaussianMixture(n_components=1, random_state=42, n_init=5).fit(X_gmm)
    gmm2 = GaussianMixture(n_components=2, random_state=42, n_init=5).fit(X_gmm)

    aic1, bic1 = gmm1.aic(X_gmm), gmm1.bic(X_gmm)
    aic2, bic2 = gmm2.aic(X_gmm), gmm2.bic(X_gmm)

    # Sort GMM components by mean
    order = np.argsort(gmm2.means_.ravel())
    gmm_means = gmm2.means_.ravel()[order]
    gmm_stds = np.sqrt(gmm2.covariances_.ravel()[order])
    gmm_weights = gmm2.weights_[order]

    print(f"\n  GMM model comparison (unsupervised on validation decision values):")
    print(f"    1-component:  AIC = {aic1:.1f},  BIC = {bic1:.1f}")
    print(f"    2-component:  AIC = {aic2:.1f},  BIC = {bic2:.1f}")
    preferred = "2-component" if bic2 < bic1 else "1-component"
    print(f"    Preferred: {preferred} (delta_BIC = {abs(bic1 - bic2):.1f})")

    print(f"\n  GMM 2-component parameters (unsupervised):")
    print(f"    Component 1: mu={gmm_means[0]:+.4f}, sigma={gmm_stds[0]:.4f}, weight={gmm_weights[0]:.4f}")
    print(f"    Component 2: mu={gmm_means[1]:+.4f}, sigma={gmm_stds[1]:.4f}, weight={gmm_weights[1]:.4f}")

    # ── Bayes-optimal threshold (estimated on validation priors + fits) ──
    prior_eq = n_eq / n_total
    prior_ex = n_ex / n_total

    def posterior_diff(x):
        """prior_eq * pdf_eq(x) - prior_ex * pdf_ex(x)"""
        return prior_eq * norm.pdf(x, mu_eq, sigma_eq) - prior_ex * norm.pdf(x, mu_ex, sigma_ex)

    try:
        optimal_threshold = brentq(posterior_diff, mu_eq, mu_ex)
    except ValueError:
        # Fallback: grid search
        x_grid = np.linspace(val_vals.min(), val_vals.max(), 10000)
        diffs = np.abs([posterior_diff(x) for x in x_grid])
        optimal_threshold = x_grid[np.argmin(diffs)]

    # ── Accuracy: default vs validation-derived Bayes threshold ──
    # Calibration threshold comes from validation; the headline numbers are on TEST.
    acc_svm_val = accuracy_score(y_val, (val_vals > 0).astype(int))
    acc_bayes_val = accuracy_score(y_val, (val_vals > optimal_threshold).astype(int))
    acc_svm_test = accuracy_score(y_test, (test_vals > 0).astype(int))
    acc_bayes_test = accuracy_score(y_test, (test_vals > optimal_threshold).astype(int))

    # Test-set threshold sweep, for the figure only. The oracle "empirical best"
    # on test is reported purely as an upper bound; it is NOT used to pick tau.
    thresholds = np.linspace(test_vals.min(), test_vals.max(), 2000)
    accuracies = np.array([
        accuracy_score(y_test, (test_vals > t).astype(int)) for t in thresholds
    ])
    best_idx = np.argmax(accuracies)
    empirical_best_threshold = thresholds[best_idx]
    empirical_best_acc = accuracies[best_idx]

    print(f"\n  Threshold analysis (calibrated on validation, evaluated on test):")
    print(f"    Validation  SVM (t=0.000):          accuracy = {acc_svm_val:.2%}")
    print(f"    Validation  Bayes (t={optimal_threshold:+.3f}):     accuracy = {acc_bayes_val:.2%}")
    print(f"    TEST        SVM (t=0.000):          accuracy = {acc_svm_test:.2%}")
    print(f"    TEST        Bayes (t={optimal_threshold:+.3f}):     accuracy = {acc_bayes_test:.2%}")
    print(f"    TEST        empirical best (t={empirical_best_threshold:+.3f}) [oracle]: accuracy = {empirical_best_acc:.2%}")
    improvement = (acc_bayes_test - acc_svm_test) * 100
    print(f"    Out-of-sample gain from Bayes recalibration: {improvement:+.2f}pp")

    # ── Overlap coefficient (validation fits) ──
    x_grid = np.linspace(val_vals.min() - 1, val_vals.max() + 1, 5000)
    pdf_eq = prior_eq * norm.pdf(x_grid, mu_eq, sigma_eq)
    pdf_ex = prior_ex * norm.pdf(x_grid, mu_ex, sigma_ex)
    overlap = trapezoid(np.minimum(pdf_eq, pdf_ex), x_grid)

    misclass_eq = np.mean(eq_vals > 0)
    misclass_ex = np.mean(ex_vals < 0)

    print(f"\n  Overlap analysis (validation set):")
    print(f"    Overlap coefficient: {overlap:.4f} ({overlap:.2%} of total density)")
    print(f"    EQ beyond SVM boundary: {misclass_eq:.1%}")
    print(f"    EX beyond SVM boundary: {misclass_ex:.1%}")

    # ═══════════════════════════════════════════════════════════
    # Figure: 3x2 grid. Distributional diagnostics (panels showing the fit) use
    # the VALIDATION set; the accuracy panel uses the TEST set to show the
    # validation-derived threshold generalizes out of sample.
    # ═══════════════════════════════════════════════════════════
    print(f"\n  Generating analysis figure...")

    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle('Statistical Analysis of SVM Decision Function (validation-calibrated, test-evaluated)',
                 fontsize=16, fontweight='bold', y=0.98)

    x_plot = np.linspace(val_vals.min() - 0.5, val_vals.max() + 0.5, 500)
    bins = np.linspace(val_vals.min(), val_vals.max(), 60)

    # ── Panel (0,0): Validation histogram + fitted PDFs ──
    ax = axes[0, 0]
    ax.hist(eq_vals, bins=bins, alpha=0.4, color='#3498db', label='Earthquake (val data)', density=True)
    ax.hist(ex_vals, bins=bins, alpha=0.4, color='#e74c3c', label='Explosion (val data)', density=True)

    ax.plot(x_plot, norm.pdf(x_plot, mu_eq, sigma_eq) * prior_eq,
            color='#2980b9', linewidth=2.5, linestyle='-',
            label=f'EQ fit: N({mu_eq:.2f}, {sigma_eq:.2f})')
    ax.plot(x_plot, norm.pdf(x_plot, mu_ex, sigma_ex) * prior_ex,
            color='#c0392b', linewidth=2.5, linestyle='-',
            label=f'EX fit: N({mu_ex:.2f}, {sigma_ex:.2f})')

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
    ax.set_title('Validation Histogram with Fitted Gaussian PDFs', fontweight='bold')
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(alpha=0.2)

    # ── Panel (0,1): Q-Q plots (validation) ──
    ax = axes[0, 1]
    for vals, name, color in [(eq_vals, 'Earthquake', '#3498db'), (ex_vals, 'Explosion', '#e74c3c')]:
        osm, osr = probplot(vals, dist="norm", fit=False)
        ax.scatter(osm, osr, color=color, s=8, alpha=0.5, label=name, edgecolors='none')

    all_quantiles = np.concatenate([probplot(eq_vals, dist="norm", fit=False)[0],
                                     probplot(ex_vals, dist="norm", fit=False)[0]])
    q_min, q_max = all_quantiles.min(), all_quantiles.max()

    for vals, color in [(eq_vals, '#2980b9'), (ex_vals, '#c0392b')]:
        osm, osr = probplot(vals, dist="norm", fit=False)
        slope, intercept = np.polyfit(osm, osr, 1)
        ax.plot([q_min, q_max], [slope * q_min + intercept, slope * q_max + intercept],
                color=color, linewidth=1.5, linestyle='--', alpha=0.7)

    ax.set_xlabel('Theoretical Quantiles')
    ax.set_ylabel('Sample Quantiles')
    ax.set_title('Q-Q Plots (Normal Distribution, validation)', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # ── Panel (1,0): Empirical vs fitted CDFs (validation) ──
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

    overlap_left = max(eq_vals.min(), ex_vals.min())
    overlap_right = min(eq_vals.max(), ex_vals.max())
    ax.axvspan(overlap_left, overlap_right, alpha=0.08, color='gray', label='Overlap region')

    ax.set_xlabel('Decision Function Value')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Empirical vs Fitted CDFs (validation)', fontweight='bold')
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(alpha=0.3)

    # ── Panel (1,1): Accuracy vs threshold sweep (TEST set) ──
    ax = axes[1, 1]
    ax.plot(thresholds, accuracies, color='#2c3e50', linewidth=1.5)

    ax.axvline(x=0, color='black', linewidth=1.5, linestyle='--',
               label=f'SVM (t=0, test acc={acc_svm_test:.2%})')
    ax.axvline(x=optimal_threshold, color='purple', linewidth=1.5, linestyle=':',
               label=f'Bayes val-derived (t={optimal_threshold:.3f}, test acc={acc_bayes_test:.2%})')
    ax.axvline(x=empirical_best_threshold, color='green', linewidth=1.5, linestyle='-.',
               label=f'Test-oracle best (t={empirical_best_threshold:.3f}, acc={empirical_best_acc:.2%})')

    ax.scatter([0], [acc_svm_test], color='black', s=60, zorder=5)
    ax.scatter([optimal_threshold], [acc_bayes_test], color='purple', s=60, zorder=5)
    ax.scatter([empirical_best_threshold], [empirical_best_acc], color='green', s=60, zorder=5)

    ax.set_xlabel('Decision Threshold')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Test Accuracy vs Decision Threshold', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ── Panel (2,0): Overlap detail (validation fits, zoomed) ──
    ax = axes[2, 0]
    zoom_left = optimal_threshold - 2 * max(sigma_eq, sigma_ex)
    zoom_right = optimal_threshold + 2 * max(sigma_eq, sigma_ex)
    x_zoom = np.linspace(zoom_left, zoom_right, 500)

    pdf_eq_zoom = prior_eq * norm.pdf(x_zoom, mu_eq, sigma_eq)
    pdf_ex_zoom = prior_ex * norm.pdf(x_zoom, mu_ex, sigma_ex)

    ax.plot(x_zoom, pdf_eq_zoom, color='#3498db', linewidth=2, label='Earthquake PDF')
    ax.plot(x_zoom, pdf_ex_zoom, color='#e74c3c', linewidth=2, label='Explosion PDF')

    mask_eq_error = x_zoom > optimal_threshold
    ax.fill_between(x_zoom[mask_eq_error], 0, pdf_eq_zoom[mask_eq_error],
                     color='#3498db', alpha=0.3, label='EQ misclassified')
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
        "GAUSSIAN FIT PARAMETERS (VALIDATION / calibration)\n"
        "─────────────────────────────────────\n"
        f"  Earthquake:  mu = {mu_eq:+.4f},  sigma = {sigma_eq:.4f}  (n={n_eq})\n"
        f"  Explosion:   mu = {mu_ex:+.4f},  sigma = {sigma_ex:.4f}  (n={n_ex})\n"
        "\n"
        "SEPARATION METRIC (d')\n"
        "─────────────────────────────────────\n"
        f"  Validation (calibration): d' = {d_prime:.4f}\n"
        f"  Test (out-of-sample):     d' = {d_prime_test:.4f}\n"
        "\n"
        "NORMALITY TESTS (validation)\n"
        "─────────────────────────────────────\n"
        f"  Earthquake:  Shapiro W={w_eq:.4f} (p={p_eq:.2e}){shapiro_note_eq}\n"
        f"               KS D={ks_eq:.4f} (p={ksp_eq:.2e})\n"
        f"  Explosion:   Shapiro W={w_ex:.4f} (p={p_ex:.2e}){shapiro_note_ex}\n"
        f"               KS D={ks_ex:.4f} (p={ksp_ex:.2e})\n"
        "\n"
        "GMM MODEL COMPARISON (validation)\n"
        "─────────────────────────────────────\n"
        f"  1-component:  AIC={aic1:.1f}  BIC={bic1:.1f}\n"
        f"  2-component:  AIC={aic2:.1f}  BIC={bic2:.1f}\n"
        f"  Preferred: {preferred} (delta_BIC={abs(bic1-bic2):.1f})\n"
        "\n"
        "THRESHOLD ANALYSIS (calibrated on val, evaluated on test)\n"
        "─────────────────────────────────────\n"
        f"  Bayes-optimal threshold (from val): t = {optimal_threshold:+.3f}\n"
        f"  TEST  SVM boundary (t= 0.000):  acc = {acc_svm_test:.2%}\n"
        f"  TEST  Bayes (t={optimal_threshold:+.3f}):      acc = {acc_bayes_test:.2%}\n"
        f"  TEST  oracle best (t={empirical_best_threshold:+.3f}): acc = {empirical_best_acc:.2%}\n"
        f"  Out-of-sample Bayes gain: {improvement:+.2f}pp\n"
        "\n"
        "OVERLAP ANALYSIS (validation)\n"
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
    save_path = os.path.join(save_dir, "decision_function_analysis")
    save_figure(fig, save_path)
    plt.close()
    print(f"\n  Saved: {save_path}.png/pdf")

    # ═══════════════════════════════════════════════════════════
    # Standalone panel (a) — the main-text figure. Shows the held-out TEST
    # decision-value histogram overlaid with the VALIDATION-derived Gaussian
    # fits and the VALIDATION-derived Bayes-optimal threshold. The curves are
    # calibrated on data disjoint from the histogram, so a good match is a
    # genuine out-of-sample result.
    # ═══════════════════════════════════════════════════════════
    fig_a, ax = plt.subplots(1, 1, figsize=(7, 4.6))

    x_plot_t = np.linspace(test_vals.min() - 0.5, test_vals.max() + 0.5, 500)
    bins_t = np.linspace(test_vals.min(), test_vals.max(), 60)

    ax.hist(test_eq_vals, bins=bins_t, alpha=0.4, color='#3498db',
            label='Earthquake (test data)', density=True)
    ax.hist(test_ex_vals, bins=bins_t, alpha=0.4, color='#e74c3c',
            label='Explosion (test data)', density=True)

    # Validation-derived Gaussian fits (calibrated on held-out data)
    ax.plot(x_plot_t, norm.pdf(x_plot_t, mu_eq, sigma_eq) * prior_eq,
            color='#2980b9', linewidth=2.5, linestyle='-',
            label=f'EQ fit (val): N({mu_eq:.2f}, {sigma_eq:.2f})')
    ax.plot(x_plot_t, norm.pdf(x_plot_t, mu_ex, sigma_ex) * prior_ex,
            color='#c0392b', linewidth=2.5, linestyle='-',
            label=f'EX fit (val): N({mu_ex:.2f}, {sigma_ex:.2f})')

    gmm_pdf_t = np.zeros_like(x_plot_t)
    for k in range(2):
        gmm_pdf_t += gmm_weights[k] * norm.pdf(x_plot_t, gmm_means[k], gmm_stds[k])
    ax.plot(x_plot_t, gmm_pdf_t, color='green', linewidth=2, linestyle='--',
            label='GMM (val, unsupervised)', alpha=0.8)

    ax.axvline(x=0, color='black', linewidth=1.5, linestyle='--', alpha=0.7,
               label='SVM boundary ($\\tau$=0)')
    ax.axvline(x=optimal_threshold, color='purple', linewidth=1.5, linestyle=':',
               label=f'Bayes-optimal ($\\tau$={optimal_threshold:.3f}, val-derived)')

    ax.set_xlabel('Decision Function Value (test set)')
    ax.set_ylabel('Density')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(alpha=0.2)

    fig_a.tight_layout()
    save_path_a = os.path.join(save_dir, "decision_gaussian_histogram")
    save_figure(fig_a, save_path_a)
    plt.close()
    print(f"  Saved standalone panel (a): {save_path_a}.png/pdf")


if __name__ == "__main__":
    os.makedirs(PLOT_DIR, exist_ok=True)

    # Load data
    print("Loading data...")
    (X_train_raw, y_train), (X_val_raw, y_val), (X_test_raw, y_test) = load_data(DATA_DIR)

    # FFT features. The model is trained on the TRAIN split ONLY so that the
    # validation set is a genuine held-out calibration set (no leakage into the
    # Gaussian/threshold fit). Validation calibrates the threshold; test evaluates it.
    print("Computing FFT features...")
    X_train_fft = to_frequency_domain(X_train_raw)
    X_val_fft = to_frequency_domain(X_val_raw)
    X_test_fft = to_frequency_domain(X_test_raw)

    # Train k=120 FastMap model on TRAIN only
    print("Training k=120 FastMap model (train split only)...")
    np.random.seed(42)
    model = FastMapSVMClassifier(k=120, dist_func='euclidean')
    X_train_emb = model.fastmap.fit_transform(X_train_fft)
    X_val_emb = model.fastmap.transform(X_val_fft)
    X_test_emb = model.fastmap.transform(X_test_fft)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_emb)
    X_val_scaled = scaler.transform(X_val_emb)
    X_test_scaled = scaler.transform(X_test_emb)

    clf = SVC(C=1000, gamma=0.01, kernel='rbf', class_weight='balanced')
    clf.fit(X_train_scaled, y_train)

    decision_vals_val = clf.decision_function(X_val_scaled)
    decision_vals_test = clf.decision_function(X_test_scaled)
    val_acc = accuracy_score(y_val, clf.predict(X_val_scaled))
    test_acc = accuracy_score(y_test, clf.predict(X_test_scaled))
    print(f"Train-only model: validation accuracy = {val_acc:.4f}, test accuracy = {test_acc:.4f}\n")

    # Run the analysis: calibrate on validation, evaluate on test
    plot_decision_function_analysis(decision_vals_val, y_val,
                                    decision_vals_test, y_test, PLOT_DIR)
