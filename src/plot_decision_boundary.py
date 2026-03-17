"""
Decision boundary and spectral analysis plots.
Visualizes how the SVM separates earthquakes from explosions
and provides physical interpretation of the frequency-domain features.

Usage: python -m src.plot_decision_boundary
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score

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


def to_frequency_domain_raw(X):
    """Return log-magnitude FFT without L2 normalization (for spectral analysis)."""
    X_fft = np.fft.rfft(X, axis=1)
    X_mag = np.abs(X_fft)
    X_mag = np.log10(X_mag + 1e-6)
    return X_mag


def plot_decision_mesh(ax, clf, X, y, title):
    """Plot 2D decision boundary with mesh and data points."""
    cmap_light = ListedColormap(['#AADDFF', '#FFAAAA'])

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    h = max((x_max - x_min) / 300, (y_max - y_min) / 300)
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                          np.arange(y_min, y_max, h))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.6)
    ax.contour(xx, yy, Z, colors='k', linewidths=0.5, alpha=0.3)

    for label, name, color, marker in [(0, 'Earthquake', '#3498db', 'o'),
                                        (1, 'Explosion', '#e74c3c', 's')]:
        mask = y == label
        ax.scatter(X[mask, 0], X[mask, 1], c=color, marker=marker,
                   s=15, alpha=0.5, label=name, edgecolors='none')

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='best')


def plot_tsne_clusters(X_emb_scaled, y, decision_vals, save_dir):
    """t-SNE on high-k embeddings colored by class and by SVM confidence."""
    print("  Generating t-SNE visualizations...")

    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', max_iter=1000)
    X_tsne = tsne.fit_transform(X_emb_scaled)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Left: colored by class
    ax = axes[0]
    for label, name, color, marker in [(0, 'Earthquake', '#3498db', 'o'),
                                        (1, 'Explosion', '#e74c3c', 's')]:
        mask = y == label
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=color, marker=marker,
                   s=12, alpha=0.5, label=name, edgecolors='none')
    ax.set_title('t-SNE: Class Labels', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.grid(alpha=0.2)

    # Right: colored by SVM decision function (confidence)
    ax = axes[1]
    scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=decision_vals,
                          cmap='RdBu_r', s=12, alpha=0.5, edgecolors='none')
    ax.set_title('t-SNE: SVM Confidence\n(Blue = Earthquake, Red = Explosion)',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.grid(alpha=0.2)
    plt.colorbar(scatter, ax=ax, label='SVM decision value')

    plt.suptitle('t-SNE of k=120 FastMap Embeddings (FFT + Euclidean)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "tsne_clusters.png"), dpi=150)
    plt.close()
    print("  Saved: tsne_clusters.png")

    return X_tsne


def plot_decision_function_histogram(decision_vals, y, save_dir):
    """Histogram of SVM decision function values per class — shows separation without 2D."""
    print("  Generating decision function histogram...")

    fig, ax = plt.subplots(figsize=(10, 6))

    eq_vals = decision_vals[y == 0]
    ex_vals = decision_vals[y == 1]

    bins = np.linspace(decision_vals.min(), decision_vals.max(), 80)

    ax.hist(eq_vals, bins=bins, alpha=0.6, color='#3498db', label='Earthquake', density=True)
    ax.hist(ex_vals, bins=bins, alpha=0.6, color='#e74c3c', label='Explosion', density=True)

    ax.axvline(x=0, color='black', linewidth=2, linestyle='--', label='Decision boundary')

    # Annotate overlap region
    overlap_eq = np.sum(eq_vals > 0) / len(eq_vals)
    overlap_ex = np.sum(ex_vals < 0) / len(ex_vals)

    ax.text(0.02, 0.95, f'EQ misclassified: {overlap_eq:.1%}\nEX misclassified: {overlap_ex:.1%}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('SVM Decision Function Value', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Class Separation in SVM Decision Space\n'
                 '(Negative = Earthquake, Positive = Explosion)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "decision_histogram.png"), dpi=150)
    plt.close()
    print("  Saved: decision_histogram.png")


def plot_best_2_dims(X_emb, y, save_dir):
    """Find the 2 most discriminative FastMap dimensions and plot boundary on those."""
    print("  Generating best-2-dimensions decision boundary...")

    # Rank dimensions by class separability (difference in means / pooled std)
    n_dims = X_emb.shape[1]
    scores = np.zeros(n_dims)
    for d in range(n_dims):
        eq_vals = X_emb[y == 0, d]
        ex_vals = X_emb[y == 1, d]
        pooled_std = np.sqrt((np.var(eq_vals) + np.var(ex_vals)) / 2)
        if pooled_std > 1e-9:
            scores[d] = abs(np.mean(eq_vals) - np.mean(ex_vals)) / pooled_std
        else:
            scores[d] = 0

    best_dims = np.argsort(scores)[-2:]
    d1, d2 = best_dims[0], best_dims[1]

    X_2d = X_emb[:, [d1, d2]]

    scaler = StandardScaler()
    X_2d_scaled = scaler.fit_transform(X_2d)

    clf = SVC(C=1000, gamma=0.01, kernel='rbf')
    clf.fit(X_2d_scaled, y)
    acc = accuracy_score(y, clf.predict(X_2d_scaled))

    fig, ax = plt.subplots(figsize=(10, 8))
    plot_decision_mesh(ax, clf, X_2d_scaled, y,
                       f'SVM on 2 Most Discriminative FastMap Dims (train acc={acc:.1%})')
    ax.set_xlabel(f'FastMap Dim {d1+1} (separability={scores[d1]:.2f})')
    ax.set_ylabel(f'FastMap Dim {d2+1} (separability={scores[d2]:.2f})')

    plt.suptitle('Decision Boundary on Best 2 of 120 FastMap Dimensions',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "decision_boundary_best2dims.png"), dpi=150)
    plt.close()
    print(f"  Saved: decision_boundary_best2dims.png (dims {d1+1} & {d2+1})")


def plot_misclassified_on_tsne(X_tsne, y, y_pred, save_dir):
    """t-SNE with misclassified points highlighted."""
    print("  Generating misclassification map...")

    fig, ax = plt.subplots(figsize=(10, 8))

    correct = y == y_pred
    wrong = ~correct

    # Plot correct points faintly
    for label, name, color in [(0, 'EQ correct', '#3498db'), (1, 'EX correct', '#e74c3c')]:
        mask = (y == label) & correct
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=color,
                   s=8, alpha=0.2, edgecolors='none', label=name)

    # Plot misclassified points boldly
    ax.scatter(X_tsne[wrong, 0], X_tsne[wrong, 1], c='black', marker='x',
               s=40, linewidths=1.5, alpha=0.8, label=f'Misclassified ({wrong.sum()})')

    ax.set_title(f'Misclassified Samples on t-SNE ({wrong.sum()}/{len(y)} = {wrong.mean():.1%})',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "misclassified_tsne.png"), dpi=150)
    plt.close()
    print("  Saved: misclassified_tsne.png")


def plot_average_spectra(X_train_raw, y_train, save_dir):
    """Plot mean FFT magnitude spectrum for EQ vs EX per channel."""
    print("  Generating average spectra per class...")

    X_fft = to_frequency_domain_raw(X_train_raw)  # (N, 301, 3)
    sample_rate = 100
    freqs = np.fft.rfftfreq(600, d=1.0/sample_rate)

    channel_names = ['Vertical (Z)', 'North-South (N)', 'East-West (E)']
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    for ch in range(3):
        ax = axes[ch]
        eq_mask = y_train == 0
        ex_mask = y_train == 1

        eq_mean = np.mean(X_fft[eq_mask, :, ch], axis=0)
        ex_mean = np.mean(X_fft[ex_mask, :, ch], axis=0)
        eq_std = np.std(X_fft[eq_mask, :, ch], axis=0)
        ex_std = np.std(X_fft[ex_mask, :, ch], axis=0)

        ax.plot(freqs, eq_mean, color='#3498db', linewidth=1.5, label='Earthquake (mean)')
        ax.fill_between(freqs, eq_mean - eq_std, eq_mean + eq_std,
                         color='#3498db', alpha=0.15, label='Earthquake (std)')
        ax.plot(freqs, ex_mean, color='#e74c3c', linewidth=1.5, label='Explosion (mean)')
        ax.fill_between(freqs, ex_mean - ex_std, ex_mean + ex_std,
                         color='#e74c3c', alpha=0.15, label='Explosion (std)')

        ax.set_ylabel('Log10 Magnitude')
        ax.set_title(f'{channel_names[ch]} Channel', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel('Frequency (Hz)')
    plt.suptitle('Average FFT Spectrum: Earthquake vs Explosion',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "average_spectra.png"), dpi=150)
    plt.close()
    print("  Saved: average_spectra.png")


def plot_spectral_difference(X_train_raw, y_train, save_dir):
    """Plot the difference in mean spectra (EX - EQ) to highlight discriminative frequencies."""
    print("  Generating spectral difference plot...")

    X_fft = to_frequency_domain_raw(X_train_raw)
    sample_rate = 100
    freqs = np.fft.rfftfreq(600, d=1.0/sample_rate)

    channel_names = ['Vertical (Z)', 'North-South (N)', 'East-West (E)']
    channel_colors = ['#2ecc71', '#3498db', '#e74c3c']

    fig, ax = plt.subplots(figsize=(14, 6))

    for ch in range(3):
        eq_mean = np.mean(X_fft[y_train == 0, :, ch], axis=0)
        ex_mean = np.mean(X_fft[y_train == 1, :, ch], axis=0)
        diff = ex_mean - eq_mean
        ax.plot(freqs, diff, color=channel_colors[ch], linewidth=1.5,
                label=f'{channel_names[ch]}', alpha=0.8)

    ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Log-Magnitude Difference (Explosion - Earthquake)')
    ax.set_title('Spectral Difference: Explosion minus Earthquake\n'
                 '(Positive = higher energy in explosions, Negative = higher in earthquakes)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "spectral_difference.png"), dpi=150)
    plt.close()
    print("  Saved: spectral_difference.png")


def plot_example_waveforms(X_train_raw, y_train, save_dir):
    """Plot example time-domain waveforms for EQ vs EX side by side."""
    print("  Generating example waveform comparison...")

    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharey='row')
    channel_names = ['Vertical (Z)', 'North-South (N)', 'East-West (E)']

    np.random.seed(42)
    eq_idx = np.random.choice(np.where(y_train == 0)[0])
    ex_idx = np.random.choice(np.where(y_train == 1)[0])

    t = np.arange(600) / 100.0

    for ch in range(3):
        axes[0, ch].plot(t, X_train_raw[eq_idx, :, ch], color='#3498db', linewidth=0.8)
        axes[0, ch].set_title(f'{channel_names[ch]}', fontsize=11)
        axes[0, ch].grid(alpha=0.3)
        if ch == 0:
            axes[0, ch].set_ylabel('Earthquake\nAmplitude')

        axes[1, ch].plot(t, X_train_raw[ex_idx, :, ch], color='#e74c3c', linewidth=0.8)
        axes[1, ch].grid(alpha=0.3)
        axes[1, ch].set_xlabel('Time (s)')
        if ch == 0:
            axes[1, ch].set_ylabel('Explosion\nAmplitude')

    plt.suptitle('Example Waveforms: Earthquake vs Explosion',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "example_waveforms.png"), dpi=150)
    plt.close()
    print("  Saved: example_waveforms.png")


def plot_example_spectra(X_train_raw, y_train, save_dir):
    """Plot the FFT of example waveforms to show spectral differences."""
    print("  Generating example spectra comparison...")

    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharey='row')
    channel_names = ['Vertical (Z)', 'North-South (N)', 'East-West (E)']

    np.random.seed(42)
    eq_idx = np.random.choice(np.where(y_train == 0)[0])
    ex_idx = np.random.choice(np.where(y_train == 1)[0])

    sample_rate = 100
    freqs = np.fft.rfftfreq(600, d=1.0/sample_rate)
    X_fft = to_frequency_domain_raw(X_train_raw)

    for ch in range(3):
        axes[0, ch].plot(freqs, X_fft[eq_idx, :, ch], color='#3498db', linewidth=0.8)
        axes[0, ch].set_title(f'{channel_names[ch]}', fontsize=11)
        axes[0, ch].grid(alpha=0.3)
        if ch == 0:
            axes[0, ch].set_ylabel('Earthquake\nLog Magnitude')

        axes[1, ch].plot(freqs, X_fft[ex_idx, :, ch], color='#e74c3c', linewidth=0.8)
        axes[1, ch].grid(alpha=0.3)
        axes[1, ch].set_xlabel('Frequency (Hz)')
        if ch == 0:
            axes[1, ch].set_ylabel('Explosion\nLog Magnitude')

    plt.suptitle('Example FFT Spectra: Earthquake vs Explosion',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "example_spectra.png"), dpi=150)
    plt.close()
    print("  Saved: example_spectra.png")


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

    print(f"\nGenerating plots to: {PLOT_DIR}\n")

    # --- Build the full k=120 model (used by multiple plots) ---
    print("  Training k=120 FastMap model...")
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
    print(f"  Full model test accuracy: {test_acc:.4f}\n")

    # 1. t-SNE with class colors and SVM confidence
    X_tsne = plot_tsne_clusters(X_test_scaled, y_test, decision_vals_test, PLOT_DIR)

    # 2. Decision function histogram
    plot_decision_function_histogram(decision_vals_test, y_test, PLOT_DIR)

    # 3. Best 2 discriminative dims boundary
    plot_best_2_dims(X_trainval_scaled, y_trainval, PLOT_DIR)

    # 4. Misclassified points on t-SNE
    plot_misclassified_on_tsne(X_tsne, y_test, y_pred_test, PLOT_DIR)

    # 5. Physical interpretation plots
    plot_average_spectra(X_trainval_raw, y_trainval, PLOT_DIR)
    plot_spectral_difference(X_trainval_raw, y_trainval, PLOT_DIR)
    plot_example_waveforms(X_trainval_raw, y_trainval, PLOT_DIR)
    plot_example_spectra(X_trainval_raw, y_trainval, PLOT_DIR)

    print(f"\nAll plots saved to: {PLOT_DIR}")
