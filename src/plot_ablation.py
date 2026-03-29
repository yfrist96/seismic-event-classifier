"""
Generate all ablation study comparison plots from ablation_summary.json.
Usage: python -m src.plot_ablation
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

SUMMARY_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "output", "ablation", "results", "ablation_summary.json")
PLOT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "output", "ablation", "plots")


def save_figure(fig, path_without_ext):
    """Save figure as both PNG (300 DPI) and PDF (vector)."""
    fig.savefig(path_without_ext + ".png", dpi=300, bbox_inches='tight')
    fig.savefig(path_without_ext + ".pdf", bbox_inches='tight')


def load_summary():
    with open(SUMMARY_PATH, "r") as f:
        data = json.load(f)
    # Separate baseline from experiments
    baseline = [r for r in data if r['experiment'] == 'baseline_RF']
    experiments = [r for r in data if r['experiment'] != 'baseline_RF']
    return baseline, experiments


def filter_exps(experiments, **kwargs):
    results = experiments
    for key, val in kwargs.items():
        results = [r for r in results if r.get(key) == val]
    return results


def get_sorted_k_values(experiments):
    return sorted(set(r['k'] for r in experiments))


def plot_fft_vs_time(experiments, baseline, save_dir):
    """Bar chart: FFT vs Time domain accuracy for each distance metric (ensemble, k=120)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    distances = sorted(set(r['distance'] for r in experiments))
    x = np.arange(len(distances))
    width = 0.35

    fft_accs = []
    time_accs = []
    for dist in distances:
        fft = filter_exps(experiments, domain="FFT", distance=dist, model="ensemble", k=120)
        time = filter_exps(experiments, domain="time", distance=dist, model="ensemble", k=120)
        fft_accs.append(fft[0]['test_accuracy'] if fft else 0)
        time_accs.append(time[0]['test_accuracy'] if time else 0)

    bars1 = ax.bar(x - width/2, fft_accs, width, label='FFT Domain', color='#2ecc71')
    bars2 = ax.bar(x + width/2, time_accs, width, label='Time Domain', color='#e74c3c')

    # Baseline reference line
    if baseline:
        ax.axhline(y=baseline[0]['test_accuracy'], color='gray', linestyle='--',
                    linewidth=1.5, label=f"Baseline RF ({baseline[0]['test_accuracy']:.1%})")

    ax.bar_label(bars1, fmt='%.3f', padding=3, fontsize=9)
    ax.bar_label(bars2, fmt='%.3f', padding=3, fontsize=9)

    ax.set_xlabel('Distance Metric')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('FFT vs Time Domain by Distance Metric (Ensemble, k=120)')
    ax.set_xticks(x)
    ax.set_xticklabels(distances)
    ax.set_ylim(0.5, 1.02)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_figure(plt.gcf(), os.path.join(save_dir, "fft_vs_time"))
    plt.close()
    print("  Saved: fft_vs_time.png/pdf")


def plot_accuracy_vs_k(experiments, baseline, save_dir):
    """Line plot: Test accuracy vs k for each distance metric (FFT, single vs ensemble)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    distances = sorted(set(r['distance'] for r in experiments))
    k_values = get_sorted_k_values(experiments)
    colors = {'euclidean': '#2ecc71', 'lorentzian': '#3498db', 'canberra': '#e74c3c'}

    for idx, model_type in enumerate(['single', 'ensemble']):
        ax = axes[idx]
        for dist in distances:
            accs = []
            for k in k_values:
                exp = filter_exps(experiments, domain="FFT", distance=dist, model=model_type, k=k)
                accs.append(exp[0]['test_accuracy'] if exp else None)
            ax.plot(k_values, accs, marker='o', linewidth=2, markersize=6,
                    label=dist, color=colors.get(dist, 'gray'))

        if baseline:
            ax.axhline(y=baseline[0]['test_accuracy'], color='gray', linestyle='--',
                        linewidth=1.5, label='Baseline RF')

        ax.set_xlabel('FastMap Dimensions (k)')
        ax.set_ylabel('Test Accuracy')
        ax.set_title(f'FFT Domain - {model_type.capitalize()} Model')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_ylim(0.5, 1.0)
        ax.set_xticks(k_values)

    plt.suptitle('Test Accuracy vs FastMap Dimensions (FFT Domain)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(plt.gcf(), os.path.join(save_dir, "accuracy_vs_k_fft"))
    plt.close()
    print("  Saved: accuracy_vs_k_fft.png/pdf")


def plot_accuracy_vs_k_time(experiments, baseline, save_dir):
    """Line plot: Test accuracy vs k for each distance metric (Time domain, single vs ensemble)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    distances = sorted(set(r['distance'] for r in experiments))
    k_values = get_sorted_k_values(experiments)
    colors = {'euclidean': '#2ecc71', 'lorentzian': '#3498db', 'canberra': '#e74c3c'}

    for idx, model_type in enumerate(['single', 'ensemble']):
        ax = axes[idx]
        for dist in distances:
            accs = []
            for k in k_values:
                exp = filter_exps(experiments, domain="time", distance=dist, model=model_type, k=k)
                accs.append(exp[0]['test_accuracy'] if exp else None)
            ax.plot(k_values, accs, marker='o', linewidth=2, markersize=6,
                    label=dist, color=colors.get(dist, 'gray'))

        if baseline:
            ax.axhline(y=baseline[0]['test_accuracy'], color='gray', linestyle='--',
                        linewidth=1.5, label='Baseline RF')

        ax.set_xlabel('FastMap Dimensions (k)')
        ax.set_ylabel('Test Accuracy')
        ax.set_title(f'Time Domain - {model_type.capitalize()} Model')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_ylim(0.5, 1.0)
        ax.set_xticks(k_values)

    plt.suptitle('Test Accuracy vs FastMap Dimensions (Time Domain)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(plt.gcf(), os.path.join(save_dir, "accuracy_vs_k_time"))
    plt.close()
    print("  Saved: accuracy_vs_k_time.png/pdf")


def plot_single_vs_ensemble(experiments, save_dir):
    """Scatter plot: single accuracy vs ensemble accuracy for all configs."""
    fig, ax = plt.subplots(figsize=(8, 8))

    distances = sorted(set(r['distance'] for r in experiments))
    colors = {'euclidean': '#2ecc71', 'lorentzian': '#3498db', 'canberra': '#e74c3c'}
    markers = {'FFT': 'o', 'time': 's'}

    for domain in ['FFT', 'time']:
        for dist in distances:
            k_values = get_sorted_k_values(experiments)
            single_accs = []
            ensemble_accs = []
            for k in k_values:
                s = filter_exps(experiments, domain=domain, distance=dist, model="single", k=k)
                e = filter_exps(experiments, domain=domain, distance=dist, model="ensemble", k=k)
                if s and e:
                    single_accs.append(s[0]['test_accuracy'])
                    ensemble_accs.append(e[0]['test_accuracy'])

            label = f"{domain} {dist}"
            ax.scatter(single_accs, ensemble_accs, color=colors.get(dist, 'gray'),
                       marker=markers[domain], s=60, label=label, alpha=0.8)

    # Diagonal line (ensemble = single)
    lims = [0.5, 1.0]
    ax.plot(lims, lims, 'k--', alpha=0.3, label='No improvement')

    ax.set_xlabel('Single Model Accuracy')
    ax.set_ylabel('Ensemble Accuracy')
    ax.set_title('Single vs Ensemble Model Accuracy')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(alpha=0.3)
    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(0.5, 1.0)
    ax.set_aspect('equal')

    plt.tight_layout()
    save_figure(plt.gcf(), os.path.join(save_dir, "single_vs_ensemble"))
    plt.close()
    print("  Saved: single_vs_ensemble.png/pdf")


def plot_ensemble_gain(experiments, save_dir):
    """Bar chart: accuracy gain from ensemble for each config (FFT domain)."""
    fig, ax = plt.subplots(figsize=(14, 6))

    distances = sorted(set(r['distance'] for r in experiments))
    k_values = get_sorted_k_values(experiments)
    colors = {'euclidean': '#2ecc71', 'lorentzian': '#3498db', 'canberra': '#e74c3c'}

    bar_width = 0.25
    x = np.arange(len(k_values))

    for i, dist in enumerate(distances):
        gains = []
        for k in k_values:
            s = filter_exps(experiments, domain="FFT", distance=dist, model="single", k=k)
            e = filter_exps(experiments, domain="FFT", distance=dist, model="ensemble", k=k)
            if s and e:
                gains.append((e[0]['test_accuracy'] - s[0]['test_accuracy']) * 100)
            else:
                gains.append(0)
        bars = ax.bar(x + i * bar_width, gains, bar_width, label=dist, color=colors.get(dist, 'gray'))
        ax.bar_label(bars, fmt='%.1f%%', padding=3, fontsize=8)

    ax.set_xlabel('FastMap Dimensions (k)')
    ax.set_ylabel('Accuracy Gain (%)')
    ax.set_title('Ensemble Accuracy Gain over Single Model (FFT Domain)')
    ax.set_xticks(x + bar_width)
    ax.set_xticklabels(k_values)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='black', linewidth=0.5)

    plt.tight_layout()
    save_figure(plt.gcf(), os.path.join(save_dir, "ensemble_gain"))
    plt.close()
    print("  Saved: ensemble_gain.png/pdf")


def plot_f1_per_class(experiments, baseline, save_dir):
    """Grouped bar chart: F1 scores for EQ and EX across best configs."""
    # Pick the best config per distance metric (FFT, ensemble)
    fig, ax = plt.subplots(figsize=(12, 6))

    distances = sorted(set(r['distance'] for r in experiments))

    labels = []
    f1_eq = []
    f1_ex = []

    # Baseline first
    if baseline:
        labels.append("Baseline RF")
        f1_eq.append(baseline[0]['test_f1_eq'])
        f1_ex.append(baseline[0]['test_f1_ex'])

    # Best ensemble per distance (FFT)
    for dist in distances:
        fft_ens = filter_exps(experiments, domain="FFT", distance=dist, model="ensemble")
        if fft_ens:
            best = max(fft_ens, key=lambda r: r['test_accuracy'])
            labels.append(f"FFT {dist}\nk={best['k']} ens")
            f1_eq.append(best['test_f1_eq'])
            f1_ex.append(best['test_f1_ex'])

    # Best time domain overall (ensemble)
    time_ens = filter_exps(experiments, domain="time", model="ensemble")
    if time_ens:
        best_time = max(time_ens, key=lambda r: r['test_accuracy'])
        labels.append(f"Time {best_time['distance']}\nk={best_time['k']} ens")
        f1_eq.append(best_time['test_f1_eq'])
        f1_ex.append(best_time['test_f1_ex'])

    x = np.arange(len(labels))
    width = 0.35

    bars1 = ax.bar(x - width/2, f1_eq, width, label='Earthquake (F1)', color='#3498db')
    bars2 = ax.bar(x + width/2, f1_ex, width, label='Explosion (F1)', color='#e74c3c')

    ax.bar_label(bars1, fmt='%.3f', padding=3, fontsize=9)
    ax.bar_label(bars2, fmt='%.3f', padding=3, fontsize=9)

    ax.set_ylabel('F1 Score')
    ax.set_title('Per-Class F1 Scores: Best Configs Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0.4, 1.05)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_figure(plt.gcf(), os.path.join(save_dir, "f1_per_class"))
    plt.close()
    print("  Saved: f1_per_class.png/pdf")


def plot_heatmap_fft(experiments, save_dir):
    """Heatmap: test accuracy by distance x k (FFT domain, ensemble)."""
    distances = sorted(set(r['distance'] for r in experiments))
    k_values = get_sorted_k_values(experiments)

    for model_type in ['single', 'ensemble']:
        fig, ax = plt.subplots(figsize=(10, 5))

        matrix = np.zeros((len(distances), len(k_values)))
        for i, dist in enumerate(distances):
            for j, k in enumerate(k_values):
                exp = filter_exps(experiments, domain="FFT", distance=dist, model=model_type, k=k)
                matrix[i, j] = exp[0]['test_accuracy'] if exp else 0

        im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0.55, vmax=0.95)

        # Annotate cells
        for i in range(len(distances)):
            for j in range(len(k_values)):
                val = matrix[i, j]
                color = 'white' if val < 0.7 else 'black'
                ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=10, color=color)

        ax.set_xticks(range(len(k_values)))
        ax.set_xticklabels(k_values)
        ax.set_yticks(range(len(distances)))
        ax.set_yticklabels(distances)
        ax.set_xlabel('FastMap Dimensions (k)')
        ax.set_ylabel('Distance Metric')
        ax.set_title(f'Test Accuracy Heatmap - FFT Domain ({model_type.capitalize()})')

        plt.colorbar(im, ax=ax, label='Test Accuracy')
        plt.tight_layout()
        save_figure(plt.gcf(), os.path.join(save_dir, f"heatmap_fft_{model_type}"))
        plt.close()
        print(f"  Saved: heatmap_fft_{model_type}.png/pdf")


def plot_heatmap_time(experiments, save_dir):
    """Heatmap: test accuracy by distance x k (Time domain, ensemble)."""
    distances = sorted(set(r['distance'] for r in experiments))
    k_values = get_sorted_k_values(experiments)

    for model_type in ['single', 'ensemble']:
        fig, ax = plt.subplots(figsize=(10, 5))

        matrix = np.zeros((len(distances), len(k_values)))
        for i, dist in enumerate(distances):
            for j, k in enumerate(k_values):
                exp = filter_exps(experiments, domain="time", distance=dist, model=model_type, k=k)
                matrix[i, j] = exp[0]['test_accuracy'] if exp else 0

        im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0.55, vmax=0.95)

        for i in range(len(distances)):
            for j in range(len(k_values)):
                val = matrix[i, j]
                color = 'white' if val < 0.7 else 'black'
                ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=10, color=color)

        ax.set_xticks(range(len(k_values)))
        ax.set_xticklabels(k_values)
        ax.set_yticks(range(len(distances)))
        ax.set_yticklabels(distances)
        ax.set_xlabel('FastMap Dimensions (k)')
        ax.set_ylabel('Distance Metric')
        ax.set_title(f'Test Accuracy Heatmap - Time Domain ({model_type.capitalize()})')

        plt.colorbar(im, ax=ax, label='Test Accuracy')
        plt.tight_layout()
        save_figure(plt.gcf(), os.path.join(save_dir, f"heatmap_time_{model_type}"))
        plt.close()
        print(f"  Saved: heatmap_time_{model_type}.png/pdf")


def plot_val_vs_test(experiments, save_dir):
    """Scatter: validation accuracy vs test accuracy — checks for overfitting to val."""
    fig, ax = plt.subplots(figsize=(8, 8))

    colors = {'FFT': '#2ecc71', 'time': '#e74c3c'}

    for domain in ['FFT', 'time']:
        domain_exps = filter_exps(experiments, domain=domain, model="single")
        val_accs = [r['val_accuracy'] for r in domain_exps]
        test_accs = [r['test_accuracy'] for r in domain_exps]
        ax.scatter(val_accs, test_accs, color=colors[domain], label=f'{domain} domain',
                   alpha=0.7, s=50)

    lims = [0.5, 1.0]
    ax.plot(lims, lims, 'k--', alpha=0.3, label='Perfect correlation')

    ax.set_xlabel('Validation Accuracy')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Validation vs Test Accuracy (Overfitting Check)')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(0.5, 1.0)
    ax.set_aspect('equal')

    plt.tight_layout()
    save_figure(plt.gcf(), os.path.join(save_dir, "val_vs_test"))
    plt.close()
    print("  Saved: val_vs_test.png/pdf")


def plot_overall_ranking(experiments, baseline, save_dir):
    """Horizontal bar chart: top 15 experiments ranked by test accuracy."""
    fig, ax = plt.subplots(figsize=(12, 8))

    all_results = []
    if baseline:
        all_results.append({"label": "Baseline RF", "acc": baseline[0]['test_accuracy'], "domain": "baseline"})

    for r in experiments:
        label = f"{r['domain']} {r['distance']} k={r['k']} {r['model']}"
        all_results.append({"label": label, "acc": r['test_accuracy'], "domain": r['domain']})

    all_results.sort(key=lambda x: x['acc'], reverse=True)
    top = all_results[:15]
    top.reverse()  # so highest is on top in horizontal bar

    colors_map = {'FFT': '#2ecc71', 'time': '#e74c3c', 'baseline': '#95a5a6', 'time_stats': '#95a5a6'}
    bar_colors = [colors_map.get(r['domain'], 'gray') for r in top]

    y_pos = np.arange(len(top))
    bars = ax.barh(y_pos, [r['acc'] for r in top], color=bar_colors, height=0.7)
    ax.bar_label(bars, fmt='%.4f', padding=5, fontsize=9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([r['label'] for r in top], fontsize=9)
    ax.set_xlabel('Test Accuracy')
    ax.set_title('Top 15 Experiments by Test Accuracy')
    ax.set_xlim(0.55, 1.0)
    ax.grid(axis='x', alpha=0.3)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#2ecc71', label='FFT'),
                       Patch(facecolor='#e74c3c', label='Time'),
                       Patch(facecolor='#95a5a6', label='Baseline')]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    save_figure(plt.gcf(), os.path.join(save_dir, "overall_ranking"))
    plt.close()
    print("  Saved: overall_ranking.png/pdf")


def plot_domain_gap(experiments, save_dir):
    """Line plot: FFT accuracy - Time accuracy for each distance at each k (ensemble)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    distances = sorted(set(r['distance'] for r in experiments))
    k_values = get_sorted_k_values(experiments)
    colors = {'euclidean': '#2ecc71', 'lorentzian': '#3498db', 'canberra': '#e74c3c'}

    for dist in distances:
        gaps = []
        for k in k_values:
            fft = filter_exps(experiments, domain="FFT", distance=dist, model="ensemble", k=k)
            time = filter_exps(experiments, domain="time", distance=dist, model="ensemble", k=k)
            if fft and time:
                gaps.append((fft[0]['test_accuracy'] - time[0]['test_accuracy']) * 100)
            else:
                gaps.append(0)
        ax.plot(k_values, gaps, marker='o', linewidth=2, markersize=6,
                label=dist, color=colors.get(dist, 'gray'))

    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_xlabel('FastMap Dimensions (k)')
    ax.set_ylabel('FFT Advantage (percentage points)')
    ax.set_title('FFT vs Time Domain Accuracy Gap (Ensemble)')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xticks(k_values)

    plt.tight_layout()
    save_figure(plt.gcf(), os.path.join(save_dir, "domain_gap"))
    plt.close()
    print("  Saved: domain_gap.png/pdf")


if __name__ == "__main__":
    os.makedirs(PLOT_DIR, exist_ok=True)
    baseline, experiments = load_summary()

    print("Generating ablation plots...\n")

    plot_fft_vs_time(experiments, baseline, PLOT_DIR)
    plot_accuracy_vs_k(experiments, baseline, PLOT_DIR)
    plot_accuracy_vs_k_time(experiments, baseline, PLOT_DIR)
    plot_single_vs_ensemble(experiments, PLOT_DIR)
    plot_ensemble_gain(experiments, PLOT_DIR)
    plot_f1_per_class(experiments, baseline, PLOT_DIR)
    plot_heatmap_fft(experiments, PLOT_DIR)
    plot_heatmap_time(experiments, PLOT_DIR)
    plot_val_vs_test(experiments, PLOT_DIR)
    plot_overall_ranking(experiments, baseline, PLOT_DIR)
    plot_domain_gap(experiments, PLOT_DIR)

    print(f"\nAll plots saved to: {PLOT_DIR}")
