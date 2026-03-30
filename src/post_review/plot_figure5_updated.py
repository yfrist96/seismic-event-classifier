"""
Generate updated Figure 5: Top 5 FFT models, best time-series model, and baseline.
Horizontal bar chart ranked by test accuracy.

Usage: python -m src.plot_figure5_updated
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "output", "ablation", "results")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "ablation", "plots")

# Top 5 FFT models, best time-series, baseline (from ablation results)
ENTRIES = [
    {"label": "FFT Euclidean k=120 ensemble",   "file": "ablation_FFT_euclidean_k120_ensemble_report.json",  "domain": "FFT"},
    {"label": "FFT Euclidean k=80 ensemble",     "file": "ablation_FFT_euclidean_k80_ensemble_report.json",   "domain": "FFT"},
    {"label": "FFT Euclidean k=120 single",      "file": "ablation_FFT_euclidean_k120_single_report.json",    "domain": "FFT"},
    {"label": "FFT Euclidean k=60 ensemble",     "file": "ablation_FFT_euclidean_k60_ensemble_report.json",   "domain": "FFT"},
    {"label": "FFT Euclidean k=80 single",       "file": "ablation_FFT_euclidean_k80_single_report.json",     "domain": "FFT"},
    {"label": "Time Lorentzian k=120 ensemble",  "file": "ablation_time_lorentzian_k120_ensemble_report.json","domain": "time"},
    {"label": "Baseline (Random Forest)",        "file": "baseline_report.json",                              "domain": "baseline"},
]

COLORS = {'FFT': '#2ecc71', 'time': '#e74c3c', 'baseline': '#95a5a6'}


def load_accuracy(filename):
    with open(os.path.join(RESULTS_DIR, filename)) as f:
        data = json.load(f)
    return data['accuracy']


if __name__ == "__main__":
    # Load accuracies
    results = []
    for entry in ENTRIES:
        acc = load_accuracy(entry['file'])
        results.append({**entry, 'acc': acc})
        print(f"  {acc:.4f}  {entry['label']}")

    # Sort descending, then reverse for horizontal bar (highest on top)
    results.sort(key=lambda x: x['acc'], reverse=True)
    results.reverse()

    fig, ax = plt.subplots(figsize=(10, 5))

    y_pos = np.arange(len(results))
    bar_colors = [COLORS[r['domain']] for r in results]
    bars = ax.barh(y_pos, [r['acc'] for r in results], color=bar_colors, height=0.65)
    ax.bar_label(bars, fmt='%.4f', padding=5, fontsize=10)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([r['label'] for r in results], fontsize=10)
    ax.set_xlabel('Test Accuracy', fontsize=11)
    ax.set_title('Model Comparison: Top FFT, Best Time-Series, and Baseline', fontsize=12, fontweight='bold')
    ax.set_xlim(0.55, 1.0)
    ax.grid(axis='x', alpha=0.3)

    legend_elements = [
        Patch(facecolor=COLORS['FFT'], label='FFT Domain'),
        Patch(facecolor=COLORS['time'], label='Time Domain'),
        Patch(facecolor=COLORS['baseline'], label='Baseline'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "overall_ranking")
    fig.savefig(out_path + ".png", dpi=300, bbox_inches='tight')
    fig.savefig(out_path + ".pdf", bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_path}.png + .pdf")
