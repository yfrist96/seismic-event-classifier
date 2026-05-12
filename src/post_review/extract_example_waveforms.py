"""
Extract 3 example waveforms and their FFT spectra for Ittai.
Outputs labeled 1-3 so they can be matched to updated Figures 2 & 3.
All figures saved as both PNG (300 DPI) and PDF (vector).

Usage: python -m src.extract_example_waveforms
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.dataloader import load_data

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "example_waveforms_for_ittai")

CHANNEL_NAMES = ['Vertical (Z)', 'North-South (N)', 'East-West (E)']
SAMPLE_RATE = 200
N_SAMPLES = 600


def save_figure(fig, path_without_ext):
    """Save figure as both PNG (300 DPI) and PDF (vector)."""
    fig.savefig(path_without_ext + ".png", dpi=300, bbox_inches='tight')
    fig.savefig(path_without_ext + ".pdf", bbox_inches='tight')


def select_examples(X, y, seed=42):
    """Select 3 diverse examples: 2 earthquakes, 1 explosion."""
    np.random.seed(seed)
    eq_indices = np.where(y == 0)[0]
    ex_indices = np.where(y == 1)[0]

    examples = [
        (np.random.choice(eq_indices), 'Earthquake'),
        (np.random.choice(ex_indices), 'Explosion'),
        (np.random.choice(eq_indices), 'Earthquake'),
    ]
    return examples


def plot_waveform_and_spectrum(X_sample, label, example_num, save_dir):
    """Plot 3-channel time-domain waveform and FFT spectrum side by side."""
    t = np.arange(N_SAMPLES) / SAMPLE_RATE
    freqs = np.fft.rfftfreq(N_SAMPLES, d=1.0 / SAMPLE_RATE)
    channel_colors = ['#2ecc71', '#3498db', '#e74c3c']  # Z=green, N=blue, E=red

    fig, axes = plt.subplots(3, 2, figsize=(16, 8))

    for ch in range(3):
        # Time domain (left column)
        axes[ch, 0].plot(t, X_sample[:, ch], color=channel_colors[ch], linewidth=0.8)
        axes[ch, 0].set_ylabel(f'{CHANNEL_NAMES[ch]}\nAmplitude')
        axes[ch, 0].grid(alpha=0.3)
        if ch == 0:
            axes[ch, 0].set_title('Time Domain', fontsize=12)
        if ch == 2:
            axes[ch, 0].set_xlabel('Time (s)')

        # FFT spectrum (right column)
        fft_mag = np.abs(np.fft.rfft(X_sample[:, ch]))
        log_mag = np.log10(fft_mag + 1e-6)
        axes[ch, 1].plot(freqs, log_mag, color=channel_colors[ch], linewidth=0.8)
        axes[ch, 1].set_ylabel('Log Magnitude')
        axes[ch, 1].grid(alpha=0.3)
        if ch == 0:
            axes[ch, 1].set_title('FFT Spectrum', fontsize=12)
        if ch == 2:
            axes[ch, 1].set_xlabel('Frequency (Hz)')

    fig.suptitle(f'Example {example_num} — {label}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_figure(fig, os.path.join(save_dir, f"example_{example_num}_{label.lower()}"))
    plt.close()
    print(f"  Saved: example_{example_num}_{label.lower()}.png + .pdf")


def save_raw_data(X_sample, label, example_num, save_dir):
    """Save raw waveform data as CSV for Ittai to re-plot if needed."""
    t = np.arange(N_SAMPLES) / SAMPLE_RATE
    freqs = np.fft.rfftfreq(N_SAMPLES, d=1.0 / SAMPLE_RATE)

    # Time-domain CSV
    header = "time_s,Z,N,E"
    data = np.column_stack([t, X_sample[:, 0], X_sample[:, 1], X_sample[:, 2]])
    path = os.path.join(save_dir, f"waveform_{example_num}_{label.lower()}.csv")
    np.savetxt(path, data, delimiter=',', header=header, comments='')

    # Frequency-domain CSV
    header = "freq_hz,Z_log_mag,N_log_mag,E_log_mag"
    fft_data = []
    for ch in range(3):
        fft_mag = np.abs(np.fft.rfft(X_sample[:, ch]))
        fft_data.append(np.log10(fft_mag + 1e-6))
    data = np.column_stack([freqs] + fft_data)
    path = os.path.join(save_dir, f"spectrum_{example_num}_{label.lower()}.csv")
    np.savetxt(path, data, delimiter=',', header=header, comments='')

    print(f"  Saved CSV data for example {example_num}")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading data...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data(DATA_DIR)
    X_all = np.vstack([X_train, X_val])
    y_all = np.concatenate([y_train, y_val])

    examples = select_examples(X_all, y_all)

    print(f"\nExtracting {len(examples)} examples...")
    for i, (idx, label) in enumerate(examples, start=1):
        print(f"\nExample {i}: index={idx}, label={label}")
        X_sample = X_all[idx]
        plot_waveform_and_spectrum(X_sample, label, i, OUTPUT_DIR)
        save_raw_data(X_sample, label, i, OUTPUT_DIR)

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
