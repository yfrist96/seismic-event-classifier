"""
Plot Figure 2 / Figure 3 style example waveforms and spectra for a specified
earthquake / explosion pair, matching the layout used in the paper:
  - Two rows: earthquake on top (blue), explosion on bottom (red).
  - Three columns: Vertical (Z), North-South (N), East-West (E).
  - Figure 2 = 3-channel time-domain trace.
  - Figure 3 = 3-channel FFT log-magnitude spectrum.

Public entry point:
    plot_example_pair(eq_event_id, eq_station, ex_event_id, ex_station,
                      eq_split="train", ex_split="train", output_dir=...)

Running this module as __main__ produces the plots for the pair Ittai approved:
  EQ event 123583 @ OFRI, EX event 399313 @ ZFRI.

Usage: python -m src.post_review.plot_advisor_examples
"""

import os
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "example_waveforms_for_ittai")

SAMPLE_RATE = 200
N_SAMPLES = 600
CHANNEL_NAMES = ["Vertical (Z)", "North-South (N)", "East-West (E)"]
EQ_COLOR = "#3498db"
EX_COLOR = "#e74c3c"


def load_best_window(event_id, station, split):
    """Return the highest-SNR window for the given (event_id, station) in `split`."""
    path = os.path.join(DATA_DIR, f"dataset_{split}.h5")
    with h5py.File(path, "r") as f:
        eids = f["event_id"][:]
        stas = f["station"][:].astype(str)
        idxs = np.where((eids == event_id) & (stas == station))[0]
        if len(idxs) == 0:
            raise ValueError(f"No windows for event {event_id} station {station} in {split}")
        windows = f["X"][idxs]
    abs_Z = np.abs(windows[:, :, 0])
    snr = abs_Z.max(axis=1) / (np.median(abs_Z, axis=1) + 1e-12)
    return windows[int(np.argmax(snr))]


def _plot_two_row(top_X, bottom_X, mode, output_path):
    """Render a 2x3 figure with the EQ on top (blue) and EX on bottom (red).

    mode: "time" -> 3-channel time-domain trace
          "freq" -> 3-channel FFT log-magnitude spectrum
    """
    if mode == "time":
        x = np.arange(N_SAMPLES) / SAMPLE_RATE
        ylab = "Amplitude"
        xlab = "Time (s)"
        suptitle = "Example Waveforms: Earthquake vs Explosion"
        transform = lambda X, ch: X[:, ch]
    elif mode == "freq":
        x = np.fft.rfftfreq(N_SAMPLES, d=1.0 / SAMPLE_RATE)
        ylab = "Log Magnitude"
        xlab = "Frequency (Hz)"
        suptitle = "Example FFT Spectra: Earthquake vs Explosion"
        transform = lambda X, ch: np.log10(np.abs(np.fft.rfft(X[:, ch])) + 1e-6)
    else:
        raise ValueError(f"unknown mode: {mode}")

    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharey="row")
    for ch in range(3):
        axes[0, ch].plot(x, transform(top_X, ch), color=EQ_COLOR, linewidth=0.8)
        axes[0, ch].set_title(CHANNEL_NAMES[ch], fontsize=11)
        axes[0, ch].grid(alpha=0.3)
        if ch == 0:
            axes[0, ch].set_ylabel(f"Earthquake\n{ylab}")

        axes[1, ch].plot(x, transform(bottom_X, ch), color=EX_COLOR, linewidth=0.8)
        axes[1, ch].grid(alpha=0.3)
        axes[1, ch].set_xlabel(xlab)
        if ch == 0:
            axes[1, ch].set_ylabel(f"Explosion\n{ylab}")

    plt.suptitle(suptitle, fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_path + ".pdf", bbox_inches="tight")
    fig.savefig(output_path + ".png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Wrote {output_path}.pdf + .png")


def plot_example_pair(eq_event_id, eq_station, ex_event_id, ex_station,
                      eq_split="train", ex_split="train",
                      output_dir=DEFAULT_OUTPUT_DIR):
    """Generate example_waveforms and example_spectra for the given pair."""
    os.makedirs(output_dir, exist_ok=True)
    eq_X = load_best_window(eq_event_id, eq_station, eq_split)
    ex_X = load_best_window(ex_event_id, ex_station, ex_split)

    print(f"Earthquake: event {eq_event_id} @ {eq_station} "
          f"({eq_split}, peak Z = {np.max(np.abs(eq_X[:, 0])):.2f})")
    print(f"Explosion:  event {ex_event_id} @ {ex_station} "
          f"({ex_split}, peak Z = {np.max(np.abs(ex_X[:, 0])):.2f})")

    _plot_two_row(eq_X, ex_X, mode="time",
                  output_path=os.path.join(output_dir, "example_waveforms"))
    _plot_two_row(eq_X, ex_X, mode="freq",
                  output_path=os.path.join(output_dir, "example_spectra"))


if __name__ == "__main__":
    plot_example_pair(eq_event_id=123583, eq_station="OFRI",
                      ex_event_id=399313, ex_station="ZFRI")
