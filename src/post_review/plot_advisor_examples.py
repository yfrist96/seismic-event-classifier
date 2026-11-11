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

Figure 6 of the paper is a per-channel comparison of the same pair. The top row shows
the spectra normalized to each channel's peak, and the bottom row the log10 EX/EQ
ratio. Over 0-60 Hz, the area outside the 1-20 Hz bandpass is shaded. A separate entry
point writes it and does not touch the Figure 2 files:
    plot_spectra_and_ratio_pair(123583, "OFRI", 399313, "ZFRI", fmax=60.0)
    -> output/decision_boundary_plots/spectra_and_ratio_by_channel.{pdf,png}
plot_spectral_comparison_pair(...) writes two earlier single-row candidates that were
not used, in "classifier" and "peak" scalings.
"""

import os
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

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


# --- Figure 6 candidates: per-channel spectral comparison of the same pair ---

PAPER_PLOT_DIR = os.path.join(PROJECT_ROOT, "output", "decision_boundary_plots")
FEATURE_EPS = 1e-6           # eps of the classifier's log transform (src/ablation.py, to_frequency_domain)
BANDPASS_HZ = (1.0, 20.0)    # bandpass applied to the windows upstream (README, "Waveform preprocessing")
SPECTRAL_COMPARISON_MODES = ("classifier", "peak")


def _log_spectra(X):
    """Per-channel log10(|rfft| + eps) of one (600, 3) window -> (301, 3).

    Same computation as the "freq" mode of _plot_two_row (Figure 2b).
    """
    return np.stack([np.log10(np.abs(np.fft.rfft(X[:, ch])) + FEATURE_EPS)
                     for ch in range(X.shape[1])], axis=1)


def spectral_comparison_curves(eq_X, ex_X, mode):
    """Return (freqs, eq_curve, ex_curve) with curves of shape (301, 3).

    mode "classifier": the classifier's feature (paper Eq. 1). Each window's
        301x3 log spectrum is flattened to 903 values, L2-normalized as one
        vector, then reshaped back into channels.
    mode "peak": per channel, log10 amplitude relative to that channel's own
        maximum (0 at the peak).
    """
    freqs = np.fft.rfftfreq(N_SAMPLES, d=1.0 / SAMPLE_RATE)
    curves = []
    for X in (eq_X, ex_X):
        S = _log_spectra(X)
        if mode == "classifier":
            v = S.reshape(-1)
            S = (v / np.linalg.norm(v)).reshape(S.shape)
        elif mode == "peak":
            S = S - S.max(axis=0, keepdims=True)
        else:
            raise ValueError(f"unknown mode: {mode}")
        curves.append(S)
    return freqs, curves[0], curves[1]


def plot_spectral_comparison(eq_X, ex_X, output_path, mode):
    """Write a 1x3 (Z, N, E) overlay of the EQ and EX spectra to output_path.{pdf,png}.

    The area between the curves is filled red where EX > EQ and blue where
    EQ > EX. Dashed lines mark the 1-20 Hz bandpass. The x-range is left on
    autoscale over 0 Hz to Nyquist, as in Figure 2b.
    """
    freqs, eq_S, ex_S = spectral_comparison_curves(eq_X, ex_X, mode)
    ylabel = {"classifier": "L2-normalized log$_{10}$ amplitude",
              "peak": "log$_{10}$ amplitude re. channel peak"}[mode]
    band_kw = dict(color="0.45", linewidth=0.6, linestyle="--")

    rc = {"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8.5,
          "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 8,
          "pdf.fonttype": 42}
    with plt.rc_context(rc):
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), sharey=True)
        for ch, ax in enumerate(axes):
            eq, ex = eq_S[:, ch], ex_S[:, ch]
            ax.fill_between(freqs, eq, ex, where=ex > eq, interpolate=True,
                            color=EX_COLOR, alpha=0.3, linewidth=0)
            ax.fill_between(freqs, eq, ex, where=eq > ex, interpolate=True,
                            color=EQ_COLOR, alpha=0.3, linewidth=0)
            ax.plot(freqs, eq, color=EQ_COLOR, linewidth=0.8)
            ax.plot(freqs, ex, color=EX_COLOR, linewidth=0.8)
            for f in BANDPASS_HZ:
                ax.axvline(f, zorder=0.5, **band_kw)
            ax.set_title(CHANNEL_NAMES[ch])
            ax.set_xlabel("Frequency (Hz)")
            ax.grid(alpha=0.3, linewidth=0.5)
        axes[0].set_ylabel(ylabel)

        handles = [
            Line2D([], [], color=EQ_COLOR, linewidth=1.2),
            Line2D([], [], color=EX_COLOR, linewidth=1.2),
            Patch(facecolor=EX_COLOR, alpha=0.3, linewidth=0),
            Patch(facecolor=EQ_COLOR, alpha=0.3, linewidth=0),
            Line2D([], [], **band_kw),
        ]
        labels = ["Earthquake (EQ)", "Explosion (EX)", "EX > EQ", "EQ > EX",
                  "1–20 Hz bandpass"]
        fig.tight_layout(w_pad=0.6)
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.98),
                   ncol=len(labels), frameon=False, handlelength=1.8,
                   columnspacing=1.4)
        fig.savefig(output_path + ".pdf", bbox_inches="tight")
        fig.savefig(output_path + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {output_path}.pdf + .png")


def plot_spectral_comparison_pair(eq_event_id, eq_station, ex_event_id, ex_station,
                                  eq_split="train", ex_split="train",
                                  output_dir=PAPER_PLOT_DIR):
    """Write spectral_comparison_by_channel_{classifier,peak}.{pdf,png} for the pair."""
    os.makedirs(output_dir, exist_ok=True)
    eq_X = load_best_window(eq_event_id, eq_station, eq_split)
    ex_X = load_best_window(ex_event_id, ex_station, ex_split)
    for mode in SPECTRAL_COMPARISON_MODES:
        plot_spectral_comparison(
            eq_X, ex_X, mode=mode,
            output_path=os.path.join(output_dir, f"spectral_comparison_by_channel_{mode}"))


def plot_spectra_and_ratio(eq_X, ex_X, output_path, fmax=60.0):
    """2x3 figure (columns Z, N, E) for the Figure 2 pair.

    Top row: EQ and EX log10 spectra, each relative to its own channel peak (0 = peak),
        showing spectral width (narrow EX spike vs wide EQ plateau).
    Bottom row: log10 spectral ratio EX/EQ of those peak-normalized spectra, i.e. the
        difference of the two top-row curves; filled red above 0 (EX relatively stronger)
        and blue below 0 (EQ relatively stronger).
    Frequencies outside the 1-20 Hz bandpass are shaded grey (filter stopband / noise floor).
    """
    freqs, eq_S, ex_S = spectral_comparison_curves(eq_X, ex_X, mode="peak")
    keep = freqs <= fmax
    f, eq_S, ex_S = freqs[keep], eq_S[keep], ex_S[keep]
    ratio = ex_S - eq_S

    rc = {"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8.5,
          "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 8,
          "pdf.fonttype": 42}
    with plt.rc_context(rc):
        fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.4), sharex=True, sharey="row")
        for ch in range(3):
            top, bot = axes[0, ch], axes[1, ch]
            for ax in (top, bot):
                ax.axvspan(0, BANDPASS_HZ[0], color="0.88", zorder=0, linewidth=0)
                ax.axvspan(BANDPASS_HZ[1], fmax, color="0.88", zorder=0, linewidth=0)
                ax.grid(alpha=0.3, linewidth=0.5)
                ax.set_xlim(0, fmax)
            eq, ex = eq_S[:, ch], ex_S[:, ch]
            top.fill_between(f, eq, ex, where=ex > eq, interpolate=True,
                             color=EX_COLOR, alpha=0.3, linewidth=0)
            top.fill_between(f, eq, ex, where=eq > ex, interpolate=True,
                             color=EQ_COLOR, alpha=0.3, linewidth=0)
            top.plot(f, eq_S[:, ch], color=EQ_COLOR, linewidth=0.8)
            top.plot(f, ex_S[:, ch], color=EX_COLOR, linewidth=0.8)
            top.set_title(CHANNEL_NAMES[ch])
            r = ratio[:, ch]
            bot.fill_between(f, 0, r, where=r > 0, interpolate=True,
                             color=EX_COLOR, alpha=0.45, linewidth=0)
            bot.fill_between(f, 0, r, where=r < 0, interpolate=True,
                             color=EQ_COLOR, alpha=0.45, linewidth=0)
            bot.plot(f, r, color="0.25", linewidth=0.6)
            bot.axhline(0, color="black", linewidth=0.6)
            bot.set_xlabel("Frequency (Hz)")
        axes[0, 0].set_ylabel("log$_{10}$ amplitude\nre. channel peak")
        axes[1, 0].set_ylabel("log$_{10}$ ratio EX/EQ")

        handles = [Line2D([], [], color=EQ_COLOR, linewidth=1.2),
                   Line2D([], [], color=EX_COLOR, linewidth=1.2),
                   Patch(facecolor=EX_COLOR, alpha=0.45, linewidth=0),
                   Patch(facecolor=EQ_COLOR, alpha=0.45, linewidth=0),
                   Patch(facecolor="0.88", linewidth=0)]
        labels = ["Earthquake (EQ)", "Explosion (EX)", "EX relatively stronger",
                  "EQ relatively stronger", "Outside 1–20 Hz bandpass"]
        fig.tight_layout(h_pad=0.6, w_pad=0.6)
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.985),
                   ncol=3, frameon=False, handlelength=1.8, columnspacing=1.4)
        fig.savefig(output_path + ".pdf", bbox_inches="tight")
        fig.savefig(output_path + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {output_path}.pdf + .png")


def plot_spectra_and_ratio_pair(eq_event_id, eq_station, ex_event_id, ex_station,
                                eq_split="train", ex_split="train",
                                output_dir=PAPER_PLOT_DIR, fmax=60.0):
    """Write spectra_and_ratio_by_channel.{pdf,png} for the pair (Figure 6 candidate)."""
    os.makedirs(output_dir, exist_ok=True)
    eq_X = load_best_window(eq_event_id, eq_station, eq_split)
    ex_X = load_best_window(ex_event_id, ex_station, ex_split)
    plot_spectra_and_ratio(eq_X, ex_X, fmax=fmax,
                           output_path=os.path.join(output_dir, "spectra_and_ratio_by_channel"))


if __name__ == "__main__":
    plot_example_pair(eq_event_id=123583, eq_station="OFRI",
                      ex_event_id=399313, ex_station="ZFRI")
