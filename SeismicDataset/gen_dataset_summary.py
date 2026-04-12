import h5py
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import os
from collections import Counter

DATASET = "seismic_classifier_dataset.h5"
OUTDIR = pathlib.Path("summary_plots")
OUTDIR.mkdir(exist_ok=True)


def main():
    print("\n=== Loading dataset ===")
    with h5py.File(DATASET, "r") as f:
        X = f["X"][:]           # (N, 600, 3)
        y = f["y"][:]           # (N,)
        stations = f["station"][:].astype(str)
        event_ids = f["event_id"][:]

    N, T, C = X.shape

    print("\n=== BASIC SHAPE INFO ===")
    print(f"Total windows: {N}")
    print(f"Window shape: {T} samples, {C} channels")
    print(f"Sampling rate: 200 Hz")
    print(f"Duration: {T/200:.2f} seconds")

    # -------------------------------
    # 1. CLASS BALANCE
    # -------------------------------
    print("\n=== CLASS DISTRIBUTION ===")
    eq = np.sum(y == 0)
    ex = np.sum(y == 1)
    print(f"Earthquakes (0): {eq}")
    print(f"Explosions (1):  {ex}")

    # Plot class distribution
    plt.figure(figsize=(5, 4))
    plt.bar(["Earthquake", "Explosion"], [eq, ex])
    plt.title("Class Distribution")
    plt.savefig(OUTDIR / "class_distribution.png")
    plt.close()

    # -------------------------------
    # 2. PER-STATION COUNTS
    # -------------------------------
    print("\n=== PER-STATION WINDOW COUNTS ===")
    station_counts = Counter(stations)
    for sta, cnt in station_counts.most_common():
        print(f"{sta:5s}: {cnt}")

    # bar plot
    plt.figure(figsize=(10, 5))
    plt.bar(station_counts.keys(), station_counts.values())
    plt.xticks(rotation=45)
    plt.title("Windows per Station")
    plt.tight_layout()
    plt.savefig(OUTDIR / "station_distribution.png")
    plt.close()

    # -------------------------------
    # 3. AMPLITUDE STATISTICS
    # -------------------------------
    print("\n=== AMPLITUDE STATISTICS ===")
    print(f"Global mean amplitude: {X.mean():.4f}")
    print(f"Global std amplitude:  {X.std():.4f}")
    print(f"Min: {X.min():.2f}, Max: {X.max():.2f}")

    # histogram of amplitudes
    plt.figure(figsize=(7, 4))
    plt.hist(X.flatten(), bins=200, alpha=0.7)
    plt.title("Amplitude Histogram")
    plt.xlim(-1, 1)
    plt.savefig(OUTDIR / "amplitude_histogram.png")
    plt.close()

    # -------------------------------
    # 4. EXAMPLE WAVEFORMS
    # -------------------------------
    print("\n=== EXPORTING SAMPLE WAVEFORMS ===")
    sample_idxs = np.random.choice(N, size=5, replace=False)
    for i, idx in enumerate(sample_idxs):
        # time axis in seconds
        time = np.arange(T) / 200.0     # 200 Hz → 0.005s per sample

        plt.figure(figsize=(10, 4))
        plt.plot(time, X[idx, :, 0], label="Z")
        plt.plot(time, X[idx, :, 1], label="N")
        plt.plot(time, X[idx, :, 2], label="E")

        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude (normalized)")
        plt.title(f"Example window #{idx} (label={y[idx]}, station={stations[idx]})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTDIR / f"example_waveform_{i}.png")
        plt.close()


    # -------------------------------
    # 5. P vs S WINDOWS (optional)
    # -------------------------------
    # NOTE: If needed, we can tag P/S using arrival metadata, but not required now.

    # -------------------------------
    # 6. SUMMARY MARKDOWN FILE
    # -------------------------------
    print("\n=== Writing Markdown Summary ===")
    with open("DATASET_SUMMARY.md", "w") as f:
        f.write("# Seismic Classification Dataset Summary\n\n")
        f.write(f"- Total windows: **{N}**\n")
        f.write(f"- Earthquakes: **{eq}**\n")
        f.write(f"- Explosions: **{ex}**\n")
        f.write(f"- Window shape: **{T} × {C}**\n")
        f.write(f"- Duration: **{T/200:.2f} sec**\n")
        f.write(f"- Stations included ({len(station_counts)}):\n")
        for sta, cnt in station_counts.most_common():
            f.write(f"  - **{sta}**: {cnt} windows\n")
        f.write("\nPlots saved under `summary_plots/`\n")

    print("\n✅ Summary complete.")
    print("See: DATASET_SUMMARY.md")
    print("See plots in: summary_plots/")


if __name__ == "__main__":
    main()
