"""
Inventory every window in the dataset with per-window characteristics that help
pick visually-clean candidates for Figure 2 (example waveforms).

Outputs:
  output/example_waveforms_for_ittai/window_inventory.csv     -- one row per window
  output/example_waveforms_for_ittai/event_inventory.csv      -- one row per event
  output/example_waveforms_for_ittai/top_candidates.csv       -- curated top-15 per class

Usage: python -m src.post_review.list_candidate_waveforms
"""

import os
import numpy as np
import pandas as pd
import h5py

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "example_waveforms_for_ittai")

SAMPLE_RATE = 200  # Hz, per SeismicDataset/metadata.json
N_SAMPLES = 600    # 3.0 s window

LABEL_MAP = {0: "earthquake", 1: "explosion"}


def load_split(name):
    path = os.path.join(DATA_DIR, f"dataset_{name}.h5")
    with h5py.File(path, "r") as f:
        X = f["X"][:]
        y = f["y"][:]
        event_id = f["event_id"][:]
        station = f["station"][:].astype(str)
    split = np.full(len(X), name)
    return X, y, event_id, station, split


def compute_stats(X):
    freqs = np.fft.rfftfreq(N_SAMPLES, d=1.0 / SAMPLE_RATE)
    low = (freqs >= 0.0) & (freqs < 2.0)
    mid = (freqs >= 2.0) & (freqs < 10.0)
    high = (freqs >= 10.0) & (freqs <= SAMPLE_RATE / 2.0)

    Z = X[:, :, 0]
    NS = X[:, :, 1]
    EW = X[:, :, 2]

    abs_Z = np.abs(Z)
    peak_Z = abs_Z.max(axis=1)
    peak_t = abs_Z.argmax(axis=1) / SAMPLE_RATE
    snr = peak_Z / (np.median(abs_Z, axis=1) + 1e-12)

    fft_mag = np.abs(np.fft.rfft(Z, axis=1))
    power = fft_mag ** 2
    total = power.sum(axis=1) + 1e-20

    peak_freq = freqs[fft_mag.argmax(axis=1)]
    centroid = (freqs[None, :] * power).sum(axis=1) / total
    energy_low = power[:, low].sum(axis=1) / total
    energy_mid = power[:, mid].sum(axis=1) / total
    energy_high = power[:, high].sum(axis=1) / total

    return pd.DataFrame({
        "peak_abs_Z": peak_Z,
        "peak_abs_N": np.abs(NS).max(axis=1),
        "peak_abs_E": np.abs(EW).max(axis=1),
        "peak_time_Z_s": peak_t,
        "snr_proxy_Z": snr,
        "peak_freq_Z_hz": peak_freq,
        "spectral_centroid_hz": centroid,
        "energy_0_2hz": energy_low,
        "energy_2_10hz": energy_mid,
        "energy_10_100hz": energy_high,
    })


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    X_all, y_all, eid_all, sta_all, split_all = [], [], [], [], []
    for s in ["train", "val", "test"]:
        X, y, eid, sta, sp = load_split(s)
        X_all.append(X); y_all.append(y); eid_all.append(eid)
        sta_all.append(sta); split_all.append(sp)
        print(f"  loaded {s}: {len(X)} windows")

    X = np.vstack(X_all)
    y = np.concatenate(y_all)
    eid = np.concatenate(eid_all)
    sta = np.concatenate(sta_all)
    split = np.concatenate(split_all)
    print(f"Total: {len(X)} windows")

    stats = compute_stats(X)
    stats.insert(0, "window_index", np.arange(len(X)))
    stats.insert(1, "event_id", eid)
    stats.insert(2, "station", sta)
    stats.insert(3, "class", [LABEL_MAP[v] for v in y])
    stats.insert(4, "split", split)

    win_path = os.path.join(OUTPUT_DIR, "window_inventory.csv")
    stats.to_csv(win_path, index=False, float_format="%.4f")
    print(f"\nWrote {win_path} ({len(stats)} rows)")

    # Event-level inventory: count windows/stations per event
    ev = (stats.groupby(["event_id", "class", "split"])
                .agg(n_windows=("window_index", "size"),
                     n_stations=("station", "nunique"),
                     stations=("station", lambda s: ",".join(sorted(set(s)))),
                     mean_snr_Z=("snr_proxy_Z", "mean"),
                     max_snr_Z=("snr_proxy_Z", "max"),
                     mean_peak_freq_hz=("peak_freq_Z_hz", "mean"),
                     mean_centroid_hz=("spectral_centroid_hz", "mean"),
                     mean_high_energy=("energy_10_100hz", "mean"))
                .reset_index()
                .sort_values(["class", "mean_snr_Z"], ascending=[True, False]))
    ev_path = os.path.join(OUTPUT_DIR, "event_inventory.csv")
    ev.to_csv(ev_path, index=False, float_format="%.4f")
    print(f"Wrote {ev_path} ({len(ev)} events)")

    # Curated top candidates: high SNR, peak in middle of window (avoid edge artifacts),
    # split balanced across train/val/test, distinct stations
    candidates = []
    for cls in ["earthquake", "explosion"]:
        sub = stats[(stats["class"] == cls)
                    & (stats["peak_time_Z_s"].between(0.4, 2.6))
                    & (stats["snr_proxy_Z"] > stats["snr_proxy_Z"].quantile(0.85))].copy()
        sub = sub.sort_values("snr_proxy_Z", ascending=False)
        # Drop duplicate (event_id, station) — only need one window per recording
        sub = sub.drop_duplicates(subset=["event_id", "station"]).head(15)
        candidates.append(sub)
    top = pd.concat(candidates, ignore_index=True)
    top_path = os.path.join(OUTPUT_DIR, "top_candidates.csv")
    cols = ["class", "event_id", "station", "split", "snr_proxy_Z",
            "peak_time_Z_s", "peak_freq_Z_hz", "spectral_centroid_hz",
            "energy_0_2hz", "energy_2_10hz", "energy_10_100hz"]
    top[cols].to_csv(top_path, index=False, float_format="%.4f")
    print(f"Wrote {top_path} ({len(top)} curated rows)")


if __name__ == "__main__":
    main()
