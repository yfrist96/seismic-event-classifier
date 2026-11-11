"""
P-onset vs S-onset accuracy and source-station distance checks on the test set.

The dataset holds two kinds of 3-s windows: one aligned on the analyst P pick and one on
the analyst S pick (0.5 s before to 2.5 s after the pick in both cases). The HDF5 files do
not store the phase, but it is recoverable from window order:
SeismicDataset/build_training_dataset.py concatenates, for every (station, class) block,
the event-sorted P windows followed by the event-sorted S windows, and
SeismicDataset/split_by_event.py keeps that order. So inside each contiguous
(station, class) block, the point where event_id stops increasing starts the S block.

The script reports:
  1. Window- and event-level test accuracy on all, P-onset-only and S-onset-only windows.
     It uses the saved predictions in output/test_predictions.csv (written by
     src/post_review/export_test_predictions.py), so nothing is retrained. Event-level
     accuracy is a majority vote with ties going to earthquake, the same rule as
     src/post_review/event_level_predictions.py.
  2. Hypocentral source-station distance for every window. Event locations come from
     SeismicDataset/gsi_event_match.csv, which maps dataset event IDs to the public GSI
     catalog (seis.gsi.gov.il FDSN). Station coordinates come from
     SeismicDataset/station_metadata.csv.
  3. A distance-only baseline: one threshold, and one threshold per station, both fit on
     train windows.
  4. Ensemble accuracy by distance band, and on distance-matched subsets (equal numbers of
     earthquake and explosion windows in each band, averaged over repeated random draws).

Usage:
    python -m src.post_review.p_only_and_distance [--draws 200] [--seed 0]
Writes output/p_only_distance_results.json and data/phase_labels.csv.
"""

import argparse
import json
import os

import h5py
import numpy as np
import pandas as pd
from scipy.stats import mode

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
META_DIR = os.path.join(PROJECT_ROOT, "SeismicDataset")
SPLITS = ("train", "val", "test")
DIST_BINS_KM = [0, 40, 60, 80, 100, 130, 160, 250]
EARTH_RADIUS_KM = 6371.0


def phase_labels(split):
    """Return a DataFrame of window_index, event_id, station, y, phase for one split."""
    with h5py.File(os.path.join(DATA_DIR, f"dataset_{split}.h5"), "r") as f:
        ev = f["event_id"][:]
        st = f["station"][:].astype(str)
        y = f["y"][:]
    phase = np.empty(len(ev), dtype=object)
    i = 0
    while i < len(ev):
        j = i
        while j < len(ev) and st[j] == st[i] and y[j] == y[i]:
            j += 1
        drops = np.where(np.diff(ev[i:j]) < 0)[0]
        if len(drops) > 1:
            raise RuntimeError(f"Unexpected window order in {split}, block {st[i]}/{y[i]}")
        cut = drops[0] + 1 if len(drops) else j - i
        phase[i:i + cut] = "P"
        phase[i + cut:j] = "S"
        i = j
    df = pd.DataFrame({"split": split, "window_index": np.arange(len(ev)), "event_id": ev,
                       "station": st, "y": y, "phase": phase})
    # Sanity check: a station that recorded an event twice must have a P window then an S window.
    pairs = df.groupby(["event_id", "station"])["phase"].agg(list)
    pairs = pairs[pairs.map(len) == 2]
    if not pairs.map(lambda p: p == ["P", "S"]).all():
        raise RuntimeError(f"P/S order check failed in {split}")
    return df


def add_distance(df):
    ev = pd.read_csv(os.path.join(META_DIR, "gsi_event_match.csv"))
    sta = pd.read_csv(os.path.join(META_DIR, "station_metadata.csv"))
    df = df.merge(ev[["event_id", "lat", "lon", "depth_km"]], on="event_id", how="left")
    df = df.merge(sta[["station", "latitude", "longitude"]], on="station", how="left")
    la1, lo1, la2, lo2 = map(np.radians, [df.lat, df.lon, df.latitude, df.longitude])
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    epi = 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(h))
    df["D_km"] = np.sqrt(epi ** 2 + df.depth_km.fillna(0) ** 2)
    return df


def event_vote_accuracy(df, col):
    g = df.groupby("event_id").agg(true=("true_label", "first"),
                                   pred=(col, lambda p: int(mode(p, keepdims=False).mode)))
    return float((g.true == g.pred).mean()), int(len(g))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    res = {}

    windows = pd.concat([phase_labels(s) for s in SPLITS], ignore_index=True)
    windows[["split", "window_index", "event_id", "station", "y", "phase"]].to_csv(
        os.path.join(DATA_DIR, "phase_labels.csv"), index=False)
    res["composition"] = {f"{s}/{'EQ' if c == 0 else 'EX'}/{p}": int(n)
                          for (s, c, p), n in windows.groupby(["split", "y", "phase"]).size().items()}
    windows = add_distance(windows)

    test = pd.read_csv(os.path.join(OUTPUT_DIR, "test_predictions.csv"))
    test = test.merge(windows[windows.split == "test"][["window_index", "event_id", "station", "phase", "D_km"]],
                      on=["window_index", "event_id", "station"], how="left", validate="1:1")

    # 1. P-onset vs S-onset accuracy
    res["phase_accuracy"] = {}
    for model, col in [("single", "single_pred_label"), ("ensemble", "ensemble_pred_label"),
                       ("calibrated", "calibrated_pred_label")]:
        r = {}
        for name, sub in [("all", test), ("P", test[test.phase == "P"]), ("S", test[test.phase == "S"])]:
            ev_acc, n_ev = event_vote_accuracy(sub, col)
            r[name] = {"window_acc": float((sub[col] == sub.true_label).mean()), "n_windows": int(len(sub)),
                       "event_acc": ev_acc, "n_events": n_ev}
        res["phase_accuracy"][model] = r

    # 2-3. distance and distance-only baselines (located windows only)
    train = windows[(windows.split == "train") & windows.D_km.notna()]
    loc = test[test.D_km.notna()].copy()
    res["distance_km_quartiles"] = {
        nm: {("EQ" if c == 0 else "EX"): [float(x) for x in g.D_km.quantile([.25, .5, .75])]
             for c, g in df.groupby(key)}
        for nm, df, key in [("train", train, "y"), ("test", loc, "true_label")]}
    grid = np.arange(10, 250, 0.5)

    def best_threshold(d):
        accs = [((d.D_km > t).astype(int) == d.y).mean() for t in grid]
        return float(grid[int(np.argmax(accs))])

    t_all = best_threshold(train)
    t_sta = {s: best_threshold(g) for s, g in train.groupby("station")}
    res["distance_only_baseline"] = {
        "threshold_km": t_all,
        "test_acc_single_threshold": float(((loc.D_km > t_all).astype(int) == loc.true_label).mean()),
        "test_acc_per_station_threshold": float(((loc.D_km > loc.station.map(t_sta)).astype(int)
                                                 == loc.true_label).mean()),
        "ensemble_test_acc_same_windows": float((loc.ensemble_pred_label == loc.true_label).mean()),
        "n_test_windows": int(len(loc)),
    }

    # 4. accuracy by distance band and on distance-matched subsets
    loc["band"] = pd.cut(loc.D_km, DIST_BINS_KM)
    loc["ok"] = loc.ensemble_pred_label == loc.true_label
    rng = np.random.default_rng(args.seed)
    res["by_band"], res["distance_matched"] = {}, {}
    for name, sub in [("all", loc), ("P", loc[loc.phase == "P"])]:
        g = sub.groupby(["band", "true_label"], observed=True).ok.agg(["mean", "size"])
        res["by_band"][name] = {f"{b}/{'EQ' if c == 0 else 'EX'}": {"acc": float(r["mean"]), "n": int(r["size"])}
                                for (b, c), r in g.iterrows()}
        accs, base = [], []
        for _ in range(args.draws):
            keep = []
            for _, gb in sub.groupby("band", observed=True):
                eq, ex = gb[gb.true_label == 0], gb[gb.true_label == 1]
                k = min(len(eq), len(ex))
                if k:
                    keep += list(rng.choice(eq.index, k, replace=False)) + list(rng.choice(ex.index, k, replace=False))
            m = sub.loc[keep]
            accs.append(m.ok.mean())
            base.append(((m.D_km > t_all).astype(int) == m.true_label).mean())
        res["distance_matched"][name] = {
            "ensemble_acc_mean": float(np.mean(accs)),
            "ensemble_acc_p2.5": float(np.percentile(accs, 2.5)),
            "ensemble_acc_p97.5": float(np.percentile(accs, 97.5)),
            "distance_only_acc_mean": float(np.mean(base)),
            "n_windows_per_draw": int(len(keep)), "draws": args.draws}

    out = os.path.join(OUTPUT_DIR, "p_only_distance_results.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    pa = res["phase_accuracy"]["ensemble"]
    print(f"Ensemble window acc: all {pa['all']['window_acc']:.4f} | P {pa['P']['window_acc']:.4f} | S {pa['S']['window_acc']:.4f}")
    print(f"Ensemble event acc:  all {pa['all']['event_acc']:.4f} | P {pa['P']['event_acc']:.4f} | S {pa['S']['event_acc']:.4f}")
    b = res["distance_only_baseline"]
    print(f"Distance-only: single threshold {b['test_acc_single_threshold']:.4f} ({b['threshold_km']} km), "
          f"per-station {b['test_acc_per_station_threshold']:.4f}; ensemble {b['ensemble_test_acc_same_windows']:.4f}")
    for nm, d in res["distance_matched"].items():
        print(f"Distance-matched ({nm}): ensemble {d['ensemble_acc_mean']:.4f} "
              f"[{d['ensemble_acc_p2.5']:.4f}, {d['ensemble_acc_p97.5']:.4f}], distance-only {d['distance_only_acc_mean']:.4f}")
    print(f"Wrote {out} and data/phase_labels.csv")


if __name__ == "__main__":
    main()
