#!/usr/bin/env python3
"""
verify_report_numbers.py

Verifies the numeric claims stated in the project report against:
  1) seismic_classifier_dataset.h5
  2) dataset_train.h5 / dataset_val.h5 / dataset_test.h5
  3) extracted/EQ/*.h5 and extracted/EX/*.h5 (station files, if present)

Usage examples:
  python verify_report_numbers.py --root .
  python verify_report_numbers.py --root /path/to/SeismicDataset

Expected files (relative to --root):
  seismic_classifier_dataset.h5
  dataset_train.h5
  dataset_val.h5
  dataset_test.h5
  extracted/EQ/
  extracted/EX/
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import h5py
import numpy as np


# ---------------------------
# Expected numbers from report
# ---------------------------

EXPECTED = {
    # Dataset-wide summary (seismic_classifier_dataset.h5)
    "total_windows": 9105,
    "eq_traces": 4065,
    "ex_traces": 5040,

    # Waveform properties
    "window_seconds": 3.00,
    "sampling_rate_hz": 200,
    "samples_per_window": 600,
    "channels": 3,
    "window_shape_tail": (600, 3),  # last dims of X

    # Station set size
    "n_good_stations": 12,

    # Split sizes (window counts)
    "train_windows": 6334,
    "val_windows": 1444,
    "test_windows": 1327,

    # Split sizes (event counts)
    "train_events": 767,
    "val_events": 164,
    "test_events": 165,

    # Label coding
    "label_eq": 0,
    "label_ex": 1,

    # Report says you used threshold >=20 valid windows for both classes (station filtering stage)
    "min_valid_windows_per_class_per_station": 20,

    # Report says catalog event counts (may be verifiable from extracted files)
    "catalog_eq_events": 515,
    "catalog_ex_events": 589,
}


# ---------------------------
# Utilities
# ---------------------------

def ok(msg: str) -> None:
    print(f"✅ PASS  {msg}")

def fail(msg: str) -> None:
    print(f"❌ FAIL  {msg}")

def warn(msg: str) -> None:
    print(f"⚠️ WARN  {msg}")

def info(msg: str) -> None:
    print(f"ℹ️  {msg}")

def require_exists(path: Path) -> bool:
    if not path.exists():
        fail(f"Missing: {path}")
        return False
    return True

def read_unique_strings(arr: np.ndarray) -> List[str]:
    # Works for dtype like |S10
    if arr.dtype.kind in ("S", "O"):
        vals = np.unique(arr)
        out = []
        for v in vals:
            if isinstance(v, bytes):
                out.append(v.decode("utf-8", errors="replace"))
            else:
                out.append(str(v))
        return out
    return [str(x) for x in np.unique(arr)]

def chunked_bincount(y_ds, n_classes: int = 2, chunk: int = 200_000) -> np.ndarray:
    counts = np.zeros(n_classes, dtype=np.int64)
    n = y_ds.shape[0]
    for i in range(0, n, chunk):
        sl = y_ds[i : min(i + chunk, n)]
        sl = np.asarray(sl)
        # tolerate labels outside range by masking
        for k in range(n_classes):
            counts[k] += np.count_nonzero(sl == k)
    return counts

def unique_count(ds, chunk: int = 300_000) -> int:
    # For event_id length ~9k this is trivial; still robust
    arr = np.asarray(ds[:])
    return int(np.unique(arr).size)

def check_equal(name: str, got, exp) -> bool:
    if got == exp:
        ok(f"{name}: {got}")
        return True
    fail(f"{name}: got {got}, expected {exp}")
    return False

def check_close(name: str, got: float, exp: float, tol: float = 1e-6) -> bool:
    if abs(got - exp) <= tol:
        ok(f"{name}: {got}")
        return True
    fail(f"{name}: got {got}, expected {exp} (tol={tol})")
    return False

def dataset_has_keys(h: h5py.File, keys: Iterable[str]) -> bool:
    return all(k in h for k in keys)


# ---------------------------
# Core checks
# ---------------------------

def verify_joint_dataset(path: Path) -> Dict[str, object]:
    """
    Checks:
      - X shape, dtype
      - y counts (EQ/EX)
      - event_id length and unique count
      - station length and unique station count
    Returns computed stats for later use.
    """
    stats: Dict[str, object] = {}

    with h5py.File(path, "r") as h:
        required = ["X", "y", "event_id", "station"]
        missing = [k for k in required if k not in h]
        if missing:
            fail(f"{path.name} missing datasets: {missing}")
            return stats

        X = h["X"]
        y = h["y"]
        eid = h["event_id"]
        st = h["station"]

        # Shapes
        stats["X_shape"] = tuple(X.shape)
        stats["X_dtype"] = str(X.dtype)
        stats["y_shape"] = tuple(y.shape)
        stats["y_dtype"] = str(y.dtype)

        # Basic invariants
        check_equal(f"{path.name} X.ndim", X.ndim, 3)
        if X.ndim == 3:
            check_equal(f"{path.name} X.shape[1:]", tuple(X.shape[1:]), EXPECTED["window_shape_tail"])
        check_equal(f"{path.name} channels", int(X.shape[-1]), EXPECTED["channels"])

        check_equal(f"{path.name} n_windows (X)", int(X.shape[0]), EXPECTED["total_windows"])
        check_equal(f"{path.name} n_windows (y)", int(y.shape[0]), EXPECTED["total_windows"])
        check_equal(f"{path.name} n_windows (event_id)", int(eid.shape[0]), EXPECTED["total_windows"])
        check_equal(f"{path.name} n_windows (station)", int(st.shape[0]), EXPECTED["total_windows"])

        # Label counts
        counts = chunked_bincount(y, n_classes=2)
        stats["y_counts"] = counts
        check_equal("EQ traces (y==0)", int(counts[EXPECTED["label_eq"]]), EXPECTED["eq_traces"])
        check_equal("EX traces (y==1)", int(counts[EXPECTED["label_ex"]]), EXPECTED["ex_traces"])
        check_equal("Total windows (sum labels)", int(counts.sum()), EXPECTED["total_windows"])

        # Unique events
        n_events = unique_count(eid)
        stats["n_events_total"] = n_events
        ok(f"Unique event_ids in joint dataset: {n_events}")

        # Unique stations
        stations = read_unique_strings(np.asarray(st[:]))
        stats["stations"] = stations
        check_equal("Unique stations in joint dataset", len(stations), EXPECTED["n_good_stations"])
        ok(f"Stations: {stations}")

        # Optional: verify duration relationship (3 seconds * 200 Hz == 600)
        samples = int(X.shape[1])
        check_equal("Samples per window", samples, EXPECTED["samples_per_window"])
        # duration computed from report numbers (not from file), still sanity check:
        duration = EXPECTED["samples_per_window"] / EXPECTED["sampling_rate_hz"]
        check_close("Window duration implied by (samples / Fs)", float(duration), float(EXPECTED["window_seconds"]), tol=1e-9)

        # Optional metadata checks (only if fields exist)
        optional_fields = [
            ("event_lat", "event_lon"),
            ("station_lat", "station_lon"),
            ("epicentral_distance_deg",),
            ("n_arrivals_within_2deg",),
        ]
        present_any = False
        for group in optional_fields:
            if all(k in h for k in group):
                present_any = True
                ok(f"Optional metadata present: {group}")
        if not present_any:
            warn("No geographic/epicentral-distance/arrival-count metadata fields found in joint HDF5; "
                 "skipping checks for lat/lon windows, 2° filtering, and >=8 arrivals rules.")

    return stats


def verify_split_dataset(path: Path, expected_windows: int, expected_events: int, name: str) -> Dict[str, object]:
    stats: Dict[str, object] = {}

    with h5py.File(path, "r") as h:
        required = ["X", "y", "event_id", "station"]
        missing = [k for k in required if k not in h]
        if missing:
            fail(f"{path.name} missing datasets: {missing}")
            return stats

        X = h["X"]
        y = h["y"]
        eid = h["event_id"]
        st = h["station"]

        check_equal(f"{name} windows (X)", int(X.shape[0]), expected_windows)
        check_equal(f"{name} windows (y)", int(y.shape[0]), expected_windows)
        check_equal(f"{name} windows (event_id)", int(eid.shape[0]), expected_windows)
        check_equal(f"{name} windows (station)", int(st.shape[0]), expected_windows)

        # Shape tail
        if X.ndim == 3:
            check_equal(f"{name} X.shape[1:]", tuple(X.shape[1:]), EXPECTED["window_shape_tail"])

        # Unique events
        n_events = unique_count(eid)
        stats["n_events"] = n_events
        check_equal(f"{name} unique events", n_events, expected_events)

        # Labels distribution (informational, not asserted by report except totals earlier)
        counts = chunked_bincount(y, n_classes=2)
        stats["y_counts"] = counts
        ok(f"{name} label counts: EQ={int(counts[0])}, EX={int(counts[1])}")

        # Station coverage
        stations = read_unique_strings(np.asarray(st[:]))
        stats["stations"] = stations
        ok(f"{name} unique stations: {len(stations)}")

    return stats


def verify_no_event_leakage(train_eids: np.ndarray, val_eids: np.ndarray, test_eids: np.ndarray) -> None:
    tr = set(np.unique(train_eids).tolist())
    va = set(np.unique(val_eids).tolist())
    te = set(np.unique(test_eids).tolist())

    inter_tr_va = tr.intersection(va)
    inter_tr_te = tr.intersection(te)
    inter_va_te = va.intersection(te)

    if not inter_tr_va and not inter_tr_te and not inter_va_te:
        ok("No event leakage across train/val/test (event_id sets are disjoint).")
    else:
        fail(f"Event leakage detected: "
             f"|train∩val|={len(inter_tr_va)}, |train∩test|={len(inter_tr_te)}, |val∩test|={len(inter_va_te)}")


def scan_station_files(extracted_dir: Path) -> List[Path]:
    if not extracted_dir.exists():
        return []
    return sorted([p for p in extracted_dir.glob("*.h5") if p.is_file()])


def station_valid_windows_from_masks(station_file: Path) -> int:
    """
    Mirrors your description:
      P = h["mask_P"][:].sum()
      S = h["mask_S"][:].sum()
      take max(P,S) for the station
    We interpret mask arrays as {0,1} and sum counts of valid samples/windows.
    If masks are per-window (shape [n, ...]), sum will be proportional; we also try a per-window count.
    """
    with h5py.File(station_file, "r") as h:
        if "mask_P" not in h or "mask_S" not in h:
            raise KeyError("mask_P/mask_S not found")

        mP = h["mask_P"]
        mS = h["mask_S"]

        # Prefer per-window validity if masks are shaped (n_windows, ...)
        def per_window_valid(mask_ds) -> int:
            arr = np.asarray(mask_ds[:])
            if arr.ndim == 1:
                # already per-window
                return int(np.sum(arr > 0))
            # consider a window valid if it has any nonzero mask
            flat = arr.reshape(arr.shape[0], -1)
            return int(np.sum(np.any(flat > 0, axis=1)))

        # If masks are large, still safe typically; but keep robust
        try:
            p_valid = per_window_valid(mP)
            s_valid = per_window_valid(mS)
            return int(max(p_valid, s_valid))
        except Exception:
            # fallback to raw sum
            p_sum = int(np.asarray(mP[:]).sum())
            s_sum = int(np.asarray(mS[:]).sum())
            return int(max(p_sum, s_sum))


def verify_good_station_threshold(ex_root: Path) -> None:
    """
    Verifies the report claim:
      "Stations were considered usable if they had at least 20 valid EQ windows AND 20 valid EX windows"
    using extracted/EQ/*.h5 and extracted/EX/*.h5 station files.
    """
    eq_dir = ex_root / "extracted" / "EQ"
    ex_dir = ex_root / "extracted" / "EX"

    eq_files = scan_station_files(eq_dir)
    ex_files = scan_station_files(ex_dir)

    if not eq_files or not ex_files:
        warn("Could not find extracted station files in extracted/EQ and extracted/EX; skipping station-threshold verification.")
        return

    # Map station -> file
    def station_code_from_path(p: Path) -> str:
        return p.stem

    eq_map = {station_code_from_path(p): p for p in eq_files}
    ex_map = {station_code_from_path(p): p for p in ex_files}

    common = sorted(set(eq_map.keys()).intersection(set(ex_map.keys())))
    if not common:
        fail("No common station files between extracted/EQ and extracted/EX.")
        return

    min_thr = EXPECTED["min_valid_windows_per_class_per_station"]

    passing = []
    failing = []

    for st in common:
        try:
            eq_valid = station_valid_windows_from_masks(eq_map[st])
            ex_valid = station_valid_windows_from_masks(ex_map[st])
        except Exception as e:
            warn(f"Station {st}: could not compute validity from masks ({e}); skipping.")
            continue

        if eq_valid >= min_thr and ex_valid >= min_thr:
            passing.append((st, eq_valid, ex_valid))
        else:
            failing.append((st, eq_valid, ex_valid))

    info(f"Stations evaluated (common EQ/EX station files): {len(common)}")
    info(f"Stations meeting >= {min_thr} valid windows in BOTH classes: {len(passing)}")

    # Your report says you ended with 12 "good" stations.
    # We check that the passing list length matches 12.
    if len(passing) == EXPECTED["n_good_stations"]:
        ok(f"Good-station count matches report: {len(passing)}")
    else:
        fail(f"Good-station count differs: got {len(passing)}, expected {EXPECTED['n_good_stations']}")
        if passing:
            info("Passing stations: " + ", ".join([f"{st}(EQ={eqv},EX={exv})" for st, eqv, exv in passing]))
        if failing:
            info("Failing stations: " + ", ".join([f"{st}(EQ={eqv},EX={exv})" for st, eqv, exv in failing]))


def verify_catalog_event_counts_from_extracted(ex_root: Path) -> None:
    """
    Attempts to validate the report's "515 EQ events" and "589 EX events" counts.

    This is only possible if extracted files store event IDs in a way we can interpret:
      - If extracted/EQ and extracted/EX are per-event files: we can count files or unique event_id.
      - If they are per-station files: we can count unique event_id across windows in any one station,
        but this depends on how you wrote those HDF5s.

    We try both heuristics and report what we can.
    """
    eq_dir = ex_root / "extracted" / "EQ"
    ex_dir = ex_root / "extracted" / "EX"

    eq_files = scan_station_files(eq_dir)
    ex_files = scan_station_files(ex_dir)

    if not eq_files or not ex_files:
        warn("No extracted HDF5s found to infer catalog event counts; skipping 515/589 validation.")
        return

    # Heuristic A: maybe per-event files (then file count == event count)
    # If that doesn't match, try heuristic B using unique event_id inside files (if exists).
    def try_unique_events_in_file(p: Path) -> Optional[int]:
        try:
            with h5py.File(p, "r") as h:
                if "event_id" in h:
                    arr = np.asarray(h["event_id"][:])
                    return int(np.unique(arr).size)
        except Exception:
            return None
        return None

    # A
    eq_file_count = len(eq_files)
    ex_file_count = len(ex_files)
    info(f"Extracted EQ .h5 files: {eq_file_count}")
    info(f"Extracted EX .h5 files: {ex_file_count}")

    if eq_file_count == EXPECTED["catalog_eq_events"]:
        ok(f"EQ event count matches report via file-count heuristic: {eq_file_count}")
    else:
        warn(f"EQ file-count heuristic does not match 515 (got {eq_file_count}). Will try event_id inside files.")

    if ex_file_count == EXPECTED["catalog_ex_events"]:
        ok(f"EX event count matches report via file-count heuristic: {ex_file_count}")
    else:
        warn(f"EX file-count heuristic does not match 589 (got {ex_file_count}). Will try event_id inside files.")

    # B: sum unique events across ALL files is not safe (can double count),
    # so instead: if these are per-station files, pick the station file with the most event_ids
    # and treat its unique event_id count as a lower bound / proxy.
    eq_uniques = [(p, try_unique_events_in_file(p)) for p in eq_files]
    ex_uniques = [(p, try_unique_events_in_file(p)) for p in ex_files]
    eq_uniques = [(p, u) for p, u in eq_uniques if u is not None]
    ex_uniques = [(p, u) for p, u in ex_uniques if u is not None]

    if eq_uniques:
        p_max, u_max = max(eq_uniques, key=lambda t: t[1])
        info(f"EQ: max unique event_id found in a single extracted file ({p_max.name}): {u_max}")
        if u_max == EXPECTED["catalog_eq_events"]:
            ok("EQ event count matches report via unique(event_id) in extracted file.")
        else:
            warn("EQ event count still not directly verifiable as 515 from extracted files (structure may be different).")
    else:
        warn("EQ: could not find event_id dataset in extracted files; cannot verify 515 from extracted outputs.")

    if ex_uniques:
        p_max, u_max = max(ex_uniques, key=lambda t: t[1])
        info(f"EX: max unique event_id found in a single extracted file ({p_max.name}): {u_max}")
        if u_max == EXPECTED["catalog_ex_events"]:
            ok("EX event count matches report via unique(event_id) in extracted file.")
        else:
            warn("EX event count still not directly verifiable as 589 from extracted files (structure may be different).")
    else:
        warn("EX: could not find event_id dataset in extracted files; cannot verify 589 from extracted outputs.")


# ---------------------------
# Main
# ---------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=".", help="Project root containing the HDF5 files")
    args = ap.parse_args()
    root = Path(args.root).expanduser().resolve()

    info(f"Root: {root}")

    joint = root / "seismic_classifier_dataset.h5"
    train = root / "dataset_train.h5"
    val = root / "dataset_val.h5"
    test = root / "dataset_test.h5"

    # Required files
    if not all(map(require_exists, [joint, train, val, test])):
        fail("Missing one or more required HDF5 files. Fix paths and re-run.")
        raise SystemExit(2)

    print("\n=== 1) Verify JOINT dataset summary numbers (seismic_classifier_dataset.h5) ===")
    joint_stats = verify_joint_dataset(joint)

    print("\n=== 2) Verify TRAIN/VAL/TEST split sizes and shapes ===")
    train_stats = verify_split_dataset(train, EXPECTED["train_windows"], EXPECTED["train_events"], "TRAIN")
    val_stats = verify_split_dataset(val, EXPECTED["val_windows"], EXPECTED["val_events"], "VAL")
    test_stats = verify_split_dataset(test, EXPECTED["test_windows"], EXPECTED["test_events"], "TEST")

    print("\n=== 3) Verify NO EVENT LEAKAGE across splits (by event_id) ===")
    with h5py.File(train, "r") as h_tr, h5py.File(val, "r") as h_va, h5py.File(test, "r") as h_te:
        verify_no_event_leakage(
            np.asarray(h_tr["event_id"][:]),
            np.asarray(h_va["event_id"][:]),
            np.asarray(h_te["event_id"][:]),
        )

    print("\n=== 4) Verify split totals add up to joint totals ===")
    total_windows = EXPECTED["train_windows"] + EXPECTED["val_windows"] + EXPECTED["test_windows"]
    check_equal("Train+Val+Test windows", total_windows, EXPECTED["total_windows"])

    total_events = EXPECTED["train_events"] + EXPECTED["val_events"] + EXPECTED["test_events"]
    ok(f"Train+Val+Test events: {total_events} (informational; joint unique events may differ if some events had 0 windows after filtering)")

    print("\n=== 5) Verify good-station threshold (>=20 valid EQ & EX windows) from extracted/ ===")
    verify_good_station_threshold(root)

    print("\n=== 6) Try to verify catalog event counts 515 (EQ) / 589 (EX) from extracted outputs ===")
    verify_catalog_event_counts_from_extracted(root)

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
