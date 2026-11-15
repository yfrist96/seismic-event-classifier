"""
Time the distance evaluations behind the paper's computational-cost claim (Section 6.4).

Compares, on real test-set windows:
  * Euclidean distance between two 903-dimensional log-spectral feature vectors
    (src.distances.euclidean_distance on src.ablation.to_frequency_domain output),
    the distance FastMap uses in the selected configuration;
  * normalized cross-correlation over all lags between two three-component
    600-sample waveforms (src.distances.ncc_distance), the distance of the
    original FastMapSVM.

It reports the median time per call over many random window pairs, and the
end-to-end FastMap fit and transform times with each distance on a subset, because
FastMap's own Python bookkeeping is shared by both distances and dilutes the
per-call ratio.

Usage (from the repository root):
    python -m src.post_review.time_distances [--pairs 300] [--repeats 200]
        [--fit-n 1000] [--fit-k 10] [--seed 0] [--out output/timing/distance_timing.json]

Writes only the JSON file given by --out. Run it on an otherwise idle machine.
"""

import argparse
import contextlib
import io
import json
import os
import platform
import time

import numpy as np
import scipy
import sklearn

from src.ablation import to_frequency_domain
from src.dataloader import load_split
from src.distances import euclidean_distance, ncc_distance
from src.fastmap import FastMap

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def per_call_seconds(func, a, b, repeats):
    """Mean time of one call of func(a, b), averaged over `repeats` back-to-back calls."""
    func(a, b)  # warm-up
    t0 = time.perf_counter()
    for _ in range(repeats):
        func(a, b)
    return (time.perf_counter() - t0) / repeats


def time_pairs(func, data, pairs, repeats):
    """Median and interquartile range of the per-call time over random pairs (microseconds)."""
    times = np.array([per_call_seconds(func, data[i], data[j], repeats) for i, j in pairs]) * 1e6
    return {"median_us": float(np.median(times)),
            "p25_us": float(np.percentile(times, 25)),
            "p75_us": float(np.percentile(times, 75)),
            "n_pairs": len(pairs), "calls_per_pair": repeats}


def time_fastmap(data, dist, k, n_transform):
    """Wall-clock FastMap fit on `data` and per-object transform time, with a given distance."""
    fm = FastMap(n_components=k, dist_func=dist)
    with contextlib.redirect_stdout(io.StringIO()):          # FastMap prints per dimension
        t0 = time.perf_counter()
        fm.fit_transform(data)
        t_fit = time.perf_counter() - t0
        t0 = time.perf_counter()
        fm.transform(data[:n_transform])
        t_tr = (time.perf_counter() - t0) / n_transform
    return {"fit_s": t_fit, "transform_ms_per_object": 1e3 * t_tr}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data-dir", default=os.path.join(PROJECT_ROOT, "data"))
    p.add_argument("--pairs", type=int, default=300, help="random window pairs to time")
    p.add_argument("--repeats", type=int, default=200, help="back-to-back calls per pair")
    p.add_argument("--fit-n", type=int, default=1000, help="windows for the FastMap fit comparison")
    p.add_argument("--fit-k", type=int, default=10, help="FastMap dimensions for the fit comparison")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=os.path.join(PROJECT_ROOT, "output", "timing", "distance_timing.json"))
    args = p.parse_args()

    X, _, _ = load_split(os.path.join(args.data_dir, "dataset_test.h5"))
    X = X.astype(np.float64)
    F = to_frequency_domain(X)                               # (N, 903) log-spectral vectors
    rng = np.random.default_rng(args.seed)
    pairs = [tuple(rng.choice(len(X), 2, replace=False)) for _ in range(args.pairs)]

    print(f"Timing {args.pairs} random pairs x {args.repeats} calls each ...")
    eu = time_pairs(euclidean_distance, F, pairs, args.repeats)
    ncc = time_pairs(ncc_distance, X, pairs, max(1, args.repeats // 10))
    print(f"  Euclidean, 903-dim log-spectra : median {eu['median_us']:.2f} us "
          f"(IQR {eu['p25_us']:.2f}-{eu['p75_us']:.2f})")
    print(f"  NCC, 600 x 3 waveforms         : median {ncc['median_us']:.1f} us "
          f"(IQR {ncc['p25_us']:.1f}-{ncc['p75_us']:.1f})")
    ratio = ncc["median_us"] / eu["median_us"]
    print(f"  per-call ratio NCC / Euclidean : {ratio:.0f}x")

    idx = rng.choice(len(X), min(args.fit_n, len(X)), replace=False)
    np.random.seed(args.seed)
    fm_eu = time_fastmap(F[idx], "euclidean", args.fit_k, n_transform=200)
    np.random.seed(args.seed)
    fm_ncc = time_fastmap(X[idx].reshape(len(idx), -1), "ncc", args.fit_k, n_transform=200)
    print(f"\nFastMap on {len(idx)} windows, k={args.fit_k}:")
    print(f"  Euclidean: fit {fm_eu['fit_s']:.2f} s, transform {fm_eu['transform_ms_per_object']:.3f} ms/window")
    print(f"  NCC      : fit {fm_ncc['fit_s']:.2f} s, transform {fm_ncc['transform_ms_per_object']:.3f} ms/window")
    print(f"  fit ratio NCC / Euclidean: {fm_ncc['fit_s'] / fm_eu['fit_s']:.0f}x")

    result = {
        "euclidean_logspectra": eu, "ncc_waveforms": ncc, "per_call_ratio": ratio,
        "fastmap": {"n": int(len(idx)), "k": args.fit_k, "euclidean": fm_eu, "ncc": fm_ncc,
                    "fit_ratio": fm_ncc["fit_s"] / fm_eu["fit_s"]},
        "environment": {"machine": platform.machine(), "processor": platform.processor(),
                        "platform": platform.platform(), "python": platform.python_version(),
                        "numpy": np.__version__, "scipy": scipy.__version__,
                        "scikit_learn": sklearn.__version__},
        "args": vars(args),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
