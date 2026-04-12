import glob
import h5py
import pandas as pd
from collections import defaultdict

EQ_DIR = "extracted/EQ"
EX_DIR = "extracted/EX"

def read_valid_counts(directory):
    counts = {}
    for f in sorted(glob.glob(f"{directory}/*.h5")):
        name = f.split("/")[-1]
        station = name.split(".")[1]
        with h5py.File(f, "r") as h:
            P = h["mask_P"][:].sum()
            S = h["mask_S"][:].sum()
        counts[station] = int(max(P, S))
    return counts

eq_valid = read_valid_counts(EQ_DIR)
ex_valid = read_valid_counts(EX_DIR)

good_stations = []
print("=== Stations usable for EQ vs EX classification ===")
for sta in sorted(eq_valid.keys()):
    eqv = eq_valid.get(sta, 0)
    exv = ex_valid.get(sta, 0)
    
    if eqv >= 20 and exv >= 20:
        good_stations.append(sta)
        print(f"{sta:5s}   EQ={eqv:4d}   EX={exv:4d}")

print("\nFinal usable station count:", len(good_stations))
print("Usable stations:", good_stations)
