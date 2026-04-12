import h5py
import numpy as np
import pathlib

EQ_DIR = pathlib.Path("extracted/EQ")
EX_DIR = pathlib.Path("extracted/EX")

def summarize_station(path):
    print(f"\n===== {path.name} =====")
    with h5py.File(path, "r") as f:
        P = f["P"]
        S = f["S"]
        mask_P = f["mask_P"][:]
        mask_S = f["mask_S"][:]
        event_ids = f["event_id"][:]

        print(f"Total events in file: {len(event_ids)}")
        print(f"P-valid windows: {mask_P.sum()}")
        print(f"S-valid windows: {mask_S.sum()}")
        print(f"P shape: {P.shape}")
        print(f"S shape: {S.shape}")

        # Inspect one valid P window if available
        if mask_P.sum() > 0:
            idx = np.where(mask_P)[0][0]
            w = P[idx]
            print(f"\nExample P window stats (event {event_ids[idx]}):")
            print(f"  mean={w.mean():.4f}, std={w.std():.4f}, min={w.min():.2f}, max={w.max():.2f}")
        else:
            print("\nNo valid P windows found.")

        if mask_S.sum() > 0:
            idx = np.where(mask_S)[0][0]
            w = S[idx]
            print(f"\nExample S window stats (event {event_ids[idx]}):")
            print(f"  mean={w.mean():.4f}, std={w.std():.4f}, min={w.min():.2f}, max={w.max():.2f}")
        else:
            print("\nNo valid S windows found.")

def main():
    print("\n=== EQ extraction overview ===")
    for f in sorted(EQ_DIR.iterdir()):
        summarize_station(f)

    print("\n=== EX extraction overview ===")
    for f in sorted(EX_DIR.iterdir()):
        summarize_station(f)

if __name__ == "__main__":
    main()
