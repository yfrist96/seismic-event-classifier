import h5py
import pathlib

# Check BOTH EQ and EX extracted waveform folders
H5_DIRS = [
    pathlib.Path("extracted/EQ"),
    pathlib.Path("extracted/EX")
]

for h5_dir in H5_DIRS:
    print(f"\n=== Checking directory: {h5_dir} ===")

    if not h5_dir.exists():
        print(f"Directory not found: {h5_dir}")
        continue

    h5_files = sorted(h5_dir.glob("*.h5"))
    if not h5_files:
        print("No HDF5 files found.")
        continue

    for fpath in h5_files:
        try:
            with h5py.File(fpath, "r") as f:
                # In case masks are missing in some corrupted file
                P = int(f["mask_P"][:].sum()) if "mask_P" in f else 0
                S = int(f["mask_S"][:].sum()) if "mask_S" in f else 0
        except Exception as e:
            print(f"ERROR reading {fpath.name}: {e}")
            continue

        print(f"{fpath.name:20s}  P_valid={P:4d}   S_valid={S:4d}")
