import h5py
import numpy as np
import pathlib

EQ_DIR = pathlib.Path("extracted/EQ")
EX_DIR = pathlib.Path("extracted/EX")

GOOD_STATIONS = [
    "BLGI", "EIL", "GEM", "HNTI", "HRFI", "KZIT",
    "MBRI", "MSBI", "NATI", "OFRI", "SLTI", "ZFRI"
]

OUTPUT = "seismic_classifier_dataset.h5"

def load_station(path, label):
    """Load P/S windows from a single station file."""
    with h5py.File(path, "r") as f:
        P = f["P"][:]    # shape (n, 600, 3)
        S = f["S"][:] 
        mP = f["mask_P"][:]
        mS = f["mask_S"][:]
        eid = f["event_id"][:]

    # Only keep valid windows
    Xp = P[mP]
    Xs = S[mS]
    yp = np.full(len(Xp), label)
    ys = np.full(len(Xs), label)
    e_p = eid[mP]
    e_s = eid[mS]

    # Concatenate P + S
    X = np.concatenate([Xp, Xs])
    y = np.concatenate([yp, ys])
    e = np.concatenate([e_p, e_s])

    return X, y, e


def main():
    all_X = []
    all_y = []
    all_sta = []
    all_eid = []

    for sta in GOOD_STATIONS:
        eq_file = EQ_DIR / f"IS.{sta}.PS.h5"
        ex_file = EX_DIR / f"IS.{sta}.PS.h5"

        # EQ → label 0
        if eq_file.exists():
            Xeq, yeq, eid_eq = load_station(eq_file, 0)
            all_X.append(Xeq)
            all_y.append(yeq)
            all_sta.append(np.array([sta]*len(yeq)))
            all_eid.append(eid_eq)

        # EX → label 1
        if ex_file.exists():
            Xex, yex, eid_ex = load_station(ex_file, 1)
            all_X.append(Xex)
            all_y.append(yex)
            all_sta.append(np.array([sta]*len(yex)))
            all_eid.append(eid_ex)

    # Merge arrays
    X = np.concatenate(all_X)
    y = np.concatenate(all_y)
    sta = np.concatenate(all_sta)
    eid = np.concatenate(all_eid)

    print("Final dataset sizes:")
    print("Waveforms:", X.shape)
    print("Labels:   ", y.shape)
    print("Stations: ", sta.shape)
    print("Event IDs:", eid.shape)

    # Save to HDF5
    with h5py.File(OUTPUT, "w") as f:
        f.create_dataset("X", data=X)
        f.create_dataset("y", data=y)
        f.create_dataset("station", data=np.array(sta, dtype="S10"))
        f.create_dataset("event_id", data=eid)
        f.attrs["sampling_rate"] = 200
        f.attrs["window_length"] = 600

    print(f"\nSaved final dataset → {OUTPUT}")


if __name__ == "__main__":
    main()
