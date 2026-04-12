import h5py
import numpy as np
from sklearn.model_selection import train_test_split

INPUT = "seismic_classifier_dataset.h5"

# Output files
TRAIN_OUT = "dataset_train.h5"
VAL_OUT   = "dataset_val.h5"
TEST_OUT  = "dataset_test.h5"

# --------------------
# Load dataset
# --------------------
with h5py.File(INPUT, "r") as f:
    X = f["X"][:]
    y = f["y"][:]
    station = f["station"][:]
    event_id = f["event_id"][:]

# --------------------
# Split by EVENT IDs
# --------------------
unique_events = np.unique(event_id)

# Train/Val/Test = 70/15/15 (adjustable)
events_train, events_temp = train_test_split(
    unique_events, test_size=0.30, random_state=42, shuffle=True
)
events_val, events_test = train_test_split(
    events_temp, test_size=0.50, random_state=42, shuffle=True
)

def select_indices(events_subset):
    """Return indices corresponding to the given event IDs."""
    return np.isin(event_id, events_subset)

idx_train = select_indices(events_train)
idx_val   = select_indices(events_val)
idx_test  = select_indices(events_test)

# --------------------
# Helper function
# --------------------
def save_split(path, X_sel, y_sel, station_sel, event_sel):
    with h5py.File(path, "w") as f:
        f.create_dataset("X", data=X_sel, compression="gzip")
        f.create_dataset("y", data=y_sel, compression="gzip")
        f.create_dataset("station", data=station_sel, compression="gzip")
        f.create_dataset("event_id", data=event_sel, compression="gzip")

# --------------------
# Save splits
# --------------------
save_split(TRAIN_OUT, X[idx_train], y[idx_train], station[idx_train], event_id[idx_train])
save_split(VAL_OUT,   X[idx_val],   y[idx_val],   station[idx_val],   event_id[idx_val])
save_split(TEST_OUT,  X[idx_test],  y[idx_test],  station[idx_test],  event_id[idx_test])

# --------------------
# Summary prints
# --------------------
print("=== SPLIT COMPLETE ===")
print(f"Train windows: {idx_train.sum()}")
print(f"Val   windows: {idx_val.sum()}")
print(f"Test  windows: {idx_test.sum()}")
print()
print(f"Train events: {len(events_train)}")
print(f"Val   events: {len(events_val)}")
print(f"Test  events: {len(events_test)}")
