#!/usr/bin/env bash

# Usage:
#   ./h5_structure_json.sh dataset_train.h5

FILE="$1"

if [ -z "$FILE" ]; then
    echo "Usage: $0 <hdf5-file>"
    exit 1
fi

python3 <<PYTHON
import h5py, json, sys, numpy as np

path = "$FILE"

def describe_item(name, obj):
    """Return JSON-serializable description of an HDF5 object."""
    if isinstance(obj, h5py.Dataset):
        return {
            "type": "dataset",
            "dtype": str(obj.dtype),
            "shape": list(obj.shape)
        }
    elif isinstance(obj, h5py.Group):
        return {"type": "group"}
    return {"type": "unknown"}

with h5py.File(path, "r") as f:
    structure = {}

    def walk(name, obj):
        parts = name.split("/")
        node = structure
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = describe_item(name, obj)

    f.visititems(walk)

print(json.dumps(structure, indent=2))
PYTHON
