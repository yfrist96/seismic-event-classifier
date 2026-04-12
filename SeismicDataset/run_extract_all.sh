#!/usr/bin/env bash
set -euo pipefail

# ---- CONFIG ----
EQ_DB="/Users/ik_seislab/Accounts/YFRIST//SeismicDataset/DataMineClassification/REF4CAT_2018-2019_eq_sRECT_ED2_ndf8_nc0"
EX_DB="/Users/ik_seislab/Accounts/YFRIST//SeismicDataset/DataMineClassification/REF4CAT_2019250-365_ex_sRECT_ED2_ndf8_nc0"
NETWORK="IS"
WF_ROOT="/Users/ik_seislab/Accounts/ittaik/ReprocCat_2013-2019/WF/wfs_ydns"
EXTRACTOR="/Users/ik_seislab/Accounts/YFRIST/SeismicDataset/400_extract_event_wfs_antelope.py"

PROJECT_ROOT="/Users/ik_seislab/Accounts/YFRIST/SeismicDataset"
STATION_LIST="$PROJECT_ROOT/station_list.txt"
OUT_EQ="$PROJECT_ROOT/extracted/EQ"
OUT_EX="$PROJECT_ROOT/extracted/EX"

mkdir -p "$OUT_EQ" "$OUT_EX"

# ---- ACTIVATE ENV ----
# adjust if your env name/path is different
source ~/miniforge3/etc/profile.d/conda.sh
conda activate seis

while read -r STATION; do
    [[ -z "$STATION" ]] && continue
    echo "=== Station: $STATION ==="

    echo "[EQ] Extracting..."
    python "$EXTRACTOR" \
        -f ANTELOPE \
        "$EQ_DB" \
        "$NETWORK" \
        "$STATION" \
        "$WF_ROOT" \
        "$OUT_EQ"

    echo "[EX] Extracting..."
    python "$EXTRACTOR" \
        -f ANTELOPE \
        "$EX_DB" \
        "$NETWORK" \
        "$STATION" \
        "$WF_ROOT" \
        "$OUT_EX"

done < "$STATION_LIST"

echo "All stations done."
