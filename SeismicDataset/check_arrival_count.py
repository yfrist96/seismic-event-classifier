import pandas as pd

catalog = "DataMineClassification/REF4CAT_2018-2019_eq_sRECT_ED2_ndf8_nc0"
#catalog = "DataMineClassification/REF4CAT_2019250-365_ex_sRECT_ED2_ndf8_nc0"

ARRIVAL_COLUMNS = [
    "sta", "time", "arid", "jdate", "stassid", "chanid", "chan", "iphase",
    "stype", "deltim", "azimuth", "delaz", "slow", "delslo", "ema", "rect",
    "amp", "per", "logat", "clip", "fm", "sur", "qual", "auth", "commid",
    "lddate"
]

arr = pd.read_csv(
    catalog + ".arrival",
    sep=r"\s+",
    header=None,
    names=ARRIVAL_COLUMNS
)

print(arr["sta"].value_counts())
