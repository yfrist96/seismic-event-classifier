import glob

ROOT = "/Users/ik_seislab/Accounts/ittaik/ReprocCat_2013-2019/WF/wfs_ydns"

# Top stations from your arrival analysis
stations = [
    "HMDT","PRNI","MMA0B","DSI","YTIR","KRMI","KSHI","GEM","NATI","MBRI",
    "EIL","HRFI","MSBI","ZFRI","HNTI","OFRI","BLGI","SLTI","KZIT","MMC7"
]

print("\n=== Checking SDS waveform availability ===\n")

for sta in stations:
    # SDS structure: YEAR/JDAY/IS/STATION/*.mseed
    pattern = f"{ROOT}/**/IS/{sta}/*.mseed"
    files = glob.glob(pattern, recursive=True)
    print(f"{sta:5s} : {len(files)} files found")

print("\nDone.\n")

