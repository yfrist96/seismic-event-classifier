import glob
import os
import re

ROOT = "/Users/ik_seislab/Accounts/ittaik/ReprocCat_2013-2019/WF/wfs_ydns"

files = glob.glob(f"{ROOT}/**/*.mseed", recursive=True)

channels = {}

# Filename pattern example:
# IS.AFK.21.ENE__20190515T000000Z__20190515T235959Z.mseed
pattern = re.compile(r"^[A-Z0-9]+\.[A-Z0-9]+\.[A-Z0-9]+\.(?P<chan>[A-Z0-9]+)__")

for f in files:
    base = os.path.basename(f)
    m = pattern.match(base)
    if m:
        chan = m.group("chan")
        channels[chan] = channels.get(chan, 0) + 1

print("\n=== CHANNEL OCCURRENCES ===")
if not channels:
    print("No channels detected — pattern may still be wrong.")
else:
    for c, cnt in sorted(channels.items(), key=lambda x: -x[1]):
        print(f"{c:6s} : {cnt}")
