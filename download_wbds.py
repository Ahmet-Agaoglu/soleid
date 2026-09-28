"""Fetch the treadmill trials of the Fukuchi et al. (2018) data set (figshare,
doi:10.6084/m9.figshare.5722711, version 6): the subject table WBDSinfo.csv and, for every
treadmill trial, the marker (mkr), force-plate (grf) and Visual3D joint-kinetics (knt) text files.
The trial files are read out of the 687 MB archive WBDSascii.zip by HTTP range requests, so the
archive itself is never downloaded; files already on disk are skipped. About 1.2 GB in total."""
import os
import time

import pandas as pd
import requests
from remotezip import RemoteZip
import config

DATA = config.WBDS
ZIP_URL = "https://api.figshare.com/v2/file/download/68194516"      # WBDSascii.zip
INFO_URL = "https://api.figshare.com/v2/file/download/68504641"     # WBDSinfo.csv
LOG = os.path.join(DATA, "download_log.txt")


def main():
    os.makedirs(DATA, exist_ok=True)
    info_path = os.path.join(DATA, "WBDSinfo.csv")
    if not os.path.exists(info_path):
        r = requests.get(INFO_URL, timeout=120)
        r.raise_for_status()
        with open(info_path, "wb") as fh:
            fh.write(r.content)
    info = pd.read_csv(info_path)
    tr = info[info.FileName.str.contains("walkT") & info.FileName.str.endswith("grf.txt")]
    need = []
    for fn in tr.FileName:
        base = fn[:-7]  # strip 'grf.txt'
        for k in ("mkr", "grf", "knt"):
            f = f"{base}{k}.txt"
            if not os.path.exists(os.path.join(DATA, f)):
                need.append(f)
    with open(LOG, "a") as log:
        log.write(f"{time.ctime()} start: {len(need)} files to fetch\n")
    if not need:
        return
    done = 0
    for attempt in range(5):
        try:
            with RemoteZip(ZIP_URL) as z:
                names = set(z.namelist())
                for f in need:
                    p = os.path.join(DATA, f)
                    if os.path.exists(p):
                        continue
                    member = "51subjs/" + f
                    if member not in names:
                        with open(LOG, "a") as log:
                            log.write(f"missing in zip: {f}\n")
                        continue
                    data = z.read(member)
                    with open(p + ".part", "wb") as fh:
                        fh.write(data)
                    os.replace(p + ".part", p)
                    done += 1
                    if done % 20 == 0:
                        with open(LOG, "a") as log:
                            log.write(f"{time.ctime()} fetched {done}/{len(need)}\n")
            break
        except Exception as e:  # network hiccup: reopen and continue
            with open(LOG, "a") as log:
                log.write(f"{time.ctime()} error (attempt {attempt}): {e!r}\n")
            time.sleep(10)
    with open(LOG, "a") as log:
        log.write(f"{time.ctime()} done: fetched {done}\n")


if __name__ == "__main__":
    main()
