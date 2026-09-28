"""Selective download of the Camargo et al. (2021) data set (Mendeley Data, three parts) by HTTP
range reads, so that only the needed members of each archive are transferred.

Fetched: the sensors the analysis reads -- markers, fp (force plates: force and center of pressure),
conditions, and id (OpenSim inverse dynamics, used to check the planar reference) -- for the
treadmill, level-ground and ramp modes, into <config.CAMARGO>/raw/<subject>/<date>/<mode>/<sensor>/
(the layout camargo_to_csv.m expects); and the subject table SubjectInfo.mat into <config.CAMARGO>/.

The Mendeley file-listing API sits behind Cloudflare and rejects scripted requests, so the file ids
below were resolved once and are hard-coded (each download URL redirects to S3, which supports
range requests).
"""
import os
import sys
import time

import requests
from remotezip import RemoteZip
import config

OUT = os.path.join(config.CAMARGO, "raw")
LOG = os.path.join(config.CAMARGO, "download_log.txt")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/128.0 Safari/537.36"}
SENSORS = ("markers", "fp", "conditions", "id")
MODES = ("treadmill", "ramp", "levelground")

ARCHIVES = [
    ("Subjects_Part1_AB06-AB14.zip", "fcgm3chfff", "9c30f2e4-2d39-4d95-b691-1163c927455e"),
    ("AB15.zip", "k9kvm5tn3f", "789b7453-81fe-4d1b-87c1-df727ac4853a"),
    ("AB16.zip", "k9kvm5tn3f", "b554466b-48da-49e1-a94d-ded704817aab"),
    ("AB17.zip", "k9kvm5tn3f", "f31474c4-a718-4354-be12-3dcca6246ae6"),
    ("AB18.zip", "k9kvm5tn3f", "5e779990-e1e4-406d-97b5-b05ccb488494"),
    ("AB19.zip", "k9kvm5tn3f", "647b775c-23a2-4031-b890-b0cfe39875a1"),
    ("AB20.zip", "k9kvm5tn3f", "b4f97255-0721-4342-af38-3f390eac9a2e"),
    ("AB21.zip", "k9kvm5tn3f", "202ed926-688b-49e4-b6f1-36fe29c2b020"),
    ("AB23.zip", "k9kvm5tn3f", "422c1fc9-cda8-4a25-9654-4cc87c0de6ab"),
    ("AB24.zip", "k9kvm5tn3f", "be42627c-0c76-4562-8f2c-f70d2309820d"),
    ("AB25.zip", "k9kvm5tn3f", "5ed70f75-9d81-41ac-8e21-a58f93f2544e"),
    ("AB27.zip", "jj3r5f9pnf", "3f30407d-1690-4e7a-9b25-1ad58f9a2e21"),
    ("AB28.zip", "jj3r5f9pnf", "b2edc2c4-c6d7-42c8-afa2-980011d18622"),
    ("AB30.zip", "jj3r5f9pnf", "6a2213ca-f758-4f06-af45-ad97719df8d4"),
]

# the subject table (age, sex, height, mass), a separate file of part 1
INFO = ("SubjectInfo.mat", "fcgm3chfff", "d0f204d7-903c-4064-aff6-7080b2b5acb8")


def log(msg):
    with open(LOG, "a") as f:
        f.write(f"{time.ctime()} {msg}\n")
    print(msg, flush=True)


def resolve(ds, fid):
    """Mendeley blocks the python-requests fingerprint (Cloudflare), curl gets through."""
    import re
    import subprocess
    url = f"https://data.mendeley.com/public-files/datasets/{ds}/files/{fid}/file_downloaded"
    out = subprocess.run(["curl", "-sI", "-A", UA["User-Agent"], url], capture_output=True, text=True).stdout
    m = re.search(r"^location:\s*(\S+)", out, re.I | re.M)
    return m.group(1) if m else None


def wanted(name):
    p = name.split("/")
    # <subject>/<date>/<mode>/<sensor>/<trial>.mat
    return len(p) == 5 and p[2] in MODES and p[3] in SENSORS and name.endswith(".mat")


def fetch_archive(fname, ds, fid):
    s3 = resolve(ds, fid)
    if not s3:
        log(f"{fname}: could not resolve redirect")
        return
    with RemoteZip(s3, headers=UA) as z:
        names = [n for n in z.namelist() if wanted(n)]
        todo = [n for n in names if not os.path.exists(os.path.join(OUT, *n.split("/")))]
        log(f"{fname}: {len(names)} wanted, {len(todo)} missing")
        for i, n in enumerate(todo, 1):
            dst = os.path.join(OUT, *n.split("/"))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            try:
                data = z.read(n)
            except Exception as e:
                log(f"  error on {n}: {e!r}")
                continue
            with open(dst + ".part", "wb") as fh:
                fh.write(data)
            os.replace(dst + ".part", dst)
            if i % 50 == 0:
                log(f"{fname}: {i}/{len(todo)}")
    log(f"{fname}: done")


def fetch_info():
    name, ds, fid = INFO
    dst = os.path.join(config.CAMARGO, name)
    if os.path.exists(dst):
        return
    s3 = resolve(ds, fid)
    if not s3:
        log(f"{name}: could not resolve redirect")
        return
    r = requests.get(s3, headers=UA, timeout=60)
    r.raise_for_status()
    with open(dst, "wb") as f:
        f.write(r.content)
    log(f"{name}: {len(r.content)} bytes")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    only = sys.argv[1:] or None
    log("start")
    fetch_info()
    for fname, ds, fid in ARCHIVES:
        if only and not any(o in fname for o in only):
            continue
        for attempt in range(4):
            try:
                fetch_archive(fname, ds, fid)
                break
            except Exception as e:
                log(f"{fname}: attempt {attempt} failed: {e!r}")
                time.sleep(20)
    log("all done")
