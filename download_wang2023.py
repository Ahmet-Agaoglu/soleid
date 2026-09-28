"""Resumable download of the Wang et al. 2023 dataset (Zenodo 6457662): Processed_data.rar (17.9 GB)
first, then Raw_data.rar (13.1 GB). HTTP Range resume, retries, MD5 check at the end."""
import hashlib
import os
import time

import requests
import config

OUT = config.WANG
FILES = [
    ("Processed_data.rar", "https://zenodo.org/api/records/6457662/files/Processed_data.rar/content",
     17916719552, "c52802227f3aaf93551122d047232361"),
    # Raw_data.rar (13.1 GB, md5 6b042eed1d4e218d0799d5378637c61a) deliberately not fetched for now
]
LOG = os.path.join(OUT, "download_log.txt")
CHUNK = 8 * 1024 * 1024


def log(msg):
    with open(LOG, "a") as f:
        f.write(f"{time.ctime()} {msg}\n")


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(CHUNK), b""):
            h.update(b)
    return h.hexdigest()


def fetch(name, url, size, checksum):
    final = os.path.join(OUT, name)
    part = final + ".part"
    if os.path.exists(final) and os.path.getsize(final) == size:
        log(f"{name}: already complete")
        return True
    attempt = 0
    while attempt < 200:
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if have >= size:
            break
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=(30, 120)) as r:
                if r.status_code not in (200, 206):
                    log(f"{name}: HTTP {r.status_code}, retrying")
                    attempt += 1
                    time.sleep(min(60, 5 * attempt))
                    continue
                if have and r.status_code == 200:   # server ignored Range: restart
                    have = 0
                mode = "ab" if have else "wb"
                t0, got = time.time(), 0
                with open(part, mode) as f:
                    for chunk in r.iter_content(CHUNK):
                        f.write(chunk)
                        got += len(chunk)
                        if got % (512 * 1024 * 1024) < CHUNK:
                            done = have + got
                            rate = got / max(time.time() - t0, 1e-6) / 1e6
                            log(f"{name}: {done / 1e9:.2f} / {size / 1e9:.2f} GB  ({rate:.1f} MB/s)")
        except Exception as e:
            attempt += 1
            log(f"{name}: error {e!r} (attempt {attempt}); resuming")
            time.sleep(min(60, 5 * attempt))
    if os.path.getsize(part) != size:
        log(f"{name}: FAILED size {os.path.getsize(part)} != {size}")
        return False
    log(f"{name}: download complete, verifying MD5")
    h = md5(part)
    if h != checksum:
        log(f"{name}: MD5 MISMATCH {h} != {checksum}")
        return False
    os.replace(part, final)
    log(f"{name}: OK, MD5 verified")
    return True


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    log("start")
    for name, url, size, checksum in FILES:
        fetch(name, url, size, checksum)
    log("all done")
