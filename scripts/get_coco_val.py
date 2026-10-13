"""Download COCO val2017 (5000 images, ~780 MB) + annotations (~250 MB) into datasets/coco.

  python scripts/get_coco_val.py
"""

import http.client
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "datasets" / "coco"
FILES = {
    "val2017.zip": "https://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip",
    "annotations_trainval2017.zip": "https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip",
}


def download(url, path, retries=20):
    """Resumable download: on a dropped connection, retry from where it stopped (HTTP Range)."""
    if path.exists():
        print(f"have {path.name}")
        return
    tmp = path.with_name(path.name + ".part")
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=30) as r:
        total = int(r.headers["Content-Length"])

    for attempt in range(1, retries + 1):
        done = tmp.stat().st_size if tmp.exists() else 0
        if done >= total:
            break
        req = urllib.request.Request(url, headers={"Range": f"bytes={done}-"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "ab") as f:
                if done and r.status != 206:  # server ignored Range: start over
                    f.truncate(0)
                    done = 0
                while chunk := r.read(1 << 20):
                    f.write(chunk)
                    done += len(chunk)
                    print(f"\r{path.name}: {done / 1e6:5.0f} / {total / 1e6:.0f} MB", end="", flush=True)
        except (OSError, http.client.HTTPException) as e:
            wait = min(30, 2 * attempt)
            print(f"\n  connection dropped ({e}); resuming in {wait}s [{attempt}/{retries}]")
            time.sleep(wait)
    print()

    if tmp.stat().st_size != total:
        raise SystemExit(f"{path.name}: incomplete after {retries} attempts, run the script again to resume")
    tmp.rename(path)


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        download(url, DEST / name)

    if not (DEST / "val2017").is_dir():
        print("extracting images...")
        with zipfile.ZipFile(DEST / "val2017.zip") as z:
            z.extractall(DEST)
    ann = DEST / "annotations" / "instances_val2017.json"
    if not ann.exists():
        print("extracting annotations...")
        ann.parent.mkdir(exist_ok=True)
        with zipfile.ZipFile(DEST / "annotations_trainval2017.zip") as z, \
                z.open("annotations/instances_val2017.json") as src, open(ann, "wb") as dst:
            shutil.copyfileobj(src, dst)

    n = len(list((DEST / "val2017").glob("*.jpg")))
    print(f"{n} images, annotations at {ann}")
    print("zips can be deleted to save space:", *[DEST / k for k in FILES])


if __name__ == "__main__":
    main()
