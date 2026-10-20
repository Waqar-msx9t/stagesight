"""Download NIMS camera frames and USGS gage height for one site.

Usage:
  python scripts/download_site.py --cam MO_Missouri_River_at_Hermann --site 06934500 --days 90
Writes data/<cam>/frames/*.jpg, data/images.csv (path,time_utc) and data/gauge.csv (time_utc,stage_ft).
"""
import argparse
import csv
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from stagesight import usgs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", required=True)
    ap.add_argument("--site", required=True)
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--size", default="overlayDir", help="overlayDir (full) or smallDir (720 px)")
    ap.add_argument("--out", default="data")
    a = ap.parse_args()

    end = np.datetime64("now", "s")
    start = end - np.timedelta64(a.days, "D")
    frame_dir = os.path.join(a.out, a.cam, "frames")
    os.makedirs(frame_dir, exist_ok=True)

    urls = usgs.list_images(a.cam, after=str(start), size=a.size)
    # Keep at most one frame per hour (the first one in each hour).
    by_hour = {}
    for u in urls:
        t = usgs.image_time(u)
        by_hour.setdefault(t.astype("datetime64[h]"), (t, u))
    picks = sorted(by_hour.values())
    print(f"{len(urls)} files listed, {len(picks)} hourly frames")

    def fetch(item):
        t, u = item
        dest = os.path.join(frame_dir, u.rsplit("/", 1)[1])
        if not os.path.exists(dest):
            try:
                usgs.download(u, dest)
            except Exception as exc:  # one bad file should not stop the run
                print("failed", u, exc)
                return None
        return dest, str(t)

    with ThreadPoolExecutor(8) as pool:
        rows = [r for r in pool.map(fetch, picks) if r]
    with open(os.path.join(a.out, "images.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "time_utc"])
        w.writerows(rows)

    gt, gh = usgs.gage_height(a.site, str(start.astype("datetime64[D]")),
                              str((end + np.timedelta64(1, "D")).astype("datetime64[D]")))
    with open(os.path.join(a.out, "gauge.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_utc", "stage_ft"])
        w.writerows(zip(map(str, gt), gh))
    print(f"{len(rows)} frames, {len(gt)} gauge values, stage {gh.min():.2f} to {gh.max():.2f} ft")


if __name__ == "__main__":
    main()
