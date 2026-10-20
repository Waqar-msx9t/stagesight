"""Download NIMS camera frames and USGS gage height for one site.

Usage (the Hancock data behind the report):
  python scripts/download_site.py --cam NY_East_Branch_Delaware_River_at_Hancock_Staff --site 01421500 \\
      --start 2026-07-21 --end 2026-10-20 --upstream 01417500 --out data/hancock
Writes <out>/<cam>/frames/*.jpg, <out>/images.csv (path,time_utc), <out>/gauge.csv (time_utc,stage_ft)
and, with --upstream, <out>/upstream_<site>.csv.
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
    ap.add_argument("--days", type=int, default=90, help="days back from now (ignored with --start)")
    ap.add_argument("--start", help="first day, e.g. 2026-07-21 (fixed window, reproducible)")
    ap.add_argument("--end", help="last day (default now)")
    ap.add_argument("--upstream", help="also save this USGS site's gage height (agent tool)")
    ap.add_argument("--size", default="overlayDir", help="overlayDir (full) or smallDir (720 px)")
    ap.add_argument("--out", default="data")
    a = ap.parse_args()

    end = np.datetime64(a.end, "s") if a.end else np.datetime64("now", "s")
    start = np.datetime64(a.start, "s") if a.start else end - np.timedelta64(a.days, "D")
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

    span = (str(start.astype("datetime64[D]")), str((end + np.timedelta64(1, "D")).astype("datetime64[D]")))
    gt, gh = usgs.gage_height(a.site, *span)
    for site, name in [(a.site, "gauge.csv")] + ([(a.upstream, f"upstream_{a.upstream}.csv")] if a.upstream else []):
        t, v = (gt, gh) if site == a.site else usgs.gage_height(site, *span)
        with open(os.path.join(a.out, name), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time_utc", "stage_ft"])
            w.writerows(zip(map(str, t), v))
    print(f"{len(rows)} frames, {len(gt)} gauge values, stage {gh.min():.2f} to {gh.max():.2f} ft")


if __name__ == "__main__":
    main()
