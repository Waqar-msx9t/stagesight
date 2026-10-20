"""Archive backfill throughput: read every frame of a folder with N worker processes.

Each worker decodes its frames and runs the full plate reader with one OpenCV thread, as a batch
re-evaluation of a camera archive would. Run once per OpenCV build (COOL, pip) on the same machine.

Usage:  python scripts/cool_scale.py --frames <dir> --out scale.json [--workers 1 4 8 16]
"""
import argparse
import glob
import json
import os
import sys
import time
from multiprocessing import get_context

import cv2 as cv
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
_site = None


def _init() -> None:
    global _site
    cv.setNumThreads(1)                         # parallelism comes from the processes
    from stagesight.reader import SiteReader
    _site = SiteReader(os.path.join(ROOT, "sites", "hancock"))


def _read(path: str):
    img = cv.imdecode(np.fromfile(path, np.uint8), cv.IMREAD_COLOR)
    r = _site.read(img) if img is not None else {"status": "undecodable", "stage_ft": None}
    return os.path.basename(path), r["status"], r["stage_ft"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 4, 8, 16])
    a = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(a.frames, "*.jpg")))
    runs, stages = [], None
    for n in a.workers:
        with get_context("fork" if sys.platform != "win32" else "spawn").Pool(n, initializer=_init) as pool:
            pool.map(_read, paths[: n * 2])     # warm-up: workers load the site
            t = time.perf_counter()
            res = pool.map(_read, paths, chunksize=8)
            dt = time.perf_counter() - t
        stages = stages or res
        runs.append({"workers": n, "seconds": round(dt, 2), "frames_per_s": round(len(paths) / dt, 1),
                     "same_stages_as_first": res == stages})
        print(runs[-1], flush=True)
    counts = {}
    for _, s, _v in stages:
        counts[s] = counts.get(s, 0) + 1
    json.dump({"cv2_version": cv.__version__, "cv2_file": cv.__file__, "cpu_count": os.cpu_count(),
               "frames": len(paths), "status_counts": counts, "runs": runs, "stages": stages},
              open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
