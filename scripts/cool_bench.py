"""Benchmark the StageSight reader with one OpenCV build (run once with COOL, once with pip OpenCV).

Usage:  python scripts/cool_bench.py --frames <dir of JPEGs> --out result.json [--repeats 3]

Records which OpenCV ran (version, file, build info), times the full plate reader per frame and
the OpenCV calls it is made of, and stores every stage it read so two builds can be compared for
identical output.
"""
import argparse
import faulthandler
import glob
import json
import os
import platform
import statistics
import sys
import time

import cv2 as cv
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from stagesight import plate  # noqa: E402
from stagesight.reader import SiteReader  # noqa: E402


def timed(fn, n: int) -> float:
    """Median seconds of n calls."""
    out = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t)
    return statistics.median(out)


def stage(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def main() -> None:
    faulthandler.enable()                      # a native crash prints the Python stack
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(a.frames, "*.jpg")))
    raw = [open(p, "rb").read() for p in paths]
    stage(f"cv2 {cv.__version__} at {cv.__file__}; {len(paths)} frames")
    site = SiteReader(os.path.join(ROOT, "sites", "hancock"))
    stage("site loaded")

    t = time.perf_counter()
    imgs = [cv.imdecode(np.frombuffer(b, np.uint8), cv.IMREAD_COLOR) for b in raw]
    decode_s = time.perf_counter() - t

    stage("decoded")
    stages, per_pass = None, []
    for k in range(a.repeats):
        stage(f"reader pass {k}")
        t = time.perf_counter()
        res = [site.read(im) for im in imgs]
        per_pass.append(time.perf_counter() - t)
        stages = [[os.path.basename(p), r["status"], r["stage_ft"]] for p, r in zip(paths, res)]

    im = imgs[len(imgs) // 2]
    gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    red = plate.red_mask(site.reference)
    x0, y0, x1, y1 = site.reg["box"]
    crop = cv.cvtColor(im[y0:y1, x0:x1], cv.COLOR_BGR2GRAY).astype(np.float32)
    ops = {
        "imdecode_1080x1920": lambda: cv.imdecode(np.frombuffer(raw[0], np.uint8), cv.IMREAD_COLOR),
        "cvtColor_BGR2Lab": lambda: cv.cvtColor(im, cv.COLOR_BGR2Lab),
        "cvtColor_BGR2HSV": lambda: cv.cvtColor(im, cv.COLOR_BGR2HSV),
        "Laplacian_var": lambda: cv.Laplacian(gray, cv.CV_64F).var(),
        "phaseCorrelate_plate": lambda: cv.phaseCorrelate(site.reg["ref"], crop, site.reg["win"]),
        "connectedComponentsWithStats": lambda: cv.connectedComponentsWithStats(red),
        "morphologyEx_open": lambda: cv.morphologyEx(red, cv.MORPH_OPEN, np.ones((3, 3), np.uint8)),
        "GaussianBlur_5x5": lambda: cv.GaussianBlur(im, (5, 5), 0),
        "resize_half": lambda: cv.resize(im, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA),
    }
    op_ms = {}
    for k, f in ops.items():
        stage(f"op {k}")
        op_ms[k] = round(timed(f, 50) * 1e3, 3)
    ocr_s = None

    info = cv.getBuildInformation()
    keep = [ln.strip() for ln in info.splitlines() if any(k in ln for k in
            ("General configuration", "Version control", "KleidiCV", "Parallel framework", "CPU/HW features",
             "Baseline", "Dispatched", "Extra modules", "Host:", "C++ flags (Release)"))]
    out = {"cv2_version": cv.__version__, "cv2_file": cv.__file__, "threads": cv.getNumThreads(),
           "optimized": cv.useOptimized(), "machine": platform.machine(), "python": platform.python_version(),
           "cpu_count": os.cpu_count(), "build_info": keep, "frames": len(imgs),
           "decode_s": round(decode_s, 3), "read_pass_s": [round(x, 3) for x in per_pass],
           "read_ms_per_frame": round(statistics.median(per_pass) / len(imgs) * 1e3, 2),
           "frames_per_s": round(len(imgs) / statistics.median(per_pass), 2),
           "ocr_datum_s": ocr_s, "op_ms": op_ms, "stages": stages}
    json.dump(out, open(a.out, "w"), indent=1)
    stage("OCR datum (DNN)")                    # last: if the DNN path fails, the timings above are kept
    out["ocr_datum_s"] = round(timed(lambda: plate.read_datum(site.reference, site.plate,
                                                             os.path.join(ROOT, "models", "crnn_en.onnx")), 3), 3)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("stages", "build_info")}, indent=1))


if __name__ == "__main__":
    main()
