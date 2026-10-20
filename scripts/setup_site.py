"""Set up a staff-plate camera once: find the plate, read its datum with OCR, save a site folder.

Usage:
  python scripts/setup_site.py --name hancock --cam NY_East_Branch_Delaware_River_at_Hancock_Staff \
      --site 01421500 --index data/hancock/images.csv --reference data/hancock/reference.jpg

Writes sites/<name>/site.json and sites/<name>/reference.jpg. No gauge data is used: the
scale comes from the plate's printed marks and the datum from its printed numbers.
"""
import argparse
import csv
import json
import os
import shutil
import sys

import cv2 as cv
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from stagesight import plate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--cam", required=True, help="NIMS camera id")
    ap.add_argument("--site", required=True, help="USGS site number (for comparison only)")
    ap.add_argument("--index", required=True, help="images.csv of archive frames for the datum vote")
    ap.add_argument("--reference", required=True, help="clear daylight frame with low water")
    ap.add_argument("--ocr-model", default="models/crnn_en.onnx")
    ap.add_argument("--action-stage", type=float, default=None, help="NWS action stage, ft")
    ap.add_argument("--flood-stage", type=float, default=None, help="NWS flood stage, ft")
    ap.add_argument("--out", default="sites")
    a = ap.parse_args()

    ref = cv.imread(a.reference)
    P = plate.find_plate(ref)
    if P is None:
        sys.exit("No staff plate found in the reference frame.")
    rows = [r for r in csv.DictReader(open(a.index)) if r["time_utc"][11:13] in ("15", "17", "19")]
    datums = []
    for r in rows[::max(1, len(rows) // 20)]:
        img = cv.imread(r["path"])
        Pi = plate.find_plate(img) if img is not None else None
        D = plate.read_datum(img, Pi, a.ocr_model) if Pi else None
        if D:
            datums.append(D["datum_ft"])
    if not datums:
        sys.exit("OCR could not read the plate numbers in any frame.")
    vals, counts = np.unique(datums, return_counts=True)
    datum = float(vals[counts.argmax()])

    out = os.path.join(a.out, a.name)
    os.makedirs(out, exist_ok=True)
    shutil.copy(a.reference, os.path.join(out, "reference.jpg"))
    site = {"name": a.name, "camera_id": a.cam, "site_id": a.site, "mode": "plate",
            "datum_ft": datum, "datum_votes": [int(counts.max()), len(datums)],
            "action_stage_ft": a.action_stage, "flood_stage_ft": a.flood_stage, "plate": P}
    json.dump(site, open(os.path.join(out, "site.json"), "w"), indent=1)
    print(f"{a.name}: plate with {len(P['bars_t'])} foot bars (fit {P['fit_rmse_ft']:.3f} ft), "
          f"datum {datum} ft ({counts.max()} of {len(datums)} OCR votes) -> {out}")


if __name__ == "__main__":
    main()
