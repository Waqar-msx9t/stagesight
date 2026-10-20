"""Staff-plate mode flood test: plate graduations give the scale, the gauge gives one offset.

Usage:
  python scripts/plate_test.py --cache results/hancock_stack.npz --reference data/hancock/reference.jpg \
      --test-start 2026-07-21 --test-end 2026-08-01 --out results/hancock_plate
"""
import argparse
import csv
import json
import os
import sys

import cv2 as cv
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from stagesight import calibrate, plate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="stack .npz from flood_stress_test (times, stage, paths)")
    ap.add_argument("--reference", required=True)
    ap.add_argument("--test-start", required=True)
    ap.add_argument("--test-end", required=True)
    ap.add_argument("--method", default="bars", choices=["bars", "both", "water"])
    ap.add_argument("--ocr-model", default="models/crnn_en.onnx")
    ap.add_argument("--out", default="results/plate")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    z = np.load(a.cache)
    stage, times, paths = z["stage"], z["times"].astype("datetime64[s]"), list(z["paths"])
    ref = cv.imread(a.reference)
    P = plate.find_plate(ref)
    if P is None:
        sys.exit("No staff plate found in the reference frame.")
    json.dump(P, open(os.path.join(a.out, "plate.json"), "w"), indent=1)
    bars = plate.bar_masks(ref, P)
    reg = plate.registration(ref, P)
    print(f"plate: {len(P['bars_t'])} foot bars, perspective fit {P['fit_rmse_ft']:.3f} ft")

    # datum from the printed numbers: majority over daylight frames outside the test window
    win = (times >= np.datetime64(a.test_start)) & (times < np.datetime64(a.test_end))
    cand = [i for i in range(len(paths)) if not win[i] and str(times[i])[11:13] in ("15", "17", "19")]
    datums = []
    for i in cand[::max(1, len(cand) // 20)]:
        img = cv.imread(paths[i])
        Pi = plate.find_plate(img)
        D = plate.read_datum(img, Pi, a.ocr_model) if Pi else None
        if D:
            datums.append(D["datum_ft"])
    vals, counts = np.unique(datums, return_counts=True)
    datum = float(vals[counts.argmax()])
    print(f"OCR datum {datum} ft ({counts.max()} of {len(datums)} frames agree)")

    units = np.full(len(paths), np.nan)
    status = []
    for i, p in enumerate(paths):
        img = cv.imread(p)
        if img is None or np.isnan(stage[i]):
            status.append("skipped")
            continue
        m = plate.read_bars(img, P, bars, reg=reg) if a.method == "bars" else plate.read(img, P, method=a.method)
        status.append(m["status"])
        if m["units"] is not None:
            units[i] = m["units"]

    has = ~np.isnan(units) & ~np.isnan(stage)
    tr, te = has & ~win, has & win
    hi = stage[tr].max()
    # plate mode: stage = offset - units, one parameter (median, robust to bad frames)
    offset = float(np.median(stage[tr] + units[tr]))
    pred_plate = offset - units
    pred_ocr = datum - units          # no gauge data at all
    # free fit on the same waterline positions, for comparison
    lin = calibrate.fit(units[tr], stage[tr], kind="poly", degree=1)
    pred_lin = calibrate.predict(lin, np.nan_to_num(units))
    sets = {"test_all": te, "test_in_range": te & (stage <= hi), "test_above_train_max": te & (stage > hi)}
    report = {"ocr_datum_ft": datum, "ocr_votes": [int(counts.max()), len(datums)],
              "plate_offset_ft": offset, "free_fit_slope": lin["coef"][0], "train_frames": int(tr.sum()),
              "train_max_ft": float(hi), "frames_measured": int(has.sum()), "frames_total": len(paths),
              "status": {k: status.count(k) for k in sorted(set(status))},
              "methods": {name: {k: calibrate.metrics(stage[m], p[m]) for k, m in sets.items() if m.sum()}
                          for name, p in (("plate_ocr_no_gauge", pred_ocr), ("plate_scale_gauge_offset", pred_plate),
                                          ("free_linear", pred_lin))}}
    json.dump(report, open(os.path.join(a.out, "metrics.json"), "w"), indent=2)
    with open(os.path.join(a.out, "series.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_utc", "stage_obs", "units", "pred_ocr", "pred_plate", "pred_linear", "split"])
        for i in range(len(paths)):
            w.writerow([str(times[i]), stage[i], units[i], pred_ocr[i], pred_plate[i], pred_lin[i],
                        "test" if te[i] else "train" if tr[i] else ""])
    for name, v in report["methods"].items():
        print(name, {k: (round(x["rmse"], 2), round(x["mae"], 2), round(x["bias"], 2),
                         round(x["within_0.5"], 2), x["n"]) for k, x in v.items()})
    print("status", report["status"])
    print("offset", round(offset, 2), "free slope", round(lin["coef"][0], 3), "(plate says -1.0)")


if __name__ == "__main__":
    main()
