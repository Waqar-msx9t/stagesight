"""Real (not synthetic) degraded frames: daylight frames with water on the plate that the reader
did not read cleanly. Runs the reader on every frame with the gauge at or above --min-stage and
writes what it returns, plus a contact sheet of the frames it could not read (other than
below_plate, which is by design near the bottom mark) or read worse than --bad-error.

Usage:
  python scripts/real_faults.py --cache results/hancock_stack.npz --reference data/hancock/reference.jpg \
      --datum-from results/hancock_plate/metrics.json --out results/hancock_real_faults
"""
import argparse
import csv
import json
import os
import sys

import cv2 as cv
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from stagesight import plate  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--datum-from", required=True, help="metrics.json of plate_test.py (OCR datum, no gauge)")
    ap.add_argument("--min-stage", type=float, default=11.0, help="gauge stage that puts water on the plate")
    ap.add_argument("--min-saturation", type=float, default=5.0,
                    help="daylight test: night frames are infrared, mean HSV saturation ~0")
    ap.add_argument("--bad-error", type=float, default=0.5, help="ft; an 'ok' reading worse than this is listed")
    ap.add_argument("--out", default="results/hancock_real_faults")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    z = np.load(a.cache)
    stage, times, paths = z["stage"], z["times"].astype("datetime64[s]"), list(z["paths"])
    datum = json.load(open(a.datum_from))["ocr_datum_ft"]
    ref = cv.imread(a.reference)
    P = plate.find_plate(ref)
    bars, reg = plate.bar_masks(ref, P), plate.registration(ref, P)

    rows, flagged = [], []
    for i in np.flatnonzero(stage >= a.min_stage):
        img = cv.imread(paths[i])
        if cv.cvtColor(img, cv.COLOR_BGR2HSV)[..., 1].mean() < a.min_saturation:
            continue                       # night (infrared, no colour): expected, not a fault
        q = plate.quality_check(img)
        m = plate.read_bars(img, P, bars, reg=reg)
        pred = datum - m["units"] if m["units"] is not None else float("nan")
        err = pred - stage[i]
        rows.append({"time_utc": str(times[i]), "stage_obs": round(float(stage[i]), 3), "status": m["status"],
                     "pred_ocr": round(pred, 3), "error_ft": round(err, 3),
                     "brightness": round(q["brightness"], 1), "sharpness": round(q["sharpness"], 1),
                     "quality_reasons": "|".join(q["reasons"]),
                     "bars_visible": "".join("1" if v else "0" for v in m.get("visible", [])),
                     "split_errors": m.get("errors", ""), "path": paths[i]})
        if m["status"] not in ("ok", "below_plate") or abs(err) > a.bad_error:   # below_plate is by design
            flagged.append((rows[-1], img))

    with open(os.path.join(a.out, "frames.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    status = [r["status"] for r in rows]
    summary = {"min_stage_ft": a.min_stage, "min_saturation": a.min_saturation, "ocr_datum_ft": datum,
               "daylight_frames": len(rows),
               "quality_rejected": sum(bool(r["quality_reasons"]) for r in rows),
               "status": {s: status.count(s) for s in sorted(set(status))},
               "flagged": [{k: r[k] for k in ("time_utc", "stage_obs", "status", "pred_ocr", "error_ft",
                                              "quality_reasons", "bars_visible")} for r, _ in flagged]}
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=2)

    if flagged:   # contact sheet: plate region of each flagged frame, labelled
        x0, y0, x1, y1 = reg["box"]
        tiles = []
        for r, img in flagged:
            t = cv.resize(img[y0:y1, x0:x1], (240, int(240 * (y1 - y0) / (x1 - x0))))
            cv.putText(t, f"{r['time_utc'][5:16]} {r['status']}", (4, 16), cv.FONT_HERSHEY_SIMPLEX, 0.42,
                       (0, 255, 255), 1)
            tiles.append(t)
        while len(tiles) % 6:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(tiles[j:j + 6]) for j in range(0, len(tiles), 6)])
        cv.imwrite(os.path.join(a.out, "flagged.jpg"), sheet)
    print(json.dumps({k: summary[k] for k in ("daylight_frames", "status")}))
    for r in summary["flagged"]:
        print(r)


if __name__ == "__main__":
    main()
