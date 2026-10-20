"""Batch evaluation: measure every image, pair it with USGS gage height, calibrate on the
first part of the record and report errors on the rest.

Usage:
  python scripts/run_batch.py --config config/site.json --index data/images.csv --out results/
images.csv has two columns: path,time_utc  (ISO time, e.g. 2026-09-12T14:00:00)
"""
import argparse
import csv
import json
import os
import sys

import cv2 as cv
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from stagesight import calibrate, usgs, vision  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--gauge-csv", help="optional local gage height CSV (time_utc,stage_ft)")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "overlays"), exist_ok=True)
    cfg = json.load(open(a.config))
    rows = list(csv.DictReader(open(a.index)))
    rows.sort(key=lambda r: r["time_utc"])
    times = np.array([np.datetime64(r["time_utc"], "s") for r in rows])

    if a.gauge_csv:
        g = list(csv.DictReader(open(a.gauge_csv)))
        gt = np.array([np.datetime64(r["time_utc"], "s") for r in g])
        gh = np.array([float(r["stage_ft"]) for r in g])
    else:
        start = str(times[0].astype("datetime64[D]"))
        end = str((times[-1] + np.timedelta64(1, "D")).astype("datetime64[D]"))
        gt, gh = usgs.gage_height(cfg["site_id"], start, end)

    ref = cv.imread(cfg["reference_image"])
    aligner = vision.Aligner(ref) if ref is not None else None
    records = []
    for r, t in zip(rows, times):
        img = cv.imread(r["path"])
        if img is None:
            records.append({"path": r["path"], "time_utc": r["time_utc"], "status": "unreadable"})
            continue
        m = vision.measure(img, cfg["transects"], aligner)
        records.append({"path": r["path"], "time_utc": r["time_utc"], "status": m["status"],
                        "fraction": m["fraction"], "stage_obs": usgs.stage_at(gt, gh, t),
                        "sharpness": m["quality"]["sharpness"],
                        "brightness": m["quality"]["brightness"]})

    usable = [x for x in records if x["status"] == "ok" and x["stage_obs"] is not None]
    if len(usable) < 10:
        sys.exit(f"Only {len(usable)} usable frames; check transects and image quality.")
    n_train = int(len(usable) * cfg.get("train_fraction", 0.7))
    train, test = usable[:n_train], usable[n_train:]
    model = calibrate.fit([x["fraction"] for x in train], [x["stage_obs"] for x in train])
    calibrate.save(model, os.path.join(a.out, "calibration.json"))
    for x in usable:
        x["stage_pred"] = float(calibrate.predict(model, [x["fraction"]])[0])
        x["split"] = "train" if x in train else "test"

    summary = {
        "frames_total": len(records),
        "frames_by_status": {s: sum(1 for x in records if x["status"] == s)
                             for s in sorted({x["status"] for x in records})},
        "train": calibrate.metrics([x["stage_obs"] for x in train], [x["stage_pred"] for x in train]),
        "test": calibrate.metrics([x["stage_obs"] for x in test], [x["stage_pred"] for x in test]),
    }
    json.dump(summary, open(os.path.join(a.out, "metrics.json"), "w"), indent=2)
    keys = ["path", "time_utc", "status", "fraction", "stage_obs", "stage_pred", "split",
            "sharpness", "brightness"]
    with open(os.path.join(a.out, "frames.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(records)

    # Overlays for the five largest test errors (failure gallery) and five typical frames.
    test_sorted = sorted(test, key=lambda x: abs(x["stage_pred"] - x["stage_obs"]))
    picks = test_sorted[-5:] + test_sorted[len(test_sorted) // 2: len(test_sorted) // 2 + 5]
    for x in picks:
        img = cv.imread(x["path"])
        if aligner is not None:
            img, _ = aligner.align(img)
        m = vision.measure(img, cfg["transects"])
        txt = f"camera {x['stage_pred']:.2f} ft | gauge {x['stage_obs']:.2f} ft"
        name = os.path.basename(x["path"]).rsplit(".", 1)[0] + "_overlay.jpg"
        cv.imwrite(os.path.join(a.out, "overlays", name), vision.draw_overlay(img, cfg["transects"], m, txt))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
