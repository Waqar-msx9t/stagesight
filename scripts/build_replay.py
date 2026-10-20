"""Package the 29 July 2026 flood replays for the dashboard's replay mode.

For each scenario it writes replay/<scenario>.json (one record per hourly frame: camera reading,
gauge as the agent saw it, true gauge, events with the agent's tool trace, actions) and the reader
overlay of every frame to replay/img/. The API Lambda serves them, so visitors can watch
the agent work even when the river is below the plate.

Usage:  python scripts/build_replay.py [--run results/agent/claude] [--out web/replay]
        (aws/build_package.py ships web/replay inside the Lambda package)
Needs the downloaded archive (data/hancock) for the overlays.
"""
import argparse
import csv
import json
import os
import sys

import cv2 as cv
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from stagesight.reader import SiteReader  # noqa: E402
from stagesight.world import corrupt_frame, plate_band  # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "scripts"))
from replay_agent import SCENARIOS  # noqa: E402

LABELS = {"none": "No fault: the flood as it happened", "flatline": "Gauge frozen for 10 h",
          "drift": "Gauge drifts to +1.5 ft", "offset": "Gauge jumps +1 ft", "dropout": "Gauge silent for 8 h",
          "occlusion": "Debris covers the plate (camera wrong)", "spray": "Spray on the lens (camera wrong)"}


def _rows(path: str) -> list[dict]:
    with open(os.path.join(ROOT, path), newline="") as f:
        return list(csv.DictReader(f))


def _overlay(site: SiteReader, path: str, fault: dict | None, out: str) -> None:
    img = cv.imread(path)
    if fault:
        img = corrupt_frame(img, fault)
    vis = site.draw(img, site.read(img))
    h, w = vis.shape[:2]
    cv.imwrite(out, cv.resize(vis, (540, int(540 * h / w)), interpolation=cv.INTER_AREA),
               [cv.IMWRITE_JPEG_QUALITY, 82])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/agent/claude")
    ap.add_argument("--out", default="web/replay")
    a = ap.parse_args()
    site = SiteReader(os.path.join(ROOT, "sites", "hancock"))
    frames = {r["time_utc"][:19]: os.path.join(ROOT, r["path"].replace("\\", "/")) for r in _rows("data/hancock/images.csv")}
    true = {r["time_utc"][:16]: float(r["stage_ft"]) for r in _rows("data/hancock/gauge.csv")}
    os.makedirs(os.path.join(a.out, "img"), exist_ok=True)
    index = []
    for name, spec in SCENARIOS.items():
        trace = [json.loads(l) for l in open(os.path.join(ROOT, a.run, name, "trace.jsonl"))]
        fault = None
        if spec and spec[0].startswith("camera:"):
            kind, start, hours, band = spec
            t0, t1 = np.datetime64(start), np.datetime64(start) + np.timedelta64(hours, "h")
            fault = (t0, t1, {"kind": kind.split(":")[1], "seed": 1,
                              "polygon": plate_band(site.plate, *band, site.site["datum_ft"]).tolist()})
        recs = []
        for r in trace:
            t = r["time_utc"][:19]
            hit = fault and fault[0] <= np.datetime64(t) < fault[1]
            img = f"img/{name if hit else 'base'}_{t.replace(':', '')}.jpg"
            if not os.path.exists(os.path.join(a.out, img)) and t in frames:
                _overlay(site, frames[t], fault[2] if hit else None, os.path.join(a.out, img))
            hh = t[:14] + ("00" if int(t[14:16]) < 8 else "15")       # nearest earlier 15-min gauge value
            recs.append({"t": t, "img": img, "camera": r["camera"], "gauge_ft": r["gauge_ft"],
                         "true_ft": true.get(hh), "events": r["events"], "actions": r["actions"]})
        json.dump({"scenario": name, "label": LABELS[name], "fault": spec and list(spec), "frames": recs},
                  open(os.path.join(a.out, f"{name}.json"), "w"), default=str)
        index.append({"id": name, "label": LABELS[name]})
        print(name, len(recs), "frames")
    json.dump({"site": site.site["name"], "action_stage_ft": site.site["action_stage_ft"],
               "flood_stage_ft": site.site["flood_stage_ft"], "datum_ft": site.site["datum_ft"],
               "planner": "claude", "scenarios": index}, open(os.path.join(a.out, "index.json"), "w"))


if __name__ == "__main__":
    main()
